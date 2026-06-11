"""
Healthcare Provider Fraud Detection — Full Pipeline (Disk-Space Safe)
Author: Tharun | Sagility Data Science Case Study
Uses: pandas, numpy, scikit-learn, xgboost (no heavy packages needed)
"""

import sys, io
# Force UTF-8 output so emoji print fine on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import pickle
import os
import json

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid', palette='Set2')

BASE = os.path.dirname(os.path.abspath(__file__))
TD   = os.path.join(BASE, "Data", "Training Data") + os.sep
UD   = os.path.join(BASE, "Data", "Unseen Data")   + os.sep

print("=" * 65)
print("  HEALTHCARE PROVIDER FRAUD DETECTION PIPELINE")
print("  Sagility Data Science Case Study — Tharun")
print("=" * 65)

# ─── PHASE 1: LOAD DATA ───────────────────────────────────────────────────────
print("\n📁 Phase 1 — Loading Datasets...")
train_labels = pd.read_csv(TD + "Train-1542865627584.csv")
bene         = pd.read_csv(TD + "Train_Beneficiarydata-1542865627584.csv")
inp          = pd.read_csv(TD + "Train_Inpatientdata-1542865627584.csv")
out          = pd.read_csv(TD + "Train_Outpatientdata-1542865627584.csv")
unseen       = pd.read_csv(UD + "Unseen-1542969243754.csv")
bene_u       = pd.read_csv(UD + "Unseen_Beneficiarydata-1542969243754.csv")
inp_u        = pd.read_csv(UD + "Unseen_Inpatientdata-1542969243754.csv")
out_u        = pd.read_csv(UD + "Unseen_Outpatientdata-1542969243754.csv")

for name, df in [("Train Labels", train_labels), ("Beneficiary(Tr)", bene),
                 ("Inpatient(Tr)",  inp),  ("Outpatient(Tr)", out),
                 ("Unseen Labels",  unseen), ("Bene(Te)",    bene_u),
                 ("Inpatient(Te)", inp_u),  ("Outpatient(Te)", out_u)]:
    print(f"  ✅ {name:18s}: {df.shape[0]:>8,} rows × {df.shape[1]:>2} cols")

fraud_dist = train_labels['PotentialFraud'].value_counts()
print(f"\n  Class Balance : Yes={fraud_dist.get('Yes',0)} | No={fraud_dist.get('No',0)}")
print(f"  Imbalance     : {fraud_dist.get('No',0)/max(fraud_dist.get('Yes',1),1):.1f}:1 (No:Yes)")


# ─── PHASE 2: PREPROCESSING ───────────────────────────────────────────────────
print("\n🔧 Phase 2 — Data Management & Preprocessing...")

def preprocess_bene(df):
    df = df.copy()
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['DOD'] = pd.to_datetime(df['DOD'], errors='coerce')
    ref_date = pd.Timestamp('2009-12-01')
    df['Age'] = ((ref_date - df['DOB']).dt.days / 365).fillna(0).astype(int)
    df['IsDead'] = df['DOD'].notna().astype(int)
    cc_cols = [c for c in df.columns if 'ChronicCond' in c]
    for c in cc_cols:
        df[c] = df[c].map({2: 0, 1: 1}).fillna(0)
    df['ChronicCondCount'] = df[cc_cols].sum(axis=1)
    df['RenalDisease'] = (df['RenalDiseaseIndicator'] == 'Y').astype(int)
    return df

def preprocess_claims(df, claim_type):
    df = df.copy()
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], errors='coerce')
    df['ClaimEndDt']   = pd.to_datetime(df['ClaimEndDt'],   errors='coerce')
    df['ClaimDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days.fillna(0)
    df['ClaimType'] = claim_type
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1,11) if f'ClmDiagnosisCode_{i}' in df.columns]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1,7)  if f'ClmProcedureCode_{i}' in df.columns]
    df['NumDiagCodes']    = df[diag_cols].notna().sum(axis=1)
    df['NumProcCodes']    = df[proc_cols].notna().sum(axis=1)
    df['UniqueDiagCodes'] = df[diag_cols].apply(lambda r: r.dropna().nunique(), axis=1)
    df['UniqueProcCodes'] = df[proc_cols].apply(lambda r: r.dropna().nunique(), axis=1)
    df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(0)
    if 'AdmissionDt' in df.columns:
        df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], errors='coerce')
        df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], errors='coerce')
        df['HospitalStay'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days.fillna(0)
    else:
        df['HospitalStay'] = 0.0
    return df

bene   = preprocess_bene(bene)
bene_u = preprocess_bene(bene_u)
inp    = preprocess_claims(inp,   'Inpatient')
out    = preprocess_claims(out,   'Outpatient')
inp_u  = preprocess_claims(inp_u, 'Inpatient')
out_u  = preprocess_claims(out_u, 'Outpatient')

all_claims   = pd.concat([inp, out],     ignore_index=True)
all_claims_u = pd.concat([inp_u, out_u], ignore_index=True)

train_merged = all_claims.merge(bene,   on='BeneID', how='left')
test_merged  = all_claims_u.merge(bene_u, on='BeneID', how='left')
train_merged = train_merged.merge(train_labels, on='Provider', how='left')

print(f"  Train merged : {train_merged.shape[0]:,} rows × {train_merged.shape[1]} cols")
print(f"  Test  merged : {test_merged.shape[0]:,} rows × {test_merged.shape[1]} cols")

# Missing value report
mv = train_merged.isnull().sum()
mv_pct = (mv / len(train_merged) * 100).round(2)
mv_df = pd.DataFrame({'Count': mv, 'Pct': mv_pct}).query('Count > 0').sort_values('Pct', ascending=False).head(15)
print(f"\n  Top missing columns ({len(mv_df)} cols with nulls):")
for col, row in mv_df.head(5).iterrows():
    print(f"    {col:40s} {row['Pct']:5.1f}%")


# ─── PHASE 3: EDA VISUALIZATIONS ─────────────────────────────────────────────
print("\n🔍 Phase 3 — Exploratory Data Analysis (generating plots)...")

fraud_merged    = train_merged[train_merged['PotentialFraud'] == 'Yes']
nonfraud_merged = train_merged[train_merged['PotentialFraud'] == 'No']

# EDA 1: Fraud label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vc = train_labels['PotentialFraud'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(vc.index, vc.values, color=colors, width=0.5, edgecolor='black')
for bar, val in zip(bars, vc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val}\n({val/vc.sum()*100:.1f}%)', ha='center', fontweight='bold')
axes[0].set_title('Provider Fraud Label Distribution (Training Set)', fontweight='bold')
axes[0].set_xlabel('Fraud Status'); axes[0].set_ylabel('Count')

# EDA 2: Age distribution
axes[1].hist([fraud_merged['Age'].dropna(), nonfraud_merged['Age'].dropna()],
             bins=30, label=['Fraud', 'Non-Fraud'], color=['#e74c3c','#2ecc71'], alpha=0.7)
axes[1].set_title('Patient Age Distribution: Fraud vs Non-Fraud', fontweight='bold')
axes[1].set_xlabel('Age'); axes[1].set_ylabel('Count'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'eda_plot1.png'), dpi=100, bbox_inches='tight')
plt.close()

# EDA 2: Claim amounts & hospital stay
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist([fraud_merged['InscClaimAmtReimbursed'].clip(0, 10000),
              nonfraud_merged['InscClaimAmtReimbursed'].clip(0, 10000)],
             bins=50, label=['Fraud','Non-Fraud'], color=['#e74c3c','#2ecc71'], alpha=0.7)
axes[0].set_title('Insurance Claim Amount Distribution', fontweight='bold')
axes[0].set_xlabel('Claim Amount ($ capped at 10K)'); axes[0].legend()

axes[1].hist([fraud_merged['HospitalStay'].clip(0,40),
              nonfraud_merged['HospitalStay'].clip(0,40)],
             bins=30, label=['Fraud','Non-Fraud'], color=['#e74c3c','#2ecc71'], alpha=0.7)
axes[1].set_title('Hospital Stay Duration', fontweight='bold')
axes[1].set_xlabel('Days'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'eda_plot2.png'), dpi=100, bbox_inches='tight')
plt.close()

# EDA 3: Chronic conditions heatmap
cc_cols = [c for c in train_merged.columns if 'ChronicCond' in c and c != 'ChronicCondCount']
if cc_cols:
    cc_labels = [c.replace('ChronicCond_','') for c in cc_cols]
    fraud_avg    = fraud_merged[cc_cols].mean()
    nonfraud_avg = nonfraud_merged[cc_cols].mean()
    hmap_data = pd.DataFrame({'Fraud': fraud_avg.values, 'Non-Fraud': nonfraud_avg.values}, index=cc_labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(hmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Chronic Condition Prevalence: Fraud vs Non-Fraud', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'eda_plot3.png'), dpi=100, bbox_inches='tight')
    plt.close()

print("  ✅ EDA plots saved (eda_plot1.png, eda_plot2.png, eda_plot3.png)")


# ─── PHASE 4: FEATURE ENGINEERING ────────────────────────────────────────────
print("\n⚙️  Phase 4 — Feature Engineering (53 Provider-Level Features)...")

def engineer_provider_features(merged_df):
    g = merged_df.groupby('Provider')
    feats = pd.DataFrame()
    # Volume
    feats['TotalClaims']            = g['ClaimID'].count()
    feats['InpatientClaims']        = g['ClaimType'].apply(lambda x: (x=='Inpatient').sum())
    feats['OutpatientClaims']       = g['ClaimType'].apply(lambda x: (x=='Outpatient').sum())
    feats['UniqueBeneficiaries']    = g['BeneID'].nunique()
    feats['UniqueAttendPhysicians'] = g['AttendingPhysician'].nunique()
    # Financial
    feats['AvgClaimAmt']            = g['InscClaimAmtReimbursed'].mean()
    feats['TotalReimbursement']     = g['InscClaimAmtReimbursed'].sum()
    feats['MaxClaimAmt']            = g['InscClaimAmtReimbursed'].max()
    feats['StdClaimAmt']            = g['InscClaimAmtReimbursed'].std().fillna(0)
    feats['AvgDeductible']          = g['DeductibleAmtPaid'].mean()
    feats['TotalDeductible']        = g['DeductibleAmtPaid'].sum()
    feats['ReimbursementPerClaim']  = feats['TotalReimbursement'] / (feats['TotalClaims'] + 1)
    feats['DeductibleRatio']        = feats['TotalDeductible'] / (feats['TotalReimbursement'] + 1)
    feats['ReimbPerBeneficiary']    = feats['TotalReimbursement'] / (feats['UniqueBeneficiaries'] + 1)
    feats['ClaimsPerBeneficiary']   = feats['TotalClaims'] / (feats['UniqueBeneficiaries'] + 1)
    feats['InpatientRatio']         = feats['InpatientClaims'] / (feats['TotalClaims'] + 1)
    feats['HighCostClaimRatio']     = g['InscClaimAmtReimbursed'].apply(
                                        lambda x: (x > x.quantile(0.9)).mean() if len(x) > 1 else 0)
    # Temporal
    feats['AvgClaimDuration']       = g['ClaimDuration'].mean()
    feats['AvgHospitalStay']        = g['HospitalStay'].mean()
    feats['TotalHospitalDays']      = g['HospitalStay'].sum()
    _df = merged_df.copy()
    _df['ClaimMonth'] = _df['ClaimStartDt'].dt.month
    feats['MonthlyClaimVariance']   = _df.groupby('Provider')['ClaimMonth'].std().fillna(0)
    feats['PeakMonthClaims']        = _df.groupby('Provider')['ClaimMonth'].apply(
                                        lambda x: x.value_counts().max() if len(x) > 0 else 0)
    # Medical coding
    feats['AvgNumDiagCodes']        = g['NumDiagCodes'].mean()
    feats['AvgNumProcCodes']        = g['NumProcCodes'].mean()
    feats['AvgUniqueDiagCodes']     = g['UniqueDiagCodes'].mean()
    feats['AvgUniqueProcCodes']     = g['UniqueProcCodes'].mean()
    feats['MaxDiagCodes']           = g['NumDiagCodes'].max()
    # Patient demographics
    feats['AvgPatientAge']          = g['Age'].mean()
    feats['MinPatientAge']          = g['Age'].min()
    feats['MaxPatientAge']          = g['Age'].max()
    feats['StdPatientAge']          = g['Age'].std().fillna(0)
    feats['PctDeadPatients']        = g['IsDead'].mean()
    # Chronic disease
    feats['AvgChronicCondCount']    = g['ChronicCondCount'].mean()
    feats['MaxChronicCondCount']    = g['ChronicCondCount'].max()
    feats['PctHighChronicCond']     = g['ChronicCondCount'].apply(lambda x: (x>=4).mean())
    feats['RenalDiseaseRatio']      = g['RenalDisease'].mean()
    for col in ['ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease',
                'ChronicCond_Cancer','ChronicCond_Diabetes','ChronicCond_stroke','ChronicCond_Depression']:
        if col in merged_df.columns:
            feats[f'Avg_{col}'] = merged_df.groupby('Provider')[col].mean()
    # Insurance coverage
    for col, alias in [('IPAnnualReimbursementAmt','AvgIPReimb'),
                       ('OPAnnualReimbursementAmt','AvgOPReimb'),
                       ('IPAnnualDeductibleAmt','AvgIPDeductible'),
                       ('OPAnnualDeductibleAmt','AvgOPDeductible'),
                       ('NoOfMonths_PartACov','AvgPartACovMonths'),
                       ('NoOfMonths_PartBCov','AvgPartBCovMonths')]:
        if col in merged_df.columns:
            feats[alias] = merged_df.groupby('Provider')[col].mean()
    # Physician / behavioral
    feats['ClaimsPerPhysician'] = feats['TotalClaims'] / (feats['UniqueAttendPhysicians'] + 1)
    feats['BenePerPhysician']   = feats['UniqueBeneficiaries'] / (feats['UniqueAttendPhysicians'] + 1)
    bcc = merged_df.groupby(['Provider','BeneID'])['ClaimID'].count().reset_index()
    repeat   = bcc[bcc['ClaimID'] > 1].groupby('Provider')['BeneID'].count()
    all_bene = bcc.groupby('Provider')['BeneID'].count()
    feats['RepeatPatientRatio'] = (repeat / all_bene).fillna(0)
    feats['PhysicianConcentration'] = merged_df.groupby('Provider')['AttendingPhysician'].apply(
        lambda x: (x.value_counts(normalize=True)**2).sum() if len(x) > 0 else 0)
    feats = feats.reset_index()
    return feats

print("  Engineering training features... (this takes ~1-2 min)")
train_feats = engineer_provider_features(train_merged)
print("  Engineering test features...")
test_feats  = engineer_provider_features(test_merged)

train_feats = train_feats.merge(train_labels, on='Provider')
train_feats['FraudLabel'] = (train_feats['PotentialFraud'] == 'Yes').astype(int)

print(f"  Train: {train_feats.shape[0]} providers × {train_feats.shape[1]-3} features")
print(f"  Test : {test_feats.shape[0]} providers × {test_feats.shape[1]-1} features")


# ─── PHASE 5: FEATURE SELECTION ───────────────────────────────────────────────
print("\n🎯 Phase 5 — Feature Selection (MI + RF Importance)...")
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier as RFC

DROP_COLS    = ['Provider', 'PotentialFraud', 'FraudLabel']
feature_cols = [c for c in train_feats.columns if c not in DROP_COLS]
for c in feature_cols:
    if c not in test_feats.columns:
        test_feats[c] = 0

X = train_feats[feature_cols].fillna(0).astype(float)
y = train_feats['FraudLabel']

print("  Computing Mutual Information...")
mi     = mutual_info_classif(X, y, random_state=42)
mi_ser = pd.Series(mi, index=feature_cols).sort_values(ascending=False)

print("  Computing Random Forest Importance (100 trees)...")
rf_sel = RFC(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X, y)
rf_imp = pd.Series(rf_sel.feature_importances_, index=feature_cols).sort_values(ascending=False)

combined     = (mi_ser.rank() + rf_imp.rank()) / 2
top_features = combined.sort_values(ascending=False).head(35).index.tolist()

X_sel      = X[top_features]
X_test_sel = test_feats[top_features].fillna(0).astype(float)

print(f"  ✅ {len(top_features)} features selected")
print(f"  Top 10 by combined rank:")
for i, f in enumerate(top_features[:10], 1):
    print(f"    {i:2d}. {f}")

# Feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
mi_ser.head(20).sort_values().plot.barh(ax=axes[0], color='#3498db', alpha=0.8)
axes[0].set_title('Mutual Information Score (Top 20)', fontweight='bold')
rf_imp.head(20).sort_values().plot.barh(ax=axes[1], color='#e67e22', alpha=0.8)
axes[1].set_title('Random Forest Feature Importance (Top 20)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()
print("  ✅ Saved feature_importance.png")


# ─── PHASE 6: MODEL TRAINING ─────────────────────────────────────────────────
print("\n🤖 Phase 6 — Model Training (5-Fold Stratified CV with Class Weights)...")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, average_precision_score,
                             roc_curve, confusion_matrix)
import xgboost as xgb

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
spw = float(y.value_counts()[0]) / float(y.value_counts()[1])

models_dict = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, C=1.0),
    'Random Forest (300)': RFC(
        n_estimators=300, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric='auc',
        random_state=42, verbosity=0, tree_method='hist'),
}

results_dict = {}
oof_probs    = {}

for name, model in models_dict.items():
    print(f"\n  ▶ Training {name}...")
    oof_pred = np.zeros(len(y))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_sel, y)):
        X_tr, X_val = X_sel.iloc[tr_idx], X_sel.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"    Fold {fold+1}/5 done", end=' ', flush=True)
    print()

    pred_class = (oof_pred >= 0.5).astype(int)
    results_dict[name] = dict(
        ROC_AUC   = round(roc_auc_score(y, oof_pred), 4),
        PR_AUC    = round(average_precision_score(y, oof_pred), 4),
        F1        = round(f1_score(y, pred_class), 4),
        Precision = round(precision_score(y, pred_class, zero_division=0), 4),
        Recall    = round(recall_score(y, pred_class), 4),
        Accuracy  = round(accuracy_score(y, pred_class), 4),
    )
    oof_probs[name] = oof_pred
    r = results_dict[name]
    print(f"    → ROC-AUC={r['ROC_AUC']}  PR-AUC={r['PR_AUC']}  F1={r['F1']}  Recall={r['Recall']}")

res_df = pd.DataFrame(results_dict).T.sort_values('ROC_AUC', ascending=False)
print("\n  ===== MODEL COMPARISON =====")
print(res_df.to_string())
res_df.to_csv(os.path.join(BASE, "model_results.csv"))

# ROC Curve plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#e74c3c', '#2ecc71', '#3498db']
for (name, prob), color in zip(oof_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, lw=2)
axes[0].plot([0,1],[0,1],'k--', alpha=0.5)
axes[0].set_title('ROC Curves — All Models', fontweight='bold')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

res_df[['ROC_AUC','PR_AUC','F1','Recall']].plot(kind='bar', ax=axes[1],
    colormap='Set2', alpha=0.85, edgecolor='black')
axes[1].set_title('Model Performance Comparison', fontweight='bold')
axes[1].set_ylabel('Score'); axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'model_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()
print("\n  ✅ Saved model_comparison.png")

# Confusion matrix for best model
best_model_name = res_df.index[0]
best_oof        = oof_probs[best_model_name]
best_pred       = (best_oof >= 0.5).astype(int)
cm = confusion_matrix(y, best_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pd.DataFrame(cm, index=['Legit','Fraud'], columns=['Pred Legit','Pred Fraud']),
            annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()
print("  ✅ Saved confusion_matrix.png")


# ─── PHASE 7: FEATURE IMPORTANCE (as SHAP proxy) ─────────────────────────────
print("\n🔬 Phase 7 — Feature Importance Analysis...")
best_model = models_dict[best_model_name]

if 'Random Forest' in best_model_name:
    shap_imp = pd.Series(best_model.feature_importances_, index=top_features).sort_values(ascending=False)
elif 'XGBoost' in best_model_name:
    shap_imp = pd.Series(best_model.feature_importances_, index=top_features).sort_values(ascending=False)
else:
    shap_imp = rf_imp.reindex(top_features).fillna(0).sort_values(ascending=False)

shap_imp.to_csv(os.path.join(BASE, "shap_importance.csv"), header=False)

fig, ax = plt.subplots(figsize=(10, 7))
shap_imp.head(20).sort_values().plot.barh(ax=ax, color='#8e44ad', alpha=0.85)
ax.set_title(f'Feature Importance — {best_model_name}', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'shap_plot.png'), dpi=100, bbox_inches='tight')
plt.close()

print(f"  Top 10 Important Features ({best_model_name}):")
for i, (feat, val) in enumerate(shap_imp.head(10).items(), 1):
    print(f"    {i:2d}. {feat:<35} Score={val:.4f}")


# ─── PHASE 8: PREDICTIONS & SUBMISSION ───────────────────────────────────────
print("\n📤 Phase 8 — Generating Test Predictions & Submission...")

# Retrain best model on FULL training data
print(f"  Retraining {best_model_name} on full training set...")
best_model.fit(X_sel, y)

test_proba = best_model.predict_proba(X_test_sel)[:, 1]
test_class = (test_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    'Provider':        test_feats['Provider'],
    'Probability':     test_proba.round(4),
    'Predicted_Class': ['Yes' if c == 1 else 'No' for c in test_class],
})

fraud_cnt = (submission['Predicted_Class'] == 'Yes').sum()

for fname in ["Tharun_Submission.csv", "Anupriya_Submission.csv"]:
    submission.to_csv(os.path.join(BASE, fname), index=False)

print(f"  Test providers  : {len(submission)}")
print(f"  Flagged fraud   : {fraud_cnt} ({fraud_cnt/len(submission)*100:.1f}%)")
print(f"  Prob min/max    : {test_proba.min():.4f} / {test_proba.max():.4f}")

# Prediction distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
vc = submission['Predicted_Class'].value_counts()
axes[0].pie(vc.values, labels=vc.index, colors=['#2ecc71','#e74c3c'],
            autopct='%1.1f%%', startangle=90)
axes[0].set_title('Prediction Distribution (Test Set)', fontweight='bold')
axes[1].hist(test_proba, bins=40, color='#3498db', alpha=0.8, edgecolor='black')
axes[1].axvline(0.5, color='red', linestyle='--', lw=2, label='Threshold=0.5')
axes[1].set_xlabel('Fraud Probability'); axes[1].set_ylabel('Count')
axes[1].set_title('Probability Distribution', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'submission_dist.png'), dpi=100, bbox_inches='tight')
plt.close()
print("  ✅ Saved submission_dist.png")


# ─── SAVE MODEL ARTIFACTS ────────────────────────────────────────────────────
print("\n💾 Saving Model Artifacts...")
with open(os.path.join(BASE, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(BASE, "top_features.pkl"), "wb") as f:
    pickle.dump(top_features, f)
print(f"  ✅ best_model.pkl  ({best_model_name})")
print(f"  ✅ top_features.pkl ({len(top_features)} features)")

# Save summary JSON for Streamlit
summary = {
    'best_model': best_model_name,
    'metrics':    results_dict[best_model_name],
    'all_results': {k: v for k, v in results_dict.items()},
    'top_features': top_features,
    'fraud_count':  int(fraud_cnt),
    'total_test':   int(len(submission)),
    'fraud_rate':   float(round(fraud_cnt/len(submission)*100, 2)),
}
with open(os.path.join(BASE, "pipeline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  ✅ pipeline_summary.json")


# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
r = results_dict[best_model_name]
print("\n" + "=" * 65)
print("  ✅ PIPELINE COMPLETE — ALL DELIVERABLES READY")
print("=" * 65)
print(f"\n  🏆 Best Model   : {best_model_name}")
print(f"     ROC-AUC      : {r['ROC_AUC']:.4f}")
print(f"     PR-AUC       : {r['PR_AUC']:.4f}")
print(f"     F1 Score     : {r['F1']:.4f}")
print(f"     Recall       : {r['Recall']:.4f}")
print(f"     Precision    : {r['Precision']:.4f}")
print(f"     Accuracy     : {r['Accuracy']:.4f}")
print(f"\n  📋 Submission    : Tharun_Submission.csv | Anupriya_Submission.csv")
print(f"     Test Fraud   : {fraud_cnt}/{len(submission)} ({fraud_cnt/len(submission)*100:.1f}%)")
print(f"\n  📦 Artifacts     : best_model.pkl, top_features.pkl")
print(f"                   : model_results.csv, shap_importance.csv")
print(f"                   : pipeline_summary.json")
print(f"\n  🖼  Plots         : eda_plot1.png, eda_plot2.png, eda_plot3.png")
print(f"                   : feature_importance.png, model_comparison.png")
print(f"                   : confusion_matrix.png, shap_plot.png, submission_dist.png")
print(f"\n  🚀 Launch App    : streamlit run streamlit_app.py")
print("=" * 65)
