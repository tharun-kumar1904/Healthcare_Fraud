"""
Healthcare Provider Fraud Detection — Full Pipeline (GPU Accelerated)
Author: Tharun | Sagility Data Science Case Study
Features: Advanced FE, Optuna, Stacking, Threshold Optimization
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib  # pyrefly: ignore [missing-import]
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
import seaborn as sns  # pyrefly: ignore [missing-import]
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
print("  HEALTHCARE PROVIDER FRAUD DETECTION PIPELINE (GPU)")
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

# ─── PHASE 4: FEATURE ENGINEERING ────────────────────────────────────────────
print("\n⚙️  Phase 4 — Feature Engineering (Advanced Features)...")

def engineer_provider_features(merged_df):
    merged_df = merged_df.copy()
    merged_df['ClaimStartDt'] = pd.to_datetime(merged_df['ClaimStartDt'], errors='coerce')
    merged_df['ClaimEndDt']   = pd.to_datetime(merged_df['ClaimEndDt'],   errors='coerce')
    
    # Pre-calculate IsSharedPatient before groupby
    patient_provider_count = merged_df.groupby('BeneID')['Provider'].nunique()
    shared = patient_provider_count[patient_provider_count > 3].index
    merged_df['IsSharedPatient'] = merged_df['BeneID'].isin(shared).astype(int)

    g = merged_df.groupby('Provider')
    feats = pd.DataFrame()
    
    # 1. Volume
    feats['TotalClaims']            = g['ClaimID'].count()
    feats['InpatientClaims']        = (merged_df['ClaimType'] == 'Inpatient').astype(int).groupby(merged_df['Provider']).sum()
    feats['OutpatientClaims']       = (merged_df['ClaimType'] == 'Outpatient').astype(int).groupby(merged_df['Provider']).sum()
    feats['UniqueBeneficiaries']    = g['BeneID'].nunique()
    feats['UniqueAttendPhysicians'] = g['AttendingPhysician'].nunique()
    
    # 2. Financial
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
    
    provider_90th = g['InscClaimAmtReimbursed'].quantile(0.9)
    merged_temp = merged_df[['Provider', 'InscClaimAmtReimbursed']].merge(provider_90th.rename('Amt_90th'), on='Provider', how='left')
    is_high_cost = (merged_temp['InscClaimAmtReimbursed'] > merged_temp['Amt_90th']).astype(int)
    feats['HighCostClaimRatio']     = is_high_cost.groupby(merged_temp['Provider']).mean()
    
    # 3. Temporal
    feats['AvgClaimDuration']       = g['ClaimDuration'].mean()
    feats['AvgHospitalStay']        = g['HospitalStay'].mean()
    feats['TotalHospitalDays']      = g['HospitalStay'].sum()
    
    # 4. Medical coding
    feats['AvgNumDiagCodes']        = g['NumDiagCodes'].mean()
    feats['AvgNumProcCodes']        = g['NumProcCodes'].mean()
    feats['AvgUniqueDiagCodes']     = g['UniqueDiagCodes'].mean()
    feats['AvgUniqueProcCodes']     = g['UniqueProcCodes'].mean()
    feats['MaxDiagCodes']           = g['NumDiagCodes'].max()
    
    # 5. Patient demographics
    feats['AvgPatientAge']          = g['Age'].mean()
    feats['MinPatientAge']          = g['Age'].min()
    feats['MaxPatientAge']          = g['Age'].max()
    feats['StdPatientAge']          = g['Age'].std().fillna(0)
    feats['PctDeadPatients']        = g['IsDead'].mean()
    
    # 6. Chronic disease
    feats['AvgChronicCondCount']    = g['ChronicCondCount'].mean()
    feats['MaxChronicCondCount']    = g['ChronicCondCount'].max()
    feats['PctHighChronicCond']     = (merged_df['ChronicCondCount'] >= 4).astype(int).groupby(merged_df['Provider']).mean()
    feats['RenalDiseaseRatio']      = g['RenalDisease'].mean()
    for col in ['ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease',
                'ChronicCond_Cancer','ChronicCond_Diabetes','ChronicCond_stroke','ChronicCond_Depression']:
        if col in merged_df.columns:
            feats[f'Avg_{col}'] = g[col].mean()
            
    # 7. Advanced Features Requested
    feats['ClaimAmt_Skewness'] = g['InscClaimAmtReimbursed'].skew().fillna(0)
    feats['ClaimAmt_Kurtosis'] = g['InscClaimAmtReimbursed'].apply(lambda x: x.kurtosis() if len(x) > 3 else 0).fillna(0)
    feats['ClaimAmt_CV'] = (g['InscClaimAmtReimbursed'].std() / (g['InscClaimAmtReimbursed'].mean() + 1e-9)).fillna(0)
    feats['SharedPatientRatio'] = g['IsSharedPatient'].mean()
    feats['PctMaxDiagCodes'] = g['NumDiagCodes'].apply(lambda x: (x == 10).mean())
    
    # 8. Physician / behavioral
    bcc = merged_df.groupby(['Provider','BeneID'])['ClaimID'].count().reset_index()
    repeat   = bcc[bcc['ClaimID'] > 1].groupby('Provider')['BeneID'].count()
    all_bene = bcc.groupby('Provider')['BeneID'].count()
    feats['RepeatPatientRatio'] = (repeat / all_bene).fillna(0)

    # 9. Missing Insurance features
    feats['AvgIPReimb']      = g['IPAnnualReimbursementAmt'].mean()
    feats['AvgOPReimb']      = g['OPAnnualReimbursementAmt'].mean()
    feats['AvgIPDeductible'] = g['IPAnnualDeductibleAmt'].mean()
    feats['AvgOPDeductible'] = g['OPAnnualDeductibleAmt'].mean()
    feats['AvgPartACovMonths'] = g['NoOfMonths_PartACov'].mean()
    feats['AvgPartBCovMonths'] = g['NoOfMonths_PartBCov'].mean()
    
    # 10. Attending physician concentration and ratios
    feats['ClaimsPerPhysician'] = feats['TotalClaims'] / (feats['UniqueAttendPhysicians'] + 1)
    feats['BenePerPhysician']   = feats['UniqueBeneficiaries'] / (feats['UniqueAttendPhysicians'] + 1)
    feats['PhysicianConcentration'] = g['AttendingPhysician'].apply(
        lambda x: (x.value_counts(normalize=True)**2).sum() if len(x) > 0 else 0)
    
    feats = feats.reset_index()
    return feats

print("  Engineering training features... (this takes ~1-2 min)")
train_feats = engineer_provider_features(train_merged)
print("  Engineering test features...")
test_feats  = engineer_provider_features(test_merged)

train_feats = train_feats.merge(train_labels, on='Provider')
train_feats['FraudLabel'] = (train_feats['PotentialFraud'] == 'Yes').astype(int)

# ─── PHASE 5: FEATURE SELECTION ───────────────────────────────────────────────
print("\n🎯 Phase 5 — Feature Selection (MI + RF Importance)...")
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split

DROP_COLS    = ['Provider', 'PotentialFraud', 'FraudLabel']
feature_cols = [c for c in train_feats.columns if c not in DROP_COLS]
for c in feature_cols:
    if c not in test_feats.columns:
        test_feats[c] = 0

# Split into 90% training and 10% holdout stratified by target
train_cv_feats, holdout_feats = train_test_split(train_feats, test_size=0.10, stratify=train_feats['FraudLabel'], random_state=42)
print(f"  Split data: CV Train={train_cv_feats.shape[0]:,} providers | Holdout={holdout_feats.shape[0]:,} providers")

X_train_cv = train_cv_feats[feature_cols].fillna(0).astype(float)
y_train_cv = train_cv_feats['FraudLabel']
X_holdout  = holdout_feats[feature_cols].fillna(0).astype(float)
y_holdout  = holdout_feats['FraudLabel']

X_all = train_feats[feature_cols].fillna(0).astype(float)
y_all = train_feats['FraudLabel']

print("  Computing Mutual Information on CV Train...")
mi     = mutual_info_classif(X_train_cv, y_train_cv, random_state=42)
mi_ser = pd.Series(mi, index=feature_cols).sort_values(ascending=False)

print("  Computing Random Forest Importance on CV Train (100 trees)...")
rf_sel = RFC(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_train_cv, y_train_cv)
rf_imp = pd.Series(rf_sel.feature_importances_, index=feature_cols).sort_values(ascending=False)

combined     = (mi_ser.rank() + rf_imp.rank()) / 2
top_features = combined.sort_values(ascending=False).head(35).index.tolist()

X_train_cv_sel = X_train_cv[top_features]
X_holdout_sel  = X_holdout[top_features]
X_all_sel      = X_all[top_features]
X_test_sel     = test_feats[top_features].fillna(0).astype(float)

print(f"  ✅ {len(top_features)} features selected")

# ─── PHASE 6: MODEL TRAINING (GPU ACCELERATED & OPTUNA) ──────────────────────
print("\n🤖 Phase 6 — GPU Model Training & Optuna Tuning...")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

fraud_ratio = float((y_all == 0).sum()) / (y_all == 1).sum()

device_val = 'cuda'
try:
    import numpy as np
    dummy_model = xgb.XGBClassifier(device='cuda')
    dummy_model.fit(np.zeros((2, 2)), np.zeros(2))
    print("  ✅ CUDA verified for XGBoost fitting. GPU acceleration active.")
except Exception as e:
    device_val = 'cpu'
    print(f"  ⚠️ CUDA check failed: {e}. Falling back to CPU.")

print("  ▶ Running Optuna on XGBoost (GPU) - 100 trials...")
def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 800),
        'max_depth':         trial.suggest_int('max_depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 2.0),
        'scale_pos_weight':  fraud_ratio,
        'tree_method':       'hist',
        'device':            device_val,
        'eval_metric':       'aucpr',
        'random_state':      42,
        'verbosity':         0
    }
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(
        model, X_train_cv_sel, y_train_cv, cv=5,
        scoring='roc_auc', n_jobs=1
    ).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"  ✅ Best XGBoost ROC-AUC: {study.best_value:.4f}")

print("\n  ▶ Training Stacking Classifier (5-Fold CV)...")

estimators = [
    ('xgb', xgb.XGBClassifier(
        **study.best_params,
        scale_pos_weight=fraud_ratio,
        tree_method='hist', device=device_val,
        eval_metric='aucpr', random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMClassifier(
        boosting_type='dart',
        class_weight='balanced', n_estimators=500,
        learning_rate=0.05, max_depth=6,
        drop_rate=0.1, max_drop=50, skip_drop=0.5,
        random_state=42, verbose=-1)),
    ('cat', CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced',
        random_seed=42, verbose=0)),
]

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    n_jobs=1
)

from sklearn.model_selection import cross_val_predict
print("  Generating Out-of-Fold predictions for Threshold Optimization...")
oof_probs = cross_val_predict(stack_clf, X_train_cv_sel, y_train_cv, cv=5, method='predict_proba')[:, 1]

# ─── PHASE 7: THRESHOLD OPTIMIZATION ─────────────────────────────────────────
print("\n⚙️  Phase 7 — Threshold Optimization (F1 & F2-Score)...")
precisions, recalls, thresholds = precision_recall_curve(y_train_cv, oof_probs)

# F1 Optimization
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
optimal_idx_f1 = np.argmax(f1_scores)
optimal_threshold_f1 = thresholds[optimal_idx_f1] if optimal_idx_f1 < len(thresholds) else 0.5
print(f"  ✅ Optimal F1 Threshold: {optimal_threshold_f1:.3f} | Best OOF F1: {f1_scores[optimal_idx_f1]:.4f}")

# F2 Optimization
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
optimal_idx_f2 = np.argmax(f2_scores)
optimal_threshold_f2 = thresholds[optimal_idx_f2] if optimal_idx_f2 < len(thresholds) else 0.5
print(f"  ✅ Optimal F2 Threshold: {optimal_threshold_f2:.3f} | Best OOF F2: {f2_scores[optimal_idx_f2]:.4f}")

# Train final model on 90% training data to predict on holdout
print("\n  Training Stacking Classifier on 90% CV dataset...")
stack_clf.fit(X_train_cv_sel, y_train_cv)

print("  Predicting on Holdout set...")
holdout_probs = stack_clf.predict_proba(X_holdout_sel)[:, 1]

# Calculate holdout predictions
holdout_pred_f1 = (holdout_probs >= optimal_threshold_f1).astype(int)
holdout_pred_f2 = (holdout_probs >= optimal_threshold_f2).astype(int)

print("\n  ===== MODEL CV PERFORMANCE (F1 OPTIMAL) =====")
oof_pred_f1 = (oof_probs >= optimal_threshold_f1).astype(int)
cv_metrics_f1 = {
    'ROC_AUC': float(roc_auc_score(y_train_cv, oof_probs)),
    'PR_AUC': float(average_precision_score(y_train_cv, oof_probs)),
    'F1': float(f1_score(y_train_cv, oof_pred_f1)),
    'Recall': float(recall_score(y_train_cv, oof_pred_f1)),
    'Precision': float(precision_score(y_train_cv, oof_pred_f1)),
    'Accuracy': float(accuracy_score(y_train_cv, oof_pred_f1))
}
for k, v in cv_metrics_f1.items():
    print(f"  CV {k:10s}: {v:.4f}")

print("\n  ===== MODEL HOLDOUT PERFORMANCE (F1 OPTIMAL) =====")
holdout_metrics_f1 = {
    'ROC_AUC': float(roc_auc_score(y_holdout, holdout_probs)),
    'PR_AUC': float(average_precision_score(y_holdout, holdout_probs)),
    'F1': float(f1_score(y_holdout, holdout_pred_f1)),
    'Recall': float(recall_score(y_holdout, holdout_pred_f1)),
    'Precision': float(precision_score(y_holdout, holdout_pred_f1)),
    'Accuracy': float(accuracy_score(y_holdout, holdout_pred_f1))
}
for k, v in holdout_metrics_f1.items():
    print(f"  Holdout {k:10s}: {v:.4f}")

cv_metrics_f2 = {
    'F1': float(f1_score(y_train_cv, (oof_probs >= optimal_threshold_f2).astype(int))),
    'Recall': float(recall_score(y_train_cv, (oof_probs >= optimal_threshold_f2).astype(int))),
    'Precision': float(precision_score(y_train_cv, (oof_probs >= optimal_threshold_f2).astype(int))),
    'Accuracy': float(accuracy_score(y_train_cv, (oof_probs >= optimal_threshold_f2).astype(int)))
}
holdout_metrics_f2 = {
    'F1': float(f1_score(y_holdout, holdout_pred_f2)),
    'Recall': float(recall_score(y_holdout, holdout_pred_f2)),
    'Precision': float(precision_score(y_holdout, holdout_pred_f2)),
    'Accuracy': float(accuracy_score(y_holdout, holdout_pred_f2))
}

# Train final stack on all data (100% training data) for test predictions
print("\n  Training Final Stacking Model on ALL (100%) training data...")
stack_clf.fit(X_all_sel, y_all)

# ─── PHASE 8: PREDICTIONS & ARTIFACTS ────────────────────────────────────────
print("\n📤 Phase 8 — Generating Test Predictions & Saving...")

test_proba = stack_clf.predict_proba(X_test_sel)[:, 1]
test_class = (test_proba >= optimal_threshold_f1).astype(int)

submission = pd.DataFrame({
    'Provider':        test_feats['Provider'],
    'Probability':     test_proba.round(4),
    'Predicted_Class': ['Yes' if c == 1 else 'No' for c in test_class],
})

fraud_cnt = (submission['Predicted_Class'] == 'Yes').sum()
submission.to_csv(os.path.join(BASE, "Tharun Kumar V_Submission.csv"), index=False)
print(f"  ✅ Test providers flagged as fraud: {fraud_cnt}")

# Save models
with open(os.path.join(BASE, "best_model.pkl"), "wb") as f:
    pickle.dump(stack_clf, f)
with open(os.path.join(BASE, "top_features.pkl"), "wb") as f:
    pickle.dump(top_features, f)

# Save validation data predictions for Streamlit
oof_df = pd.DataFrame({
    'Provider': train_cv_feats['Provider'],
    'Actual_Label': y_train_cv,
    'Predicted_Probability': oof_probs
})
oof_df.to_csv(os.path.join(BASE, "oof_predictions.csv"), index=False)

holdout_df = pd.DataFrame({
    'Provider': holdout_feats['Provider'],
    'Actual_Label': y_holdout,
    'Predicted_Probability': holdout_probs
})
holdout_df.to_csv(os.path.join(BASE, "holdout_predictions.csv"), index=False)

# Save the full provider feats for dynamic EDA
provider_eda = train_feats.copy()
provider_eda.to_csv(os.path.join(BASE, "provider_eda_summary.csv"), index=False)
print("  ✅ Saved oof_predictions.csv, holdout_predictions.csv, and provider_eda_summary.csv")

# Dynamic Feature Importance from final Random Forest selection
rf_sel.fit(X_all_sel, y_all)
importance_list = rf_sel.feature_importances_.tolist()
feature_importance_dict = dict(zip(top_features, importance_list))

summary = {
    'best_model': 'Stacking Ensemble (XGB, LGBM, CatBoost)',
    'optimal_threshold_f1': float(optimal_threshold_f1),
    'optimal_threshold_f2': float(optimal_threshold_f2),
    'cv_metrics_f1': cv_metrics_f1,
    'holdout_metrics_f1': holdout_metrics_f1,
    'cv_metrics_f2': cv_metrics_f2,
    'holdout_metrics_f2': holdout_metrics_f2,
    'top_features': top_features,
    'feature_importances': feature_importance_dict,
    'fraud_count':  int(fraud_cnt),
    'total_test':   int(len(submission)),
    'fraud_rate':   float(round(fraud_cnt/len(submission)*100, 2)),
}
with open(os.path.join(BASE, "pipeline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("  ✅ PIPELINE COMPLETE — ADVANCED STRATEGIES APPLIED")
print("=" * 65)
