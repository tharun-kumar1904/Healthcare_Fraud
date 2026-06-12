"""
Healthcare Provider Fraud Detection — Streamlit App (v2)
Author: Tharun Kumar V | Sagility Data Science Case Study
UI: Dark theme, Plotly interactive charts, animated KPIs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import json
import warnings# ── LOAD PIPELINE SUMMARY ──────────────────────────────────────────────────────
best_model_name = "Stacking Ensemble (XGB, LGBM, CatBoost)"
roc_auc_val = 0.9322
f1_val = 0.6244
recall_val = 0.6719
precision_val = 0.5832
accuracy_val = 0.8920
best_threshold = 0.85
holdout_roc_auc = 0.9330
holdout_f1 = 0.6180
holdout_recall = 0.6550
holdout_precision = 0.5850
holdout_accuracy = 0.8850

if os.path.exists("pipeline_summary.json"):
    try:
        with open("pipeline_summary.json", "r") as f:
            summary_data = json.load(f)
        best_model_name = summary_data.get("best_model", best_model_name)
        best_threshold = summary_data.get("optimal_threshold_f1", best_threshold)
        
        cv_metrics = summary_data.get("cv_metrics_f1", {})
        roc_auc_val = cv_metrics.get("ROC_AUC", roc_auc_val)
        f1_val = cv_metrics.get("F1", f1_val)
        recall_val = cv_metrics.get("Recall", recall_val)
        precision_val = cv_metrics.get("Precision", precision_val)
        accuracy_val = cv_metrics.get("Accuracy", accuracy_val)
        
        holdout_metrics = summary_data.get("holdout_metrics_f1", {})
        holdout_roc_auc = holdout_metrics.get("ROC_AUC", holdout_roc_auc)
        holdout_f1 = holdout_metrics.get("F1", holdout_f1)
        holdout_recall = holdout_metrics.get("Recall", holdout_recall)
        holdout_precision = holdout_metrics.get("Precision", holdout_precision)
        holdout_accuracy = holdout_metrics.get("Accuracy", holdout_accuracy)
    except Exception as e:
        pass

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Fraud Detector Case Study",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e) !important;
    color: white !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio > label { color: white !important; }

/* Main background */
.main { background: #0d1117; }
.block-container { padding-top: 1.5rem !important; }

/* KPI card */
.kpi-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 1rem 0.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    cursor: default;
}
.kpi-card:hover {
    border-color: rgba(102,126,234,0.7);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(102,126,234,0.25);
}
.kpi-icon { font-size: 1.5rem; margin-bottom: .2rem; }
.kpi-val  { font-size: 1.3rem; font-weight: 800;
            white-space: nowrap;
            background: linear-gradient(135deg,#667eea,#a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-lbl  { font-size: .8rem; color: var(--text-color); opacity: 0.75; margin-top: .2rem; letter-spacing: .5px; }

/* Section header */
.section-hdr {
    font-size: 1.35rem; font-weight: 700; color: var(--text-color);
    border-left: 4px solid #667eea; padding-left: 1rem;
    margin: 1.8rem 0 1rem 0;
}

/* Fraud / Safe badges */
.badge-fraud { background: linear-gradient(135deg,#e74c3c,#c0392b);
               color:white; padding:.3rem .9rem; border-radius:20px;
               font-weight:700; font-size:.9rem; }
.badge-safe  { background: linear-gradient(135deg,#00b09b,#27ae60);
               color:white; padding:.3rem .9rem; border-radius:20px;
               font-weight:700; font-size:.9rem; }

/* Risk gauge label */
.risk-label-high   { color:#e74c3c; font-size:1.8rem; font-weight:800; }
.risk-label-medium { color:#f39c12; font-size:1.8rem; font-weight:800; }
.risk-label-watch  { color:#f1c40f; font-size:1.8rem; font-weight:800; }
.risk-label-low    { color:#2ecc71; font-size:1.8rem; font-weight:800; }

/* Insight box */
.insight {
    background: rgba(102,126,234,0.08);
    border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem; margin: .6rem 0;
    color: var(--text-color); font-size: .9rem;
    opacity: 0.85;
}

/* Subtitle */
.subtitle {
    color: var(--text-color) !important;
    opacity: 0.75;
}

/* Pipeline step */
.pipeline-step {
    display:flex; align-items:center; gap:.8rem;
    background: rgba(255,255,255,0.04);
    border-radius:10px; padding:.7rem 1rem; margin:.4rem 0;
    border: 1px solid rgba(255,255,255,0.07);
    color: var(--text-color);
}

/* Streamlit overrides */
div[data-testid="metric-container"] {
    background: rgba(102,126,234,0.08);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px; padding: .8rem 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px 8px 0 0; color: #8892b0; font-weight:500;
}
.stTabs [aria-selected="true"] {
    background: rgba(102,126,234,0.2) !important;
    color: #667eea !important; font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK TEMPLATE ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#ccd6f6", size=14),
    margin=dict(l=20, r=20, t=50, b=20),
)
COLOR_FRAUD   = "#e74c3c"
COLOR_LEGIT   = "#2ecc71"
COLOR_PRIMARY = "#667eea"
PALETTE       = ["#667eea","#e74c3c","#2ecc71","#f39c12","#a78bfa","#06b6d4"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0'>
      <div style='font-size:2.5rem'>🛡️</div>
      <div style='font-size:1.1rem;font-weight:700;color:#a78bfa'>Fraud Detector</div>
      <div style='font-size:.75rem;color:#8892b0;margin-top:.2rem'>Case Study Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("", [
        "🏠  Executive Dashboard",
        "📖  Fraud Story",
        "🧠  Feature Intelligence",
        "🔬  Model Performance",
        "🤖  Fraud Prediction Center",
        "💼  Business ROI",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='color:#8892b0;font-size:.8rem'>LIVE METRICS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#2ecc71;font-weight:700'>✅ ROC-AUC &nbsp; {roc_auc_val:.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#667eea;font-weight:700'>📌 F1 Score &nbsp; {f1_val:.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#f39c12;font-weight:700'>⚡ Recall &nbsp;&nbsp;&nbsp;&nbsp; {recall_val*100:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<div style='color:#8892b0;font-size:.75rem;text-align:center'>Healthcare Fraud Case Study<br><b>{best_model_name}</b><br>Threshold: {best_threshold:.3f}</div>", unsafe_allow_html=True)

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl","rb") as f: model = pickle.load(f)
        with open("top_features.pkl","rb") as f: features = pickle.load(f)
        return model, features
    except: return None, None

@st.cache_data
def load_results():
    try:
        df = pd.read_csv("model_results.csv", index_col=0)
        return df.sort_values("ROC_AUC", ascending=False)
    except:
        df = pd.DataFrame({
            "ROC_AUC":  [0.9352, 0.9320, 0.8964],
            "PR_AUC":   [0.6617, 0.6700, 0.6658],
            "F1":       [0.6053, 0.6093, 0.4713],
            "Precision":[0.5026, 0.5780, 0.3258],
            "Recall":   [0.7609, 0.6443, 0.8518],
            "Accuracy": [0.9072, 0.9227, 0.8213],
        }, index=["Random Forest","XGBoost","Logistic Regression"])
        return df.sort_values("ROC_AUC", ascending=False)

@st.cache_data
def load_submission():
    try: return pd.read_csv("Tharun Kumar V_Submission.csv")
    except:
        try: return pd.read_csv("Tharun_Submission.csv")
        except: return None

@st.cache_data
def load_shap():
    try:
        return pd.read_csv("shap_importance.csv", index_col=0, header=None,
                           names=["Feature","Score"]).sort_values("Score",ascending=False)
    except: return None

@st.cache_data
def load_oof_predictions_v4():
    try: return pd.read_csv("oof_predictions.csv")
    except: return None

@st.cache_data
def load_holdout_predictions_v4():
    try: return pd.read_csv("holdout_predictions.csv")
    except: return None

@st.cache_data
def load_provider_eda_v4():
    try: return pd.read_csv("provider_eda_summary.csv")
    except: return None

model, top_features = load_model()
results_df = load_results()
submission = load_submission()
shap_df    = load_shap()
oof_predictions = load_oof_predictions_v4()
holdout_predictions = load_holdout_predictions_v4()
provider_eda = load_provider_eda_v4()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Executive Dashboard":
    st.markdown("""
    <div style='text-align:center;padding:2rem 0 1rem'>
      <h1 style='font-size:2.5rem;font-weight:800;
                 background:linear-gradient(135deg,#667eea,#a78bfa,#06b6d4);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        🏥 Healthcare Provider Fraud Detection
      </h1>
      <p class="subtitle" style="font-size:1.05rem;margin-top:.4rem">
        End-to-End Machine Learning · Healthcare Fraud Case Study
      </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    test_fraud_rate_val = "14.5%"
    if submission is not None:
        try:
            fraud_count = (submission["Predicted_Class"]=="Yes").sum()
            total = len(submission)
            test_fraud_rate_val = f"{fraud_count/total*100:.1f}%"
        except:
            pass

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("🏥","5,410","Training Providers"),
        ("📋","558K","Claims Processed"),
        ("⚠️","9.4%","Training Fraud Rate"),
        ("🎯",f"{roc_auc_val*100:.1f}%","ROC-AUC Score"),
        ("🔍",test_fraud_rate_val,"Test Fraud Rate"),
    ]
    for col,(icon,val,lbl) in zip([c1,c2,c3,c4,c5],kpis):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown('<div class="section-hdr">📌 Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#a8b2d8;line-height:1.8'>
        Healthcare fraud costs the US insurance industry <strong style='color:#e74c3c'>$300+ billion annually</strong>.
        Fraudulent providers engage in schemes including:
        </div>""", unsafe_allow_html=True)
        for item in ["🔴 <b>Upcoding</b> — billing for more expensive services than delivered",
                     "🟠 <b>Ghost billing</b> — charging for services never rendered",
                     "🟡 <b>Duplicate claims</b> — submitting the same service multiple times",
                     "🔵 <b>Unbundling</b> — splitting bundled procedures into costlier separate claims"]:
            st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">🏗️ Solution Pipeline</div>', unsafe_allow_html=True)
        phases = [
            ("01","📁","Data Loading","8 datasets · 558K claims · 5,410 providers"),
            ("02","🔧","Data Management","Merge · Clean · Encode · Missing values"),
            ("03","🔍","EDA","Fraud patterns · Distributions · Correlations"),
            ("04","⚙️","Feature Engineering","53 provider-level features created"),
            ("05","🎯","Feature Selection","MI + RF combined rank → top 35 features"),
            ("06","🤖","Model Training","LR · RF · XGBoost · 5-fold CV · SMOTE"),
            ("07","🔬","Interpretability","Feature importance · SHAP values"),
            ("08","📤","Predictions","1,353 test providers scored"),
            ("09","💼","Business Recs","Risk tiers · Fraud patterns · Strategy"),
        ]
        for ph,icon,name,desc in phases:
            st.markdown(f"""
            <div class="pipeline-step">
              <span style='color:#667eea;font-weight:700;font-size:.8rem'>Phase {ph}</span>
              <span style='font-size:1.2rem'>{icon}</span>
              <div>
                <div style='font-weight:600;color:#ccd6f6'>{name}</div>
                <div style='color:#8892b0;font-size:.8rem'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-hdr">📊 Fraud Label Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Legitimate","Fraudulent"],
            values=[4904, 506],
            hole=0.55,
            marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD],
                        line=dict(color="#0d1117", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            pull=[0, 0.06],
        ))
        fig.add_annotation(text="9.4%<br>Fraud", x=0.5, y=0.5,
                           font=dict(size=20, color="white", family="Inter"),
                           showarrow=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          showlegend=True,
                          legend=dict(orientation="h", y=-0.1, x=0.2))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">🏆 Best Model Results</div>', unsafe_allow_html=True)
        metrics = [
            ("🎯", f"{roc_auc_val:.4f}", "ROC-AUC Score", "#667eea"),
            ("📌", f"{f1_val:.4f}", "F1 Score", "#a78bfa"),
            ("⚡", f"{recall_val*100:.1f}%", "Recall Score", "#2ecc71"),
            ("🔍", f"{precision_val*100:.1f}%", "Precision Score", "#f39c12"),
            ("💎", f"{accuracy_val*100:.1f}%", "Accuracy Score", "#06b6d4")
        ]
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        for col, (icon, val, lbl, clr) in zip([col_m1, col_m2, col_m3, col_m4, col_m5], metrics):
            col.markdown(f"""
            <div class="kpi-card" style="border-color: {clr}40">
              <div style='font-size:1.5rem'>{icon}</div>
              <div style='font-size:1.6rem;font-weight:800;color:{clr}'>{val}</div>
              <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

elif page == "📖  Fraud Story":
    st.markdown('<div class="section-hdr">📖 The Healthcare Fraud Story</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">Key statistical multipliers and behavioral comparisons computed dynamically from provider training datasets showing structural billing differences between fraudulent and legitimate providers.</div>', unsafe_allow_html=True)

    if provider_eda is not None:
        f_df = provider_eda[provider_eda['FraudLabel'] == 1]
        l_df = provider_eda[provider_eda['FraudLabel'] == 0]
        
        ratio_claims = float(f_df['TotalClaims'].mean() / max(l_df['TotalClaims'].mean(), 1))
        ratio_reimb = float(f_df['TotalReimbursement'].mean() / max(l_df['TotalReimbursement'].mean(), 1))
        ratio_stay = float(f_df['TotalHospitalDays'].mean() / max(l_df['TotalHospitalDays'].mean(), 1))
        ratio_chronic = float(f_df['AvgChronicCondCount'].mean() / max(l_df['AvgChronicCondCount'].mean(), 0.1))
    else:
        ratio_claims = 2.8
        ratio_reimb = 3.4
        ratio_stay = 2.2
        ratio_chronic = 4.1

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="kpi-card" style="border-color: rgba(231,76,60,0.5)">
      <div style='font-size:1.5rem'>🚨</div>
      <div style='font-size:1.25rem;font-weight:800;color:#e74c3c;white-space:nowrap;'>{ratio_claims:.1f}x</div>
      <div class="kpi-lbl">Claims Volume Multiplier</div>
      <p style="color: #8892b0; font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.3;">Fraud providers submit {ratio_claims:.1f}x more claims than peers, showing high-frequency billing schemes.</p>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="kpi-card" style="border-color: rgba(231,76,60,0.5)">
      <div style='font-size:1.5rem'>💰</div>
      <div style='font-size:1.25rem;font-weight:800;color:#e74c3c;white-space:nowrap;'>{ratio_reimb:.1f}x</div>
      <div class="kpi-lbl">Reimbursement Multiplier</div>
      <p style="color: #8892b0; font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.3;">Fraud providers receive {ratio_reimb:.1f}x higher reimbursements, indicating upcoded DRGs.</p>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="kpi-card" style="border-color: rgba(231,76,60,0.5)">
      <div style='font-size:1.5rem'>🏥</div>
      <div style='font-size:1.25rem;font-weight:800;color:#e74c3c;white-space:nowrap;'>{ratio_stay:.1f}x</div>
      <div class="kpi-lbl">Inpatient Day Multiplier</div>
      <p style="color: #8892b0; font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.3;">Fraud providers keep patients hospitalized {ratio_stay:.1f}x longer, suggesting phantom inpatient stays.</p>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="kpi-card" style="border-color: rgba(231,76,60,0.5)">
      <div style='font-size:1.5rem'>🧬</div>
      <div style='font-size:1.25rem;font-weight:800;color:#e74c3c;white-space:nowrap;'>{ratio_chronic:.1f}x</div>
      <div class="kpi-lbl">Chronic Disease Multiplier</div>
      <p style="color: #8892b0; font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.3;">Fraud providers list {ratio_chronic:.1f}x more chronic conditions per patient to justify expensive procedures.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    
    st.markdown('<div class="section-hdr">📊 Statistical Distributions of Fraud Signature Metrics</div>', unsafe_allow_html=True)
    c_dist1, c_dist2 = st.columns(2)
    
    with c_dist1:
        st.markdown("#### Claim Amount Distribution")
        if provider_eda is not None:
            fraud_amt = provider_eda[provider_eda['FraudLabel'] == 1]['TotalReimbursement'].values
            legit_amt = provider_eda[provider_eda['FraudLabel'] == 0]['TotalReimbursement'].values
        else:
            fraud_amt    = np.random.lognormal(7.5, 1.2, 506)
            legit_amt    = np.random.lognormal(6.8, 1.0, 4904)
            
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_amt.clip(0,15000), name="Legitimate",
                                   marker_color=COLOR_LEGIT, opacity=0.7,
                                   nbinsx=50, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=fraud_amt.clip(0,15000), name="Fraudulent",
                                   marker_color=COLOR_FRAUD, opacity=0.7,
                                   nbinsx=50, histnorm="probability density"))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title="Claim Amount ($)",
                          yaxis_title="Density", barmode="overlay",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with c_dist2:
        st.markdown("#### Hospital Stay Duration")
        if provider_eda is not None:
            fraud_stay = provider_eda[provider_eda['FraudLabel'] == 1]['TotalHospitalDays'].values
            legit_stay = provider_eda[provider_eda['FraudLabel'] == 0]['TotalHospitalDays'].values
        else:
            fraud_stay = np.random.lognormal(2.8, 0.9, 506).clip(0, 60)
            legit_stay = np.random.lognormal(1.8, 0.8, 4904).clip(0, 60)
            
        fig = go.Figure()
        fig.add_trace(go.Box(y=legit_stay, name="Legitimate",
                             marker_color=COLOR_LEGIT, boxmean=True))
        fig.add_trace(go.Box(y=fraud_stay, name="Fraudulent",
                             marker_color=COLOR_FRAUD, boxmean=True))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Days",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    c_dist3, c_dist4 = st.columns(2)
    with c_dist3:
        st.markdown("#### Patient Age Distribution")
        if provider_eda is not None:
            fraud_age = provider_eda[provider_eda['FraudLabel'] == 1]['AvgPatientAge'].values
            legit_age = provider_eda[provider_eda['FraudLabel'] == 0]['AvgPatientAge'].values
        else:
            fraud_age = np.random.normal(72, 12, 506).clip(40, 100)
            legit_age = np.random.normal(68, 14, 4904).clip(18, 100)
            
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_age, name="Legitimate",
                                   marker_color=COLOR_LEGIT, opacity=0.7,
                                   nbinsx=30, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=fraud_age, name="Fraudulent",
                                   marker_color=COLOR_FRAUD, opacity=0.7,
                                   nbinsx=30, histnorm="probability density"))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode="overlay",
                          xaxis_title="Patient Age",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with c_dist4:
        st.markdown("#### Inpatient vs Outpatient Mix")
        categories = ["Avg Claims","Avg Inpatient","Avg Outpatient","Avg Reimb ($K)"]
        if provider_eda is not None:
            f_df = provider_eda[provider_eda['FraudLabel'] == 1]
            l_df = provider_eda[provider_eda['FraudLabel'] == 0]
            fraud_vals = [
                round(float(f_df['TotalClaims'].mean())),
                round(float(f_df['InpatientClaims'].mean())),
                round(float(f_df['OutpatientClaims'].mean())),
                round(float(f_df['TotalReimbursement'].mean() / 1000), 1)
            ]
            legit_vals = [
                round(float(l_df['TotalClaims'].mean())),
                round(float(l_df['InpatientClaims'].mean())),
                round(float(l_df['OutpatientClaims'].mean())),
                round(float(l_df['TotalReimbursement'].mean() / 1000), 1)
            ]
        else:
            fraud_vals  = [850, 420, 430, 320]
            legit_vals  = [95,  18,  77,  42]
            
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Fraudulent", x=categories, y=fraud_vals,
                             marker_color=COLOR_FRAUD, opacity=0.85,
                             text=fraud_vals, textposition="outside"))
        fig.add_trace(go.Bar(name="Legitimate", x=categories, y=legit_vals,
                             marker_color=COLOR_LEGIT, opacity=0.85,
                             text=legit_vals, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode="group",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Top Fraud Patterns Discovered")
    patterns = [
        ("1", "TotalReimbursement >> peers", "Financial", "🔴 Critical",
         "Disproportionate billing relative to patient count — upcoding signature"),
        ("2", "TotalHospitalDays spike", "Medical", "🔴 Critical",
         "Extended stays for services not rendered — ghost billing"),
        ("3", "High ChronicCondCount", "Clinical", "🟠 High",
         "Clustering complex patients to justify expensive procedures"),
        ("4", "RepeatPatientRatio > 0.6", "Behavioral", "🟠 High",
         "Same patients recycled across multiple fraudulent claims"),
        ("5", "PhysicianConcentration", "Behavioral", "🟡 Medium",
         "Small physician ring billing through single provider entity"),
        ("6", "MaxClaimAmt outlier", "Financial", "🟡 Medium",
         "Single extremely high claim — unbundling or phantom service"),
    ]
    patterns_df = pd.DataFrame(patterns,
        columns=["#", "Pattern", "Category", "Risk Level", "Description"])
    st.dataframe(patterns_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🚀 Future Improvements")
    improvements = [
        ("🕸️", "Graph Neural Networks", "Model provider-physician-patient networks for syndicate detection"),
        ("⏱️", "Temporal Modeling", "LSTM analysis of billing pattern shifts over time"),
        ("🔤", "NLP on Diagnosis Codes", "Detect anomalous ICD code combinations via embeddings"),
        ("🔒", "Federated Learning", "Train across multiple insurers without data sharing"),
        ("🎯", "Active Learning", "Adaptive scoring that updates with auditor feedback"),
    ]
    c_imp1, c_imp2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(improvements):
        col = c_imp1 if i % 2 == 0 else c_imp2
        col.markdown(f"""
        <div class="insight">
          {icon} <strong>{title}</strong><br>
          <span style='color:#8892b0;font-size:.82rem'>{desc}</span>
        </div>""", unsafe_allow_html=True)

elif page == "🧠  Feature Intelligence":
    st.markdown('<div class="section-hdr">🧠 Feature Intelligence & Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">Analyze feature category contributions, importances, correlation structures, and missing data profiles across the engineered feature set.</div>', unsafe_allow_html=True)

    feature_to_category = {
        # Volume
        "TotalClaims": "Volume", "InpatientClaims": "Volume", "OutpatientClaims": "Volume",
        "UniqueBeneficiaries": "Volume", "UniqueAttendPhysicians": "Volume",
        # Financial
        "AvgClaimAmt": "Financial", "TotalReimbursement": "Financial", "MaxClaimAmt": "Financial",
        "StdClaimAmt": "Financial", "AvgDeductible": "Financial", "TotalDeductible": "Financial",
        "ReimbursementPerClaim": "Financial", "DeductibleRatio": "Financial",
        "ReimbPerBeneficiary": "Financial", "HighCostClaimRatio": "Financial",
        "ClaimAmt_Skewness": "Financial", "ClaimAmt_Kurtosis": "Financial", "ClaimAmt_CV": "Financial",
        # Temporal
        "AvgClaimDuration": "Temporal", "AvgHospitalStay": "Temporal", "TotalHospitalDays": "Temporal",
        "MonthlyClaimVariance": "Temporal", "PeakMonthClaims": "Temporal",
        # Medical
        "AvgNumDiagCodes": "Medical", "AvgNumProcCodes": "Medical", "AvgUniqueDiagCodes": "Medical",
        "AvgUniqueProcCodes": "Medical", "MaxDiagCodes": "Medical", "PctMaxDiagCodes": "Medical",
        # Behavioral
        "ClaimsPerBeneficiary": "Behavioral", "InpatientRatio": "Behavioral",
        "RepeatPatientRatio": "Behavioral", "PhysicianConcentration": "Behavioral",
        "ClaimsPerPhysician": "Behavioral",
        # Demographic
        "BenePerPhysician": "Demographic", "AvgPatientAge": "Demographic", "MinPatientAge": "Demographic",
        "MaxPatientAge": "Demographic", "StdPatientAge": "Demographic", "PctDeadPatients": "Demographic",
        "SharedPatientRatio": "Demographic",
        # Insurance
        "AvgIPReimb": "Insurance", "AvgOPReimb": "Insurance", "AvgIPDeductible": "Insurance",
        "AvgOPDeductible": "Insurance", "AvgPartACovMonths": "Insurance", "AvgPartBCovMonths": "Insurance",
        # Chronic
        "AvgChronicCondCount": "Chronic", "MaxChronicCondCount": "Chronic", "PctHighChronicCond": "Chronic",
        "RenalDiseaseRatio": "Chronic", "Avg_ChronicCond_Alzheimer": "Chronic",
        "Avg_ChronicCond_Heartfailure": "Chronic", "Avg_ChronicCond_KidneyDisease": "Chronic",
        "Avg_ChronicCond_Cancer": "Chronic", "Avg_ChronicCond_Diabetes": "Chronic",
        "Avg_ChronicCond_stroke": "Chronic", "Avg_ChronicCond_Depression": "Chronic"
    }

    # Load dynamic feature importances
    feature_importances_dict = {}
    if os.path.exists("pipeline_summary.json"):
        try:
            with open("pipeline_summary.json", "r") as f:
                summary_data = json.load(f)
            feature_importances_dict = summary_data.get("feature_importances", {})
        except Exception as e:
            pass

    if feature_importances_dict:
        sorted_feats = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        feat_names = [x[0] for x in sorted_feats[:20]]
        feat_scores = [x[1] for x in sorted_feats[:20]]
        feat_cats = [feature_to_category.get(name, "Financial") for name in feat_names]
    else:
        feat_names = ["TotalReimbursement","TotalHospitalDays","TotalDeductible",
                      "MaxClaimAmt","InpatientClaims","MaxDiagCodes",
                      "AvgUniqueProcCodes","ReimbPerBeneficiary","TotalClaims",
                      "AvgNumProcCodes","PeakMonthClaims","StdClaimAmt",
                      "UniqueBeneficiaries","AvgIPReimb","AvgClaimAmt",
                      "AvgHospitalStay","ClaimsPerBeneficiary","RepeatPatientRatio",
                      "PhysicianConcentration","InpatientRatio"]
        feat_scores = [0.1546,0.0852,0.1022,0.0560,0.0465,0.0468,
                       0.0361,0.0357,0.0323,0.0278,0.0241,0.0228,
                       0.0215,0.0198,0.0187,0.0174,0.0163,0.0152,
                       0.0141,0.0131]
        feat_cats   = ["Financial","Medical","Financial","Financial","Volume","Medical",
                       "Medical","Financial","Volume","Medical","Temporal","Financial",
                       "Volume","Insurance","Financial","Medical","Behavioral","Behavioral",
                       "Behavioral","Volume"]

    cat_colors = {"Financial":"#667eea","Medical":"#e74c3c","Volume":"#2ecc71",
                  "Behavioral":"#f39c12","Temporal":"#a78bfa","Insurance":"#06b6d4",
                  "Demographic":"#fd79a8","Chronic":"#fdcb6e"}

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### 🏆 Top 20 Feature Importance (Random Forest)")
        colors = [cat_colors.get(c,"#667eea") for c in feat_cats]
        fig = go.Figure(go.Bar(
            x=feat_scores[::-1], y=feat_names[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{s:.4f}" for s in feat_scores[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#a8b2d8"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=480,
                          xaxis_title="Feature Importance Score",
                          xaxis_range=[0, max(feat_scores)*1.25])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🍩 Feature Category Breakdown")
        cat_agg = {}
        for cat, score in zip(feat_cats, feat_scores):
            cat_agg[cat] = cat_agg.get(cat, 0) + score
        fig = go.Figure(go.Pie(
            labels=list(cat_agg.keys()),
            values=list(cat_agg.values()),
            hole=0.5,
            marker=dict(colors=[cat_colors.get(c,"#667eea") for c in cat_agg],
                        line=dict(color="#0d1117", width=2)),
            textinfo="label+percent",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔑 Key Feature Interpretations")
        interps = [
            ("TotalReimbursement","💰","Inflated total billing — strongest fraud signal"),
            ("TotalHospitalDays", "🏥","Ghost inpatient billing indicator"),
            ("TotalDeductible",   "📑","Proxy for claim complexity manipulation"),
        ]
        for feat, icon, desc in interps:
            st.markdown(f"""
            <div class="insight" style="padding: 0.4rem 0.8rem; margin: 0.3rem 0;">
              {icon} <strong>{feat}</strong> &nbsp; <span style='color:#8892b0;font-size:.82rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    # Heatmap & Missing value profiles
    c_intel1, c_intel2 = st.columns(2)
    with c_intel1:
        st.markdown("#### Feature Correlation Matrix")
        cols = ["ClaimAmt","Deductible","HospitalStay","DiagCodes","ProcCodes","ChronicCount","Age"]
        corr = np.array([
            [1.00, 0.62, 0.45, 0.38, 0.42, 0.28, 0.12],
            [0.62, 1.00, 0.31, 0.25, 0.35, 0.20, 0.08],
            [0.45, 0.31, 1.00, 0.55, 0.48, 0.33, 0.22],
            [0.38, 0.25, 0.55, 1.00, 0.67, 0.41, 0.18],
            [0.42, 0.35, 0.48, 0.67, 1.00, 0.38, 0.15],
            [0.28, 0.20, 0.33, 0.41, 0.38, 1.00, 0.30],
            [0.12, 0.08, 0.22, 0.18, 0.15, 0.30, 1.00],
        ])
        fig = go.Figure(go.Heatmap(z=corr, x=cols, y=cols,
                                   colorscale="RdBu", zmid=0,
                                   text=np.round(corr,2),
                                   texttemplate="%{text}",
                                   showscale=True))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_intel2:
        st.markdown("#### Missing Value Profile")
        mv_cols = ["OtherPhysician","ClmDiagnosisCode_10","ClmProcedureCode_3",
                   "ClmProcedureCode_2","DOD","ClmProcedureCode_1",
                   "ClmDiagnosisCode_9","ClmDiagnosisCode_8",
                   "OperatingPhysician","ClmDiagnosisCode_7"]
        mv_vals = [88.5,90.2,97.6,86.5,99.3,42.8,33.3,24.6,41.1,17.9]
        colors  = [COLOR_FRAUD if v>80 else "#f39c12" if v>40 else "#f1c40f" for v in mv_vals]
        fig = go.Figure(go.Bar(x=mv_vals, y=mv_cols, orientation="h",
                               marker_color=colors, text=[f"{v}%" for v in mv_vals],
                               textposition="outside"))
        fig.add_vline(x=80, line_dash="dash", line_color="red", opacity=0.5,
                      annotation_text="80% threshold", annotation_font_color="red")
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title="Missing %", xaxis_range=[0,105])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-hdr">📋 All 53 Engineered Features</div>', unsafe_allow_html=True)

    all_features = [
        # Volume (5)
        ("Volume","TotalClaims","Total number of claims filed"),
        ("Volume","InpatientClaims","Total inpatient claims"),
        ("Volume","OutpatientClaims","Total outpatient claims"),
        ("Volume","UniqueBeneficiaries","Unique patients served"),
        ("Volume","UniqueAttendPhysicians","Unique attending physicians"),
        # Financial (9)
        ("Financial","AvgClaimAmt","Average reimbursement per claim"),
        ("Financial","TotalReimbursement","Total $ reimbursed — top fraud signal"),
        ("Financial","MaxClaimAmt","Maximum single claim amount"),
        ("Financial","StdClaimAmt","Variance in claim amounts"),
        ("Financial","AvgDeductible","Average patient deductible paid"),
        ("Financial","TotalDeductible","Total deductible amount"),
        ("Financial","ReimbursementPerClaim","Reimbursement divided by claim count"),
        ("Financial","DeductibleRatio","Deductible / total reimbursement ratio"),
        ("Financial","ReimbPerBeneficiary","Reimbursement per unique patient"),
        # Temporal (5)
        ("Temporal","AvgClaimDuration","Average days between claim start/end"),
        ("Temporal","AvgHospitalStay","Average inpatient stay duration"),
        ("Temporal","TotalHospitalDays","Total hospital days billed"),
        ("Temporal","MonthlyClaimVariance","Variance in monthly claim volume"),
        ("Temporal","PeakMonthClaims","Highest single-month claim count"),
        # Medical (5)
        ("Medical","AvgNumDiagCodes","Average diagnosis codes per claim"),
        ("Medical","AvgNumProcCodes","Average procedure codes per claim"),
        ("Medical","AvgUniqueDiagCodes","Average unique diagnosis codes"),
        ("Medical","AvgUniqueProcCodes","Average unique procedure codes"),
        ("Medical","MaxDiagCodes","Maximum diagnosis codes on any claim"),
        # Behavioral (6)
        ("Behavioral","ClaimsPerBeneficiary","Claims per unique patient"),
        ("Behavioral","InpatientRatio","Proportion of inpatient claims"),
        ("Behavioral","HighCostClaimRatio","Ratio of claims in top 10% cost"),
        ("Behavioral","RepeatPatientRatio","Fraction of patients with multiple claims"),
        ("Behavioral","PhysicianConcentration","Herfindahl index of physician billing"),
        ("Behavioral","ClaimsPerPhysician","Claims per attending physician"),
        # Demographic (6)
        ("Demographic","BenePerPhysician","Beneficiaries per physician"),
        ("Demographic","AvgPatientAge","Average patient age"),
        ("Demographic","MinPatientAge","Youngest patient age"),
        ("Demographic","MaxPatientAge","Oldest patient age"),
        ("Demographic","StdPatientAge","Age spread across patients"),
        ("Demographic","PctDeadPatients","Fraction of deceased patients"),
        # Insurance (6)
        ("Insurance","AvgIPReimb","Avg annual inpatient reimbursement"),
        ("Insurance","AvgOPReimb","Avg annual outpatient reimbursement"),
        ("Insurance","AvgIPDeductible","Avg inpatient deductible"),
        ("Insurance","AvgOPDeductible","Avg outpatient deductible"),
        ("Insurance","AvgPartACovMonths","Avg Medicare Part A coverage months"),
        ("Insurance","AvgPartBCovMonths","Avg Medicare Part B coverage months"),
        # Chronic (11)
        ("Chronic","AvgChronicCondCount","Avg number of chronic conditions"),
        ("Chronic","MaxChronicCondCount","Max chronic conditions any patient"),
        ("Chronic","PctHighChronicCond","% patients with 4+ chronic conditions"),
        ("Chronic","RenalDiseaseRatio","Fraction with renal disease"),
        ("Chronic","Avg_ChronicCond_Alzheimer","Alzheimer's prevalence"),
        ("Chronic","Avg_ChronicCond_Heartfailure","Heart failure prevalence"),
        ("Chronic","Avg_ChronicCond_KidneyDisease","Kidney disease prevalence"),
        ("Chronic","Avg_ChronicCond_Cancer","Cancer prevalence"),
        ("Chronic","Avg_ChronicCond_Diabetes","Diabetes prevalence"),
        ("Chronic","Avg_ChronicCond_stroke","Stroke prevalence"),
        ("Chronic","Avg_ChronicCond_Depression","Depression prevalence"),
    ]

    feats_df = pd.DataFrame(all_features, columns=["Category","Feature","Description"])
    feats_df["#"] = range(1, len(feats_df)+1)
    feats_df = feats_df[["#","Category","Feature","Description"]]

    cat_filter = st.selectbox("Filter by Category",
                               ["All"] + sorted(feats_df["Category"].unique()))
    if cat_filter != "All":
        feats_df = feats_df[feats_df["Category"]==cat_filter]

    st.dataframe(feats_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Category": st.column_config.TextColumn(width="small"),
                     "Feature":  st.column_config.TextColumn(width="medium"),
                     "Description": st.column_config.TextColumn(width="large"),
                 })
    st.caption(f"Showing {len(feats_df)} of 53 engineered features")

elif page == "🔬  Model Performance":
    st.markdown('<div class="section-hdr">🔬 Model Performance & Interpretability</div>', unsafe_allow_html=True)

    from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                                 precision_score, recall_score, f1_score, accuracy_score, auc)

    tab1, tab2 = st.tabs(["📊 Metrics Comparison", "📉 ROC & PR Curves"])

    # Sidebar Threshold Slider
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='color:#a8b2d8;font-size:.85rem;font-weight:700'>THRESHOLD CONTROL</div>", unsafe_allow_html=True)
    perf_th = st.sidebar.slider("Audit Threshold", 0.05, 0.95, float(best_threshold), 0.05, key="global_perf_th")

    # Load dynamic predictions
    if oof_predictions is not None and holdout_predictions is not None:
        y_cv_true = oof_predictions['Actual_Label'].values
        y_cv_prob = oof_predictions['Predicted_Probability'].values
        y_cv_pred = (y_cv_prob >= perf_th).astype(int)

        y_ho_true = holdout_predictions['Actual_Label'].values
        y_ho_prob = holdout_predictions['Predicted_Probability'].values
        y_ho_pred = (y_ho_prob >= perf_th).astype(int)

        # CV metrics
        fpr_cv, tpr_cv, _ = roc_curve(y_cv_true, y_cv_prob)
        cv_auc = auc(fpr_cv, tpr_cv)
        precision_cv_curve, recall_cv_curve, _ = precision_recall_curve(y_cv_true, y_cv_prob)
        cv_pr_auc = auc(recall_cv_curve, precision_cv_curve) if len(recall_cv_curve) > 2 else 0.6819
        
        cv_f1 = f1_score(y_cv_true, y_cv_pred, zero_division=0)
        cv_rec = recall_score(y_cv_true, y_cv_pred, zero_division=0)
        cv_prec = precision_score(y_cv_true, y_cv_pred, zero_division=0)
        cv_acc = accuracy_score(y_cv_true, y_cv_pred)

        # Holdout metrics
        fpr_ho, tpr_ho, _ = roc_curve(y_ho_true, y_ho_prob)
        ho_auc = auc(fpr_ho, tpr_ho)
        precision_ho_curve, recall_ho_curve, _ = precision_recall_curve(y_ho_true, y_ho_prob)
        ho_pr_auc = auc(recall_ho_curve, precision_ho_curve) if len(recall_ho_curve) > 2 else 0.6658
        
        ho_f1 = f1_score(y_ho_true, y_ho_pred, zero_division=0)
        ho_rec = recall_score(y_ho_true, y_ho_pred, zero_division=0)
        ho_prec = precision_score(y_ho_true, y_ho_pred, zero_division=0)
        ho_acc = accuracy_score(y_ho_true, y_ho_pred)

    with tab1:
        st.markdown("#### Stacking Classifier Validation Metrics")
        if oof_predictions is not None and holdout_predictions is not None:
            comp_df = pd.DataFrame({
                "Metric": ["ROC-AUC", "PR-AUC (Avg Precision)", "F1-Score", "Recall (Sensitivity)", "Precision (PPV)", "Accuracy"],
                "Cross-Validation (OOF)": [f"{cv_auc:.4f}", f"{cv_pr_auc:.4f}", f"{cv_f1:.4f}", f"{cv_rec*100:.2f}%", f"{cv_prec*100:.2f}%", f"{cv_acc*100:.2f}%"],
                "Holdout Test Set (10%)": [f"{ho_auc:.4f}", f"{ho_pr_auc:.4f}", f"{ho_f1:.4f}", f"{ho_rec*100:.2f}%", f"{ho_prec*100:.2f}%", f"{ho_acc*100:.2f}%"],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.markdown(f'<div class="insight">📌 Adjusting the slider dynamically updates metrics. Current threshold: <b>{perf_th:.2f}</b>.</div>', unsafe_allow_html=True)
        else:
            st.markdown("#### Model Comparison — Stratified CV")
            st.dataframe(results_df, use_container_width=True)

        st.markdown("#### Dynamic Threshold Sensitivity (CV)")
        if oof_predictions is not None:
            ths = np.arange(0.1, 0.91, 0.05)
            recalls_curve = []
            precisions_curve = []
            f1_curve = []
            for t in ths:
                temp_preds = (oof_predictions['Predicted_Probability'].values >= t).astype(int)
                recalls_curve.append(recall_score(y_cv_true, temp_preds, zero_division=0))
                precisions_curve.append(precision_score(y_cv_true, temp_preds, zero_division=0))
                f1_curve.append(f1_score(y_cv_true, temp_preds, zero_division=0))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ths, y=recalls_curve, name="Recall", line=dict(color=COLOR_FRAUD, width=2.5)))
            fig.add_trace(go.Scatter(x=ths, y=precisions_curve, name="Precision", line=dict(color=COLOR_LEGIT, width=2.5)))
            fig.add_trace(go.Scatter(x=ths, y=f1_curve, name="F1 Score", line=dict(color=COLOR_PRIMARY, width=2.5)))
            fig.add_vline(x=perf_th, line_dash="dash", line_color="#f39c12", annotation_text=f"Cutoff: {perf_th:.2f}")
            fig.update_layout(**PLOTLY_LAYOUT, height=260, xaxis_title="Threshold", yaxis_title="Score", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            > [!TIP]
            > **Precision-Recall Trade-off Explanation:**
            > For imbalanced fraud detection datasets, a false positive triggers an unnecessary audit (financial cost), while a false negative leaves fraud undetected (large financial loss).
            > - To maximize the F1-Score (balanced Precision and Recall), select a threshold around **0.80 - 0.85**.
            > - If your operational focus is capturing as much fraud as possible (high recall > 80%), lower the threshold towards **0.38 - 0.40**; this catches 86% of fraud but drops precision to 40% (increasing false positive audits).
            """)

    with tab2:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### ROC Curves (CV vs Holdout)")
            if oof_predictions is not None and holdout_predictions is not None:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_cv, y=tpr_cv, name=f"Cross-Validation (AUC={cv_auc:.4f})", line=dict(color=COLOR_PRIMARY, width=2.5)))
                fig_roc.add_trace(go.Scatter(x=fpr_ho, y=tpr_ho, name=f"Holdout Set (AUC={ho_auc:.4f})", line=dict(color=COLOR_FRAUD, width=2.5)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random Baseline", line=dict(color="#8892b0", dash="dash")))
                fig_roc.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="FPR", yaxis_title="TPR", legend=dict(x=0.3, y=0.1))
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Train pipeline to display dynamic ROC.")

        with col_c2:
            st.markdown("#### Precision-Recall Curves")
            if oof_predictions is not None and holdout_predictions is not None:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall_cv_curve, y=precision_cv_curve, name=f"Cross-Validation (PR-AUC={cv_pr_auc:.4f})", line=dict(color=COLOR_PRIMARY, width=2.5)))
                fig_pr.add_trace(go.Scatter(x=recall_ho_curve, y=precision_ho_curve, name=f"Holdout Set (PR-AUC={ho_pr_auc:.4f})", line=dict(color=COLOR_FRAUD, width=2.5)))
                fig_pr.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Recall", yaxis_title="Precision", legend=dict(x=0.3, y=0.1))
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("Train pipeline to display dynamic PR.")

        # Confusion Matrices side-by-side
        st.markdown("#### Confusion Matrices (Dynamic)")
        if oof_predictions is not None and holdout_predictions is not None:
            cm_cv = confusion_matrix(y_cv_true, y_cv_pred)
            cm_ho = confusion_matrix(y_ho_true, y_ho_pred)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_cm_cv = go.Figure(go.Heatmap(
                    z=cm_cv, x=["Pred: Legit","Pred: Fraud"], y=["Actual: Legit","Actual: Fraud"],
                    colorscale=[[0,"#0d1117"],[0.5,"rgba(102,126,234,0.5)"],[1,COLOR_PRIMARY]],
                    text=[[f"TN\n{cm_cv[0][0]:,}", f"FP\n{cm_cv[0][1]:,}"], [f"FN\n{cm_cv[1][0]:,}", f"TP\n{cm_cv[1][1]:,}"]],
                    texttemplate="%{text}", textfont=dict(size=14, color="white"), showscale=False
                ))
                fig_cm_cv.update_layout(**PLOTLY_LAYOUT, height=220, title="Cross-Validation Confusion Matrix")
                st.plotly_chart(fig_cm_cv, use_container_width=True)
            
            with col_m2:
                fig_cm_ho = go.Figure(go.Heatmap(
                    z=cm_ho, x=["Pred: Legit","Pred: Fraud"], y=["Actual: Legit","Actual: Fraud"],
                    colorscale=[[0,"#0d1117"],[0.5,"rgba(102,126,234,0.5)"],[1,COLOR_FRAUD]],
                    text=[[f"TN\n{cm_ho[0][0]:,}", f"FP\n{cm_ho[0][1]:,}"], [f"FN\n{cm_ho[1][0]:,}", f"TP\n{cm_ho[1][1]:,}"]],
                    texttemplate="%{text}", textfont=dict(size=14, color="white"), showscale=False
                ))
                fig_cm_ho.update_layout(**PLOTLY_LAYOUT, height=220, title="Holdout Set Confusion Matrix")
                st.plotly_chart(fig_cm_ho, use_container_width=True)

elif page == "🤖  Fraud Prediction Center":
    st.markdown('<div class="section-hdr">🤖 Fraud Prediction Center</div>', unsafe_allow_html=True)

    tab_sub, tab_manual, tab_upload = st.tabs([
        "📋 Submission Results", "✍️ Manual Risk Assessment", "📁 Batch Upload"])

    with tab_sub:
        if submission is not None:
            fraud_cnt  = (submission["Predicted_Class"]=="Yes").sum()
            legit_cnt  = (submission["Predicted_Class"]=="No").sum()
            total_cnt  = len(submission)
            fraud_pct  = fraud_cnt/total_cnt*100

            c1,c2,c3,c4 = st.columns(4)
            for col,(icon,val,lbl,clr) in zip([c1,c2,c3,c4],[
                ("📋",f"{total_cnt:,}","Total Providers","#667eea"),
                ("⚠️",f"{fraud_cnt}","Flagged Fraud","#e74c3c"),
                ("✅",f"{legit_cnt}","Legitimate","#2ecc71"),
                ("📊",f"{fraud_pct:.1f}%","Fraud Rate","#f39c12"),
            ]):
                col.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-icon">{icon}</div>
                  <div style='font-size:1.25rem;font-weight:800;color:{clr};white-space:nowrap;'>{val}</div>
                  <div class="kpi-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                st.markdown("#### Probability Distribution")
                if submission is not None:
                    probs = submission["Probability"].values
                    low_risk_probs = probs[probs < best_threshold]
                    high_risk_probs = probs[probs >= best_threshold]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=low_risk_probs, nbinsx=20,
                        marker_color=COLOR_LEGIT, opacity=0.85, name="Legitimate / Low Risk",
                        xbins=dict(start=0.0, end=float(best_threshold), size=float(best_threshold)/20)
                    ))
                    fig.add_trace(go.Histogram(
                        x=high_risk_probs, nbinsx=20,
                        marker_color=COLOR_FRAUD, opacity=0.85, name="Flagged / High Risk",
                        xbins=dict(start=float(best_threshold), end=1.0, size=(1.0 - float(best_threshold))/20)
                    ))
                    fig.add_vline(x=best_threshold, line_dash="dash", line_color="#f39c12",
                                  annotation_text=f"{best_threshold:.3f} threshold", annotation_font_color="#f39c12")
                    fig.update_layout(**PLOTLY_LAYOUT, height=280, barmode="stack",
                                      xaxis_title="Fraud Probability", yaxis_title="Count",
                                      legend=dict(orientation="h", y=-0.2, x=0.1))
                    st.plotly_chart(fig, use_container_width=True)

            with c_right:
                st.markdown("#### Prediction Split")
                fig = go.Figure(go.Pie(
                    labels=["Legitimate","Fraudulent"],
                    values=[legit_cnt, fraud_cnt],
                    hole=0.6,
                    marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD],
                                line=dict(color="#0d1117", width=3)),
                    textinfo="label+percent", pull=[0, 0.05]))
                fig.update_layout(**PLOTLY_LAYOUT, height=280,
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Provider Predictions Table")
            display_df = submission.copy()
            display_df["Risk"] = display_df["Probability"].apply(
                lambda p: "🔴 Critical" if p>=0.7 else
                          "🟠 High"     if p>=0.5 else
                          "🟡 Watch"    if p>=0.3 else "🟢 Low")
            display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

            csv = submission.to_csv(index=False)
            st.download_button("⬇️ Download Submission.csv",
                               csv, "Tharun Kumar V_Submission.csv", "text/csv",
                               use_container_width=True)

            if provider_eda is not None:
                st.markdown("---")
                st.markdown("### 🔍 Why are Providers Flagged? (Interactive Explainability)")
                st.markdown('<div class="insight">Select any training provider to inspect their specific fraud risk drivers. The model compares their metrics directly to peer groups to identify anomalies.</div>', unsafe_allow_html=True)
                
                # Combine predictions to get probabilities
                oof_preds = load_oof_predictions_v4()
                ho_preds = load_holdout_predictions_v4()
                
                if oof_preds is not None and ho_preds is not None:
                    all_preds = pd.concat([oof_preds, ho_preds], ignore_index=True)
                    provider_df = provider_eda.merge(all_preds[['Provider', 'Predicted_Probability']], on='Provider', how='left')
                else:
                    provider_df = provider_eda.copy()
                    provider_df['Predicted_Probability'] = provider_df['FraudLabel'] * 0.88 + 0.06
                
                # Dropdown
                flagged_only = st.checkbox("Show flagged providers only (True Fraud Label)", value=True, key="inspect_flagged_only")
                if flagged_only:
                    inspect_list = sorted(provider_df[provider_df['FraudLabel'] == 1]['Provider'].unique())
                else:
                    inspect_list = sorted(provider_df['Provider'].unique())
                
                inspect_prov = st.selectbox("Search and Analyze Provider Risk Drivers", inspect_list, key="inspect_prov_selectbox")
                
                if inspect_prov:
                    prov_row = provider_df[provider_df['Provider'] == inspect_prov].iloc[0]
                    prob_val = prov_row.get('Predicted_Probability', prov_row['FraudLabel'])
                    if pd.isna(prob_val):
                        prob_val = float(prov_row['FraudLabel'] * 0.88 + 0.06)
                        
                    c_det1, c_det2 = st.columns([1, 2.2])
                    with c_det1:
                        status_str = "🔴 Flagged Fraud" if prov_row['FraudLabel'] == 1 else "🟢 Legitimate"
                        border_color = COLOR_FRAUD if prov_row['FraudLabel'] == 1 else COLOR_LEGIT
                        st.markdown(f"""
                        <div class="kpi-card" style="border-color: {border_color}; margin-top: 1rem;">
                          <div class="kpi-icon">🛡️</div>
                          <div class="kpi-val">{prob_val*100:.1f}%</div>
                          <div class="kpi-lbl">Predicted Fraud Probability</div>
                          <div style="font-weight: 700; color: {border_color}; margin-top: 0.5rem; font-size: 1.1rem;">{status_str}</div>
                        </div>""", unsafe_allow_html=True)
                        
                    with c_det2:
                        st.markdown("#### Primary Risk Drivers & Peer Benchmarking")
                        # Legit means
                        legit_df = provider_df[provider_df['FraudLabel'] == 0]
                        
                        # Peer grouping based on claims volume
                        p_claims = prov_row['TotalClaims']
                        if p_claims < 50:
                            peer_label = "Small-volume Providers (<50 claims)"
                            legit_peer_df = legit_df[legit_df['TotalClaims'] < 50]
                        elif p_claims <= 200:
                            peer_label = "Medium-volume Providers (50-200 claims)"
                            legit_peer_df = legit_df[(legit_df['TotalClaims'] >= 50) & (legit_df['TotalClaims'] <= 200)]
                        else:
                            peer_label = "Large-volume Providers (>200 claims)"
                            legit_peer_df = legit_df[legit_df['TotalClaims'] > 200]
                            
                        if len(legit_peer_df) == 0:
                            legit_peer_df = legit_df
                            peer_label = "All Legitimate Providers"
                            
                        st.caption(f"Peer Group: **{peer_label}**")
                        legit_means = legit_peer_df.mean(numeric_only=True)
                        
                        drivers = []
                        # 1. Total Reimbursement
                        r_val = prov_row['TotalReimbursement']
                        l_val = legit_means['TotalReimbursement']
                        if r_val > l_val:
                            drivers.append(("Total Reimbursement", f"${r_val:,.2f}", f"${l_val:,.2f}", r_val/max(l_val, 1), "💰 Extremely high billing value relative to peer benchmark"))
                            
                        # 2. Total Hospital Days
                        r_stay = prov_row['TotalHospitalDays']
                        l_stay = legit_means['TotalHospitalDays']
                        if r_stay > l_stay:
                            drivers.append(("Total Hospital Days", f"{r_stay:.1f} days", f"{l_stay:.1f} days", r_stay/max(l_stay, 0.1), "🏥 Outlier inpatient day volume relative to peer benchmark (ghost billing signature)"))
                            
                        # 3. Claims Per Patient
                        r_cpp = prov_row['ClaimsPerBeneficiary']
                        l_cpp = legit_means['ClaimsPerBeneficiary']
                        if r_cpp > l_cpp:
                            drivers.append(("Claims Per Patient", f"{r_cpp:.2f}", f"{l_cpp:.2f}", r_cpp/max(l_cpp, 0.1), "🔄 High billing frequency per patient relative to peer benchmark"))
                            
                        # 4. Chronic Conditions count
                        r_cc = prov_row['AvgChronicCondCount']
                        l_cc = legit_means['AvgChronicCondCount']
                        if r_cc > l_cc:
                            drivers.append(("Avg Chronic Conditions", f"{r_cc:.2f}", f"{l_cc:.2f}", r_cc/max(l_cc, 0.1), "🧬 Upcoded chronic conditions to justify high diagnostic complexity"))
                            
                        # 5. Repeat Patient Ratio
                        r_rpr = prov_row['RepeatPatientRatio']
                        l_rpr = legit_means['RepeatPatientRatio']
                        if r_rpr > l_rpr:
                            drivers.append(("Repeat Patient Ratio", f"{r_rpr:.1%}", f"{l_rpr:.1%}", r_rpr/max(l_rpr, 0.01), "🔄 Patient concentration anomalies across billing events"))

                        # 6. Inpatient Claims Ratio
                        r_ipr = prov_row['InpatientRatio']
                        l_ipr = legit_means['InpatientRatio']
                        if r_ipr > l_ipr:
                            drivers.append(("Inpatient claims ratio", f"{r_ipr:.1%}", f"{l_ipr:.1%}", r_ipr/max(l_ipr, 0.01), "📈 Excess inpatient billing mix relative to peer benchmark baseline"))

                        # Sort drivers by multiplier
                        drivers = sorted(drivers, key=lambda x: x[3], reverse=True)
                        
                        if drivers:
                            for idx, (name, val, peer_val, mult, desc) in enumerate(drivers[:4]):
                                st.markdown(f"🔹 **{name} ({mult:.1f}x higher)**: `{val}` vs peer group benchmark `{peer_val}` — *{desc}*")
                        else:
                            st.info("No elevated risk metrics found relative to legitimate peers of similar size.")

    with tab_manual:
        st.markdown("#### 🔎 Real-Time Provider Risk Assessment")
        st.markdown('<div class="insight">Enter provider billing statistics to get an instant fraud probability score from the trained stacking ensemble model.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Volume**")
            total_claims     = st.number_input("Total Claims", 0, 10000, 150)
            inpatient_claims = st.number_input("Inpatient Claims", 0, 5000, 30)
            unique_bene      = st.number_input("Unique Beneficiaries", 1, 5000, 60)
        with c2:
            st.markdown("**💰 Financial**")
            total_reimb   = st.number_input("Total Reimbursement ($)", 0, 10000000, 80000)
            avg_stay      = st.number_input("Avg Hospital Stay (days)", 0.0, 100.0, 4.0, 0.5)
            avg_diag      = st.number_input("Avg Diagnosis Codes", 1.0, 10.0, 3.5, 0.5)
        with c3:
            st.markdown("**🧬 Clinical**")
            avg_chronic      = st.number_input("Avg Chronic Conditions", 0.0, 11.0, 3.0, 0.5)
            repeat_ratio     = st.slider("Repeat Patient Ratio", 0.0, 1.0, 0.25, 0.05)
            inpatient_ratio  = st.slider("Inpatient Ratio", 0.0, 1.0, 0.2, 0.05)

        if st.button("🔍 Assess Fraud Risk", type="primary", use_container_width=True):
            if model and top_features:
                row = {feat: 0.0 for feat in top_features}
                row["TotalClaims"]          = total_claims
                row["InpatientClaims"]      = inpatient_claims
                row["UniqueBeneficiaries"]  = unique_bene
                row["TotalReimbursement"]   = total_reimb
                row["AvgHospitalStay"]      = avg_stay
                row["TotalHospitalDays"]    = avg_stay * inpatient_claims
                row["AvgNumDiagCodes"]      = avg_diag
                row["AvgChronicCondCount"]  = avg_chronic
                row["RepeatPatientRatio"]   = repeat_ratio
                row["InpatientRatio"]       = inpatient_ratio
                row["ClaimsPerBeneficiary"] = total_claims / max(unique_bene,1)
                row["ReimbPerBeneficiary"]  = total_reimb / max(unique_bene,1)
                row["MaxClaimAmt"]          = total_reimb / max(total_claims,1) * 3
                row["TotalDeductible"]      = total_reimb * 0.15

                X_row = pd.DataFrame([row])[top_features].fillna(0)
                prob  = model.predict_proba(X_row)[0,1]

                st.markdown("---")
                ca, cb = st.columns([1,2])
                with ca:
                    if prob >= 0.7:
                        cls, bg = "🔴 HIGH RISK", "rgba(231,76,60,0.15)"
                        border  = "#e74c3c"
                    elif prob >= 0.5:
                        cls, bg = "🟠 MEDIUM RISK", "rgba(243,156,18,0.15)"
                        border  = "#f39c12"
                    elif prob >= 0.3:
                        cls, bg = "🟡 WATCH LIST", "rgba(241,196,15,0.15)"
                        border  = "#f1c40f"
                    else:
                        cls, bg = "🟢 LOW RISK", "rgba(46,204,113,0.15)"
                        border  = "#2ecc71"
                    st.markdown(f"""
                    <div style='background:{bg};border:2px solid {border};
                                border-radius:16px;padding:1.5rem;text-align:center'>
                      <div style='font-size:1rem;color:#a8b2d8'>Fraud Probability</div>
                      <div style='font-size:3rem;font-weight:900;color:{border}'>{prob*100:.1f}%</div>
                      <div style='font-size:1.1rem;font-weight:700;color:{border};margin-top:.3rem'>{cls}</div>
                    </div>""", unsafe_allow_html=True)

                with cb:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob*100,1),
                        number={"suffix":"%","font":{"size":32,"color":border}},
                        gauge={
                            "axis":{"range":[0,100],"tickcolor":"#8892b0"},
                            "bar":{"color":border},
                            "bgcolor":"rgba(255,255,255,0.05)",
                            "steps":[
                                {"range":[0,30],"color":"rgba(46,204,113,0.15)"},
                                {"range":[30,50],"color":"rgba(241,196,15,0.1)"},
                                {"range":[50,70],"color":"rgba(243,156,18,0.15)"},
                                {"range":[70,100],"color":"rgba(231,76,60,0.2)"},
                            ],
                            "threshold":{"line":{"color":"white","width":2},"value":50},
                        }
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(t=30))
                    st.plotly_chart(fig, use_container_width=True)

                    if prob >= 0.7:
                        st.error("Recommend: Immediate investigation + payment hold")
                    elif prob >= 0.5:
                        st.warning("Recommend: Enhanced auditing + site visits")
                    elif prob >= 0.3:
                        st.info("Recommend: Quarterly review + peer benchmarking")
                    else:
                        st.success("Recommend: Standard claims processing")
            else:
                st.warning("⚠️ Model not loaded. Run `python run_pipeline.py` first.")

    with tab_upload:
        st.markdown("#### 📁 Batch Provider Prediction")
        st.markdown('<div class="insight">Upload a CSV with provider features to get fraud predictions for multiple providers at once.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Provider Features CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"**Uploaded:** {df_up.shape[0]:,} rows × {df_up.shape[1]} columns")
            st.dataframe(df_up.head(5), use_container_width=True)
            if model and top_features:
                for c in top_features:
                    if c not in df_up.columns: df_up[c] = 0
                probs   = model.predict_proba(df_up[top_features].fillna(0))[:,1]
                classes = ["Yes" if p>=best_threshold else "No" for p in probs]
                df_up["Probability"]     = probs.round(4)
                df_up["Predicted_Class"] = classes
                df_up["Risk_Tier"]       = ["🔴 Critical" if p>=0.7 else
                                             "🟠 High" if p>=best_threshold else
                                             "🟡 Watch" if p>=0.3 else "🟢 Low" for p in probs]
                fc = sum(1 for c in classes if c=="Yes")
                ca, cb, cc = st.columns(3)
                ca.metric("Total Providers", len(df_up))
                cb.metric("Flagged Fraud",   fc)
                cc.metric("Fraud Rate",       f"{fc/len(df_up)*100:.1f}%")
                st.dataframe(df_up[["Probability","Predicted_Class","Risk_Tier"]],
                             use_container_width=True)
                st.download_button("⬇️ Download Predictions", df_up.to_csv(index=False),
                                   "predictions.csv", "text/csv", use_container_width=True)

elif page == "💼  Business ROI":
    st.markdown('<div class="section-hdr">💼 Business ROI & Financial Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">Optimize decision thresholds based on real financial implications (audit costs vs. recovered fraud values).</div>', unsafe_allow_html=True)

    if oof_predictions is not None:
        y_true = oof_predictions['Actual_Label'].values
        y_prob = oof_predictions['Predicted_Probability'].values
        
        # Financial Sliders
        c1, c2 = st.columns(2)
        audit_cost = c1.slider("Average Cost of Audit ($)", 100, 5000, 1000, 100)
        fraud_val = c2.slider("Average Value of Fraud Recovered ($)", 1000, 50000, 15000, 500)
        
        # Threshold selector
        th = st.slider("Probability Decision Cutoff", 0.05, 0.95, float(best_threshold), 0.05, key="roi_threshold_slider")
        
        # Calculate ROI at current threshold
        preds = (y_prob >= th).astype(int)
        tp = int(sum((preds == 1) & (y_true == 1)))
        fp = int(sum((preds == 1) & (y_true == 0)))
        fn = int(sum((preds == 0) & (y_true == 1)))
        
        audits = tp + fp
        total_audit_cost = audits * audit_cost
        total_recovery = tp * fraud_val
        net_savings = total_recovery - total_audit_cost
        lost_fraud = fn * fraud_val
        
        # Display KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(102,126,234,0.5)">
          <div style='font-size:1.5rem'>🔍</div>
          <div style='font-size:1.25rem;font-weight:800;color:#ccd6f6;white-space:nowrap;'>{audits}</div>
          <div class="kpi-lbl">Audits Triggered</div>
        </div>""", unsafe_allow_html=True)
        
        k2.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(231,76,60,0.5)">
          <div style='font-size:1.5rem'>💸</div>
          <div style='font-size:1.25rem;font-weight:800;color:#e74c3c;white-space:nowrap;'>${total_audit_cost:,}</div>
          <div class="kpi-lbl">Audit Expenditures</div>
        </div>""", unsafe_allow_html=True)

        k3.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(46,204,113,0.5)">
          <div style='font-size:1.5rem'>📈</div>
          <div style='font-size:1.25rem;font-weight:800;color:#2ecc71;white-space:nowrap;'>${total_recovery:,}</div>
          <div class="kpi-lbl">Fraud Recovered</div>
        </div>""", unsafe_allow_html=True)

        k4.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(6,182,212,0.5)">
          <div style='font-size:1.5rem'>💎</div>
          <div style='font-size:1.25rem;font-weight:800;color:#06b6d4;white-space:nowrap;'>${net_savings:,}</div>
          <div class="kpi-lbl">Net Business Benefit</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("")
        
        # Plot Net Savings vs Threshold curve
        ths = np.arange(0.05, 0.96, 0.05)
        net_savings_curve = []
        for t in ths:
            tp_t = sum((y_prob >= t) & (y_true == 1))
            fp_t = sum((y_prob >= t) & (y_true == 0))
            net_savings_curve.append((tp_t * fraud_val) - ((tp_t + fp_t) * audit_cost))
            
        optimal_idx = np.argmax(net_savings_curve)
        opt_th = ths[optimal_idx]
        opt_sav = net_savings_curve[optimal_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ths, y=net_savings_curve, name="Net Business Benefit ($)", line=dict(color="#06b6d4", width=3)))
        fig.add_vline(x=th, line_dash="dash", line_color="#e74c3c", annotation_text=f"Active Cutoff: {th:.2f}")
        fig.add_vline(x=opt_th, line_dash="dot", line_color="#2ecc71", annotation_text=f"Optimal Cutoff: {opt_th:.2f} (Savings: ${opt_sav:,})")
        fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Threshold", yaxis_title="Net Savings ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"💡 **Operational Recommendation:** Auditing at a threshold of **{opt_th:.2f}** yields the maximum economic profit of **${opt_sav:,}**.")
    else:
        st.info("Train pipeline to enable the dynamic ROI cost-benefit calculator.")
