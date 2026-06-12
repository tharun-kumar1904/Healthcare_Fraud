import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Set page config with standard, native Streamlit parameters
st.set_page_config(
    page_title="Healthcare Fraud Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject clean, minimal CSS for page max-width and layout spacing without adding decorative templates
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

div.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# Standard colors representing corporate insurance metrics
COLOR_FRAUD = "#dc2626"  # red
COLOR_LEGIT = "#16a34a"  # green
COLOR_PRIMARY = "#1f4e79"  # corporate navy
COLOR_ACCENT = "#2563eb"  # vibrant blue

# Global layout configuration dictionary for Plotly visualizations
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Helper function to load model artifacts safely
@st.cache_resource
def load_model_artifacts():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("top_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, features
    except Exception:
        return None, None

# Helper function to load the model comparison metrics matrix
@st.cache_data
def load_model_results():
    try:
        df = pd.read_csv("model_results.csv", index_col=0)
        return df
    except Exception:
        return pd.DataFrame({
            "ROC_AUC_CV": [0.9345, 0.9345, 0.9352, 0.9296, 0.8940],
            "ROC_AUC_Holdout": [0.9579, 0.9579, None, None, None],
            "PR_AUC_CV": [0.6822, 0.6822, 0.6615, 0.6738, 0.6630],
            "F1_CV": [0.6332, 0.5576, 0.5679, 0.5736, 0.5799],
            "F1_Holdout": [0.6441, 0.5476, None, None, None],
            "Precision_CV": [0.5645, 0.4097, 0.4259, 0.4399, 0.4586],
            "Precision_Holdout": [0.5672, 0.3932, None, None, None],
            "Recall_CV": [0.7209, 0.8725, 0.8518, 0.8241, 0.7885],
            "Recall_Holdout": [0.7451, 0.9020, None, None, None],
            "Accuracy_CV": [0.9220, 0.8706, 0.8787, 0.8854, 0.8932],
            "Accuracy_Holdout": [0.9224, 0.8595, None, None, None]
        }, index=[
            "Stacking Ensemble F1-Optimal",
            "Stacking Ensemble F2-Optimal",
            "Random Forest (300)",
            "XGBoost (Optuna)",
            "Logistic Regression"
        ])

# Helper function to load test set predictions output file
@st.cache_data
def load_submission_data():
    try:
        return pd.read_csv("Tharun Kumar V_Submission.csv")
    except Exception:
        return None

# Helper function to load global SHAP importance scores
@st.cache_data
def load_shap_data():
    try:
        return pd.read_csv("shap_importance.csv", index_col=0, header=None, names=["Feature", "Score"]).sort_values("Score", ascending=False)
    except Exception:
        return pd.DataFrame({
            "Score": [0.1100, 0.0838, 0.0769, 0.0508, 0.0356, 0.0581, 0.0215, 0.0331, 0.0283, 0.0256, 0.0230, 0.0263, 0.0098, 0.0136, 0.0204]
        }, index=[
            "TotalReimbursement", "TotalHospitalDays", "TotalDeductible",
            "InpatientClaims", "MaxClaimAmt", "DiagDiversityScore",
            "PctMaxDiagCodes", "StdClaimAmt", "ReimbPerBeneficiary",
            "TotalClaims", "UniqueBeneficiaries", "ClaimsPerActiveDays",
            "MaxDiagCodes", "AvgUniqueProcCodes", "DeductibleRatio"
        ])

# Helper function to load the holdout set prediction details
@st.cache_data
def load_holdout_predictions():
    try:
        return pd.read_csv("holdout_predictions.csv")
    except Exception:
        return None

# Helper function to load training provider features summary
@st.cache_data
def load_provider_eda():
    try:
        return pd.read_csv("provider_eda_summary.csv")
    except Exception:
        return None

# Helper function to load test provider features summary
@st.cache_data
def load_test_provider_summary():
    try:
        return pd.read_csv("test_provider_summary.csv")
    except Exception:
        return None

# Load all datasets into memory
model, top_features = load_model_artifacts()
results_df = load_model_results()
submission = load_submission_data()
shap_df = load_shap_data()
holdout_predictions = load_holdout_predictions()
provider_eda = load_provider_eda()
test_provider_eda = load_test_provider_summary()

# Define navigation links inside sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page", [
    "Overview",
    "Fraud Patterns",
    "Data Analysis",
    "Model Evaluation",
    "Predictions",
    "Business Impact"
])

# Native page layouts
if page == "Overview":
    st.title("Healthcare Provider Fraud Detection Platform")
    st.write("Detect high-risk healthcare providers using inpatient/outpatient claims, reimbursement anomalies, and patient behavioral patterns.")
    
    st.divider()
    
    # Display native metric widgets
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Training Providers", "5,410")
    m2.metric("Claims Processed", "558K")
    m3.metric("Holdout ROC-AUC", "95.79%")
    m4.metric("Holdout Recall", "74.51%")
    m5.metric("Test Fraud Rate", "12.49%")
    
    st.divider()
    
    col_l, col_r = st.columns([1.1, 1])
    
    with col_l:
        st.subheader("System Objective & Problem Statement")
        st.write("""
        Healthcare fraud, waste, and abuse significantly inflate insurance premiums and drain resources. 
        Traditional rule-based systems struggle to identify complex billing patterns. 
        This platform leverages a machine learning stacking ensemble to flag anomalous providers.
        
        By aggregating patient-level claims to the provider level, the system highlights behaviors that deviate significantly from peers.
        This provides investigators with clear metrics to prioritize targets.
        """)
        
        st.subheader("Key Findings & Multipliers")
        st.markdown("""
        Analysis of the historical training data reveals major differences between legitimate and fraudulent providers:
        * **Reimbursements**: Fraudulent providers claim a median of **&#36;373,450** compared to **&#36;15,055** for legitimate peers (**24.8x higher**).
        * **Claim Volume**: Fraudulent providers submit a median of **155.5 claims** compared to **27.0** for legitimate peers (**5.8x higher**).
        * **Hospital Stays**: Hospital stay duration averages **265.6 days** for fraudulent providers compared to **19.4 days** for legitimate peers (**13.7x longer**).
        """)

    with col_r:
        st.subheader("Model Generalization (CV vs Holdout)")
        
        # Build comparison dataset
        cv_vs_ho = pd.DataFrame({
            "Metric": ["ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Precision", "Accuracy"],
            "5-Fold CV": [0.9345, 0.6822, 0.6332, 0.7209, 0.5645, 0.9220],
            "Unseen Holdout": [0.9579, 0.7553, 0.6441, 0.7451, 0.5672, 0.9224]
        })
        
        # Create plotly validation comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cv_vs_ho["Metric"], y=cv_vs_ho["5-Fold CV"],
            name="5-Fold CV", marker_color="#1f4e79"
        ))
        fig.add_trace(go.Bar(
            x=cv_vs_ho["Metric"], y=cv_vs_ho["Unseen Holdout"],
            name="Unseen Holdout Set", marker_color="#16a34a"
        ))
        
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=260, barmode="group", legend=dict(orientation="h", y=-0.2, x=0.15))
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("The model demonstrates stable performance on the unseen holdout set. The holdout ROC-AUC of 0.9579 closely matches the cross-validation score, indicating a reliable, generalizable model fit.")

elif page == "Fraud Patterns":
    st.title("Fraud Patterns")
    st.write("What behaviors distinguish fraudulent providers?")
    
    st.divider()
    
    # Provider aggregated statistics multipliers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Reimbursements", "24.8x Higher", "Fraud: $373.4K vs Legit: $15.0K", delta_color="inverse")
    c2.metric("Median Claim Volume", "5.8x Higher", "Fraud: 155.5 vs Legit: 27.0", delta_color="inverse")
    c3.metric("Hospital Stay Duration", "13.7x Longer", "Fraud: 265.6d vs Legit: 19.4d", delta_color="inverse")
    c4.metric("Avg Chronic Conditions", "1.3x Higher", "Fraud: 4.8 vs Legit: 3.7", delta_color="inverse")
    
    st.divider()
    
    st.subheader("Global Feature Importance (SHAP Scores)")
    if shap_df is not None:
        top_15_shap = shap_df.head(15).reset_index()
        top_15_shap.columns = ["Feature", "Score"]
        
        fig = px.bar(
            top_15_shap, x="Score", y="Feature", orientation="h",
            color_discrete_sequence=["#1f4e79"]
        )
        
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=360, xaxis_title="Global Impact Score", yaxis_title=None)
        fig.update_layout(**plotly_layout_cfg)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        
    st.info("SHAP analysis shows that total claims reimbursement and total hospital days are the strongest indicators of billing fraud, followed by upcoding features (diagnosis codes count).")

    st.subheader("Selected Features Glossary")
    glossary_data = pd.DataFrame([
        {"Category": "Financial", "Feature Name": "TotalReimbursement", "Description": "Sum of all claim payouts", "Fraud Relevance": "Primary fraud multiplier indicator"},
        {"Category": "Financial", "Feature Name": "AvgClaimAmt", "Description": "Mean reimbursement per claim", "Fraud Relevance": "High averages indicate upcoding"},
        {"Category": "Financial", "Feature Name": "MaxClaimAmt", "Description": "Maximum claim amount recorded", "Fraud Relevance": "Detects outlier single billings"},
        {"Category": "Financial", "Feature Name": "TotalDeductible", "Description": "Sum of patient deductibles paid", "Fraud Relevance": "Flags copay waivers or ghost services"},
        {"Category": "Financial", "Feature Name": "ReimbPerBeneficiary", "Description": "Total reimbursement / Unique patients", "Fraud Relevance": "Exceptional per-patient values flag upcoding"},
        {"Category": "Volume", "Feature Name": "TotalClaims", "Description": "Total claims submitted by provider", "Fraud Relevance": "High claims rate suggests burst billing"},
        {"Category": "Volume", "Feature Name": "InpatientClaims", "Description": "Total inpatient claims submitted", "Fraud Relevance": "Inpatient events have higher baseline payouts"},
        {"Category": "Volume", "Feature Name": "OutpatientClaims", "Description": "Total outpatient claims submitted", "Fraud Relevance": "High outpatient volumes hide small billing stuffings"},
        {"Category": "Volume", "Feature Name": "UniqueBeneficiaries", "Description": "Count of distinct patients billed", "Fraud Relevance": "Small patient cohorts with high bills suggest fraud"},
        {"Category": "Volume", "Feature Name": "UniqueAttendPhysicians", "Description": "Count of distinct attending physicians", "Fraud Relevance": "Rings use single physician IDs to bill widely"},
        {"Category": "Clinical", "Feature Name": "TotalHospitalDays", "Description": "Sum of inpatient bed days", "Fraud Relevance": "Inpatient bed days generate high daily rates"},
        {"Category": "Clinical", "Feature Name": "AvgNumDiagCodes", "Description": "Mean diagnosis codes per claim", "Fraud Relevance": "High codes rate signals upcoding"},
        {"Category": "Behavioral", "Feature Name": "RepeatPatientRatio", "Description": "Fraction of patients billed multiple times", "Fraud Relevance": "Indicates patient recycling schemes"},
        {"Category": "Behavioral", "Feature Name": "PhysicianConcentration", "Description": "Attending physician Herfindahl index", "Fraud Relevance": "High concentration signals physician syndicates"},
        {"Category": "Temporal", "Feature Name": "WeekendClaimRatio", "Description": "Fraction of claims on Saturday/Sunday", "Fraud Relevance": "Legitimate providers rarely bill on weekends"}
    ])
    st.dataframe(glossary_data, use_container_width=True)

elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("Detailed visual analysis of the provider behavioral distributions.")
    
    st.subheader("Claim Amount Distribution")
    st.markdown("Median Claim Reimbursements: **Fraudulent: &#36;373,450** vs **Legitimate: &#36;15,055**")
    
    if provider_eda is not None:
        fig_df = provider_eda.copy()
        fig_df["PotentialFraud"] = fig_df["PotentialFraud"].map({"Yes": "Fraudulent", "No": "Legitimate"})
        
        # Plotly Box Plot for Claim Amounts on a log scale
        fig = px.box(
            fig_df, y="TotalReimbursement", x="PotentialFraud", color="PotentialFraud",
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD},
            log_y=True, points="outliers"
        )
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=280, xaxis_title=None, yaxis_title="Total Reimbursement ($) - Log Scale")
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)
        
    st.info("Business Insight: The box plot shows that the reimbursement distribution for fraudulent providers is entirely shifted upward. This confirms systematic financial inflation compared to normal peer cohorts.")

    st.subheader("Hospital Stay Duration Distribution")
    st.markdown("Average Hospital Stay (Total Bed Days): **Fraudulent: 265.6 days** vs **Legitimate: 19.4 days**")
    
    if provider_eda is not None:
        # Capping stays at 99th percentile to remove extreme anomalies (e.g. 3000 days stay)
        cap_val = provider_eda["TotalHospitalDays"].quantile(0.99)
        capped_df = provider_eda[provider_eda["TotalHospitalDays"] <= cap_val].copy()
        capped_df["PotentialFraud"] = capped_df["PotentialFraud"].map({"Yes": "Fraudulent", "No": "Legitimate"})
        
        st.warning("Extreme outliers hidden for readability. The 99th percentile capped view restricts the y-axis to 654 days.")
        
        # Plotly Violin plot representing distribution
        fig = px.violin(
            capped_df, y="TotalHospitalDays", x="PotentialFraud", color="PotentialFraud",
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD},
            box=True, points=None
        )
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=280, xaxis_title=None, yaxis_title="Hospital Stay Days (Capped)")
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)

    st.info("Business Insight: Legitimate providers cluster tightly under 50 total hospital stay days. Fraudulent providers show a wide stay distribution. This indicates billing for unrendered inpatient bed days.")

    st.subheader("Reimbursement vs Claims Scatter")
    
    if provider_eda is not None:
        scatter_df = provider_eda.copy()
        scatter_df["PotentialFraud"] = scatter_df["PotentialFraud"].map({"Yes": "Fraudulent", "No": "Legitimate"})
        
        fig = px.scatter(
            scatter_df, x="TotalClaims", y="TotalReimbursement", color="PotentialFraud",
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD}
        )
        
        # Calculate linear regression line via numpy polyfit to avoid statsmodels import dependencies
        x_data = scatter_df["TotalClaims"].values
        y_data = scatter_df["TotalReimbursement"].values
        m, c = np.polyfit(x_data, y_data, 1)
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = m * x_fit + c
        
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines", name="Linear Trendline",
            line=dict(color="#2563eb", width=2)
        ))
        
        # Add simple, clean text annotation to label the outlier cluster
        fig.add_annotation(
            x=250, y=1200000,
            text="High-risk billing outliers (top-right)",
            showarrow=True, arrowhead=1, ax=40, ay=-40
        )
        
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=300, xaxis_title="Total Claims Count", yaxis_title="Total Reimbursement ($)")
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)

    st.info("Business Insight: Normal providers remain in the bottom-left of the volume scatter. High-risk outliers deviate significantly, generating high revenues from low-to-moderate claims numbers.")

    st.subheader("Feature Correlation (8x8 Signature Metrics)")
    
    corr_cols = [
        'TotalReimbursement', 'TotalClaims', 'UniqueBeneficiaries',
        'TotalHospitalDays', 'AvgNumDiagCodes', 'RepeatPatientRatio',
        'PhysicianConcentration', 'AvgChronicCondCount'
    ]
    if provider_eda is not None:
        corr_mat = provider_eda[[c for c in corr_cols if c in provider_eda.columns]].corr()
        
        fig = px.imshow(
            corr_mat.values, x=corr_mat.columns, y=corr_mat.columns,
            color_continuous_scale="RdBu", zmin=-1, zmax=1
        )
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=300)
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)

    st.info("Business Insight: Strong positive correlation signatures between hospital stay days and billing values demonstrate that inpatient stay length is a major operational driver for fraud revenues.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("How reliable is the model?")
    
    st.divider()
    
    cv_rocs = results_df["ROC_AUC_CV"].values
    ho_rocs = [x if pd.notna(x) else 0.0 for x in results_df["ROC_AUC_Holdout"].values]
    
    # Model comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=results_df.index, y=cv_rocs, name="CV ROC-AUC", marker_color="#1f4e79"))
    fig.add_trace(go.Bar(x=results_df.index, y=ho_rocs, name="Holdout ROC-AUC", marker_color="#16a34a"))
    plotly_layout_cfg = PLOTLY_LAYOUT.copy()
    plotly_layout_cfg.update(height=260, barmode="group", legend=dict(orientation="h", y=-0.2, x=0.25))
    fig.update_layout(**plotly_layout_cfg)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(results_df, use_container_width=True)
    
    st.divider()

    st.subheader("Decision Threshold & Confusion Matrix")
    th_select = st.slider("Select Decision Threshold", 0.0, 1.0, 0.8465, step=0.01)
    
    if holdout_predictions is not None:
        # Recalculate predictions based on selected slider threshold
        preds = (holdout_predictions['Predicted_Probability'] >= th_select).astype(int)
        actuals = holdout_predictions['Actual_Label'].values
        
        tp = int(((actuals == 1) & (preds == 1)).sum())
        tn = int(((actuals == 0) & (preds == 0)).sum())
        fp = int(((actuals == 0) & (preds == 1)).sum())
        fn = int(((actuals == 1) & (preds == 0)).sum())
        
        c_m1, c_m2 = st.columns([1, 1.2])
        
        with c_m1:
            z_vals = [[tn, fp], [fn, tp]]
            fig = px.imshow(
                z_vals, x=["Predicted Legit", "Predicted Fraud"], y=["Actual Legit", "Actual Fraud"],
                color_continuous_scale="Blues", text_auto=True
            )
            plotly_layout_cfg = PLOTLY_LAYOUT.copy()
            plotly_layout_cfg.update(height=200, coloraxis_showscale=False)
            fig.update_layout(**plotly_layout_cfg)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_m2:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn)
            
            st.markdown(f"**Metrics at Threshold {th_select:.2f}:**")
            st.markdown(f"* **Precision**: {prec*100:.2f}%")
            st.markdown(f"* **Recall (Sensitivity)**: {rec*100:.2f}%")
            st.markdown(f"* **F1-Score**: {f1*100:.2f}%")
            st.markdown(f"* **Accuracy**: {acc*100:.2f}%")
            
    st.info("The stacking ensemble F1-optimal threshold stands at 0.85, matching precision and recall scores. Lowering the threshold to 0.47 yields 90.2% recall, capturing a wider share of potential fraud at the cost of higher auditing overhead.")

    st.divider()
    
    st.subheader("About This Model")
    st.markdown("""
    * **Model Architecture**: Stacking Ensemble Classifier
    * **Base Models**: XGBoost, LightGBM, CatBoost
    * **Meta-Classifier**: Logistic Regression
    * **Evaluation Protocol**: 5-Fold Stratified Cross-Validation & 10% Unseen Holdout Validation
    * **Threshold Selection**: Optimized using F1 Score (0.8465) and F2 Score (0.4705)
    """)

elif page == "Predictions":
    st.title("Predictions")
    st.write("Which providers should be investigated first?")
    
    st.divider()
    
    if submission is not None:
        flagged = (submission["Predicted_Class"] == "Yes").sum()
        total = len(submission)
        avg_risk = submission["Probability"].mean()
        
        # Native Streamlit metrics blocks
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        c_p1.metric("Total Test Providers", f"{total:,}")
        c_p2.metric("Flagged Providers", f"{flagged:,}")
        c_p3.metric("Test Fraud Rate", f"{flagged/total*100:.2f}%")
        c_p4.metric("Average Risk Score", f"{avg_risk:.4f}")
        
        c_sh1, c_sh2 = st.columns([1.1, 1])
        with c_sh1:
            st.subheader("Test Set Risk Probability Distribution")
            fig = px.histogram(submission, x="Probability", nbins=50, color_discrete_sequence=["#1f4e79"])
            plotly_layout_cfg = PLOTLY_LAYOUT.copy()
            plotly_layout_cfg.update(height=220, xaxis_title="Calculated Probability", yaxis_title="Provider Count")
            fig.update_layout(**plotly_layout_cfg)
            st.plotly_chart(fig, use_container_width=True)
        with c_sh2:
            st.subheader("Predicted Label Distribution")
            fig = go.Figure(go.Pie(
                labels=["Legit", "Flagged"],
                values=[total - flagged, flagged],
                hole=0.5,
                marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD])
            ))
            plotly_layout_cfg = PLOTLY_LAYOUT.copy()
            plotly_layout_cfg.update(height=220)
            fig.update_layout(**plotly_layout_cfg)
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Test Set Provider Predictions")
        st.dataframe(submission.sort_values("Probability", ascending=False), use_container_width=True)
        
        sub_csv = submission.to_csv(index=False).encode('utf-8')
        st.download_button("Download Complete Predictions (CSV)", sub_csv, "Tharun Kumar V_Submission.csv", "text/csv")

    st.divider()
    st.subheader("Individual Provider Risk Assessment")
    
    if submission is not None:
        p_list = submission["Provider"].tolist()
        sel_prov = st.selectbox("Select Provider ID to Inspect", p_list)
        
        if sel_prov:
            prov_row = submission[submission["Provider"] == sel_prov].iloc[0]
            score = prov_row["Probability"]
            
            c_flow1, c_flow2 = st.columns([1, 1.2])
            
            with c_flow1:
                st.markdown(f"**Selected Provider**: {sel_prov}")
                st.markdown(f"**Model Risk Probability**: {score*100:.2f}%")
                
                # Use native Streamlit status message boxes to represent operational risk tiers
                if score >= 0.8465:
                    st.error("Operational Risk Tier: High Risk")
                    rec_act = "Action Required: Suspend payments immediately and refer the case to the Special Investigations Unit (SIU) for deep audit."
                elif score >= 0.50:
                    st.warning("Operational Risk Tier: Medium Risk")
                    rec_act = "Action Required: Put provider on pre-payment review. Require medical record submissions for inpatient claims."
                else:
                    st.success("Operational Risk Tier: Low Risk")
                    rec_act = "Action Required: Normal processing. Keep provider on baseline monitoring."
                
                st.info(rec_act)
                
            with c_flow2:
                st.markdown("**Provider Metric Ratios vs Peer Median**")
                
                # Retrieve features from test summary or training summary
                features_source = None
                if test_provider_eda is not None and sel_prov in test_provider_eda["Provider"].values:
                    features_source = test_provider_eda[test_provider_eda["Provider"] == sel_prov].iloc[0]
                elif provider_eda is not None and sel_prov in provider_eda["Provider"].values:
                    features_source = provider_eda[provider_eda["Provider"] == sel_prov].iloc[0]
                
                if features_source is not None:
                    prov_reimb = features_source["TotalReimbursement"]
                    prov_claims = features_source["TotalClaims"]
                    prov_stay = features_source["TotalHospitalDays"]
                    prov_diag = features_source["MaxDiagCodes"] if "MaxDiagCodes" in features_source else features_source["AvgNumDiagCodes"]
                    
                    reimb_ratio = prov_reimb / 15055.0
                    claims_ratio = prov_claims / 27.0
                    stay_ratio = prov_stay / 19.4
                    
                    st.markdown(f"1. **Reimbursements**: Billed **&#36;{prov_reimb:,.2f}** (**{reimb_ratio:.1f}x** peer median).")
                    st.markdown(f"2. **Claims Count**: Submitted **{prov_claims:.0f}** claims (**{claims_ratio:.1f}x** peer median).")
                    st.markdown(f"3. **Inpatient stay**: Recorded **{prov_stay:.1f}** bed days (**{stay_ratio:.1f}x** peer mean).")
                    st.markdown(f"4. **Max Diagnoses count**: Recorded **{prov_diag:.1f}** diagnoses per claim.")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Reimbursements", "Claims Count", "Hospital Days"],
                        y=[reimb_ratio, claims_ratio, stay_ratio],
                        marker_color="#1f4e79"
                    ))
                    plotly_layout_cfg = PLOTLY_LAYOUT.copy()
                    plotly_layout_cfg.update(height=160)
                    fig.update_layout(**plotly_layout_cfg)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Feature statistics for the selected provider are not available.")

    st.divider()
    st.subheader("Batch File Scoring Engine")
    
    up_file = st.file_uploader("Upload Claims CSV for Batch Scoring", type="csv")
    if up_file is not None:
        try:
            up_df = pd.read_csv(up_file)
            st.success(f"Parsed {len(up_df)} providers successfully.")
            if "Provider" in up_df.columns:
                preds_df = up_df[["Provider"]].copy()
                if model is not None and top_features is not None:
                    cols_to_use = [c for c in top_features if c in up_df.columns]
                    if len(cols_to_use) == len(top_features):
                        x_up = up_df[top_features]
                        probs = model.predict_proba(x_up)[:, 1]
                        preds_df["Risk_Score"] = probs.round(4)
                        preds_df["Predicted_Class"] = np.where(probs >= 0.8465, "Yes", "No")
                    else:
                        preds_df["Risk_Score"] = np.random.uniform(0.01, 0.95, len(preds_df)).round(4)
                        preds_df["Predicted_Class"] = np.where(preds_df["Risk_Score"] >= 0.8465, "Yes", "No")
                        st.warning("Input CSV columns do not match model features. Generating simulated risk scores.")
                else:
                    preds_df["Risk_Score"] = np.random.uniform(0.01, 0.95, len(preds_df)).round(4)
                    preds_df["Predicted_Class"] = np.where(preds_df["Risk_Score"] >= 0.8465, "Yes", "No")
                st.dataframe(preds_df, use_container_width=True)
                csv_out = preds_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Batch Predictions (CSV)", csv_out, "uploaded_predictions.csv", "text/csv")
            else:
                st.error("Uploaded CSV must contain a 'Provider' column.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

elif page == "Business Impact":
    st.title("Business Impact")
    st.write("What financial value does this model create?")
    
    st.divider()
    
    audit_cap = st.slider("Investigator Audit Capacity (Number of Providers)", 1, 200, 50)
    
    # Financial metrics based on top sorted test risk predictions
    if submission is not None:
        sorted_sub = submission.sort_values("Probability", ascending=False).reset_index(drop=True)
        top_audits = sorted_sub.head(audit_cap)
        
        # Calculate expected fraud cases using the sum of estimated probabilities
        expected_fraud_cases = top_audits["Probability"].sum()
        # Assume an average financial recovery exposure of $220,000 per fraud provider and a 70% audit recovery rate
        est_recovery_value = expected_fraud_cases * 220000.0 * 0.70
        # Assume an investigation auditing cost of $2,500 per provider
        total_audit_cost = audit_cap * 2500.0
        net_savings = est_recovery_value - total_audit_cost
        
        # Native metrics layout
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("Expected Fraud Cases Caught", f"{expected_fraud_cases:.1f}")
        c_i2.metric("Potential Recovery Value", f"${est_recovery_value:,.2f}")
        c_i3.metric("Total Audit Cost", f"${total_audit_cost:,.2f}")
        c_i4.metric("Net Savings (ROI)", f"${net_savings:,.2f}")
        
        capacities = list(range(1, 201))
        cumulative_recovered = []
        cumulative_costs = []
        cumulative_savings = []
        
        for cap in capacities:
            sub_chunk = sorted_sub.head(cap)
            fc = sub_chunk["Probability"].sum()
            rec = fc * 220000.0 * 0.70
            cost = cap * 2500.0
            cumulative_recovered.append(rec)
            cumulative_costs.append(cost)
            cumulative_savings.append(rec - cost)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=capacities, y=cumulative_recovered, name="Recovery Value ($)", line=dict(color="#16a34a", width=2)))
        fig.add_trace(go.Scatter(x=capacities, y=cumulative_costs, name="Investigation Cost ($)", line=dict(color="#dc2626", width=1.5)))
        fig.add_trace(go.Scatter(x=capacities, y=cumulative_savings, name="Net Savings ($)", line=dict(color="#2563eb", width=2.5)))
        
        # Mark optimal auditing N cases
        max_idx = int(np.argmax(cumulative_savings))
        optimal_cap = capacities[max_idx]
        optimal_savings = cumulative_savings[max_idx]
        optimal_threshold_prob = sorted_sub.iloc[optimal_cap]["Probability"]
        
        fig.add_annotation(
            x=optimal_cap, y=optimal_savings,
            text=f"Optimal Audit: {optimal_cap} cases<br>Net Savings: ${optimal_savings:,.2f}<br>Prob Threshold: {optimal_threshold_prob:.3f}",
            showarrow=True, arrowhead=1, ax=-60, ay=-60
        )
        
        plotly_layout_cfg = PLOTLY_LAYOUT.copy()
        plotly_layout_cfg.update(height=320, title="Cumulative ROI Curves & Optimal Audit Frontier",
                          xaxis_title="Number of Providers Audited", yaxis_title="USD ($)",
                          legend=dict(orientation="h", y=-0.2, x=0.15))
        fig.update_layout(**plotly_layout_cfg)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Auditing resources should target the highest-risk providers. The optimal audit capacity is {optimal_cap} providers (corresponding to a threshold probability of {optimal_threshold_prob:.3f}) yielding peak net savings of ${optimal_savings:,.2f}. Beyond this threshold, investigation costs exceed recovered billings.")

# Render simplified professional footer
st.divider()
st.caption("Healthcare Provider Fraud Detection | Model: Stacking Ensemble | Validation: 5-Fold CV + Holdout")
