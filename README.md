# Healthcare Provider Fraud Detection & Decision Support Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badge-gradient-darkblue.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Problem Statement

Healthcare fraud is a critical economic challenge, draining hundreds of billions of dollars annually from insurance providers, Medicare/Medicaid, and ultimately patients. Fraudulent organizations leverage billing vulnerabilities to submit phantom claims, overstate medical necessities, or systematically duplicate claim events. Traditional claims-auditing relies on rule-based processing, which is slow and struggles to detect shifting fraud patterns.

To solve this problem, we develop an end-to-end Machine Learning pipeline that aggregates claim-level billing records to provider-level behavioral summaries. By grouping outpatient, inpatient, and beneficiary features, our model flags anomalous billing signatures. This provides auditors with a targeted list of high-value fraud suspects to review.

The primary target is the provider ID. Since fraud cases are relatively rare compared to legitimate billings (yielding a 9.7:1 class imbalance), the model is optimized for high recall while preserving high precision to ensure operational auditing efficiency.

---

## Solution Architecture

```
Raw Data Ingestion
(Inpatient + Outpatient + Beneficiary)
       │
       ▼
Data Management & Preprocessing
(Standardize schemas, handle missing values, format datetime)
       │
       ▼
Behavioral Feature Engineering
(Generate 61 custom provider-level features)
       │
       ▼
Feature Selection (MI + RF Rank)
(Isolate top 35 high-value indicators)
       │
       ▼
Model Stacking Ensemble
(Optuna XGBoost GPU + LightGBM DART + CatBoost)
       │
       ▼
Operational Risk Tiers & Business ROI Calculator
(Dual F1 & F2 Threshold Auditing Actions)
```

---

## Dataset Summary

| Dataset Name | Row Count | Column Count | Missing Values | Duplicate Count | Memory Footprint | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train Labels** | 5,410 | 2 | 0 | 0 | 0.08 MB | Training class targets (Yes/No) |
| **Beneficiary (Train)** | 138,556 | 25 | 8,202 | 0 | 27.70 MB | Patient medical profile & coverages |
| **Inpatient (Train)** | 40,474 | 30 | 128,103 | 0 | 9.20 MB | Inpatient claims details & dates |
| **Outpatient (Train)** | 517,737 | 27 | 1,540,111 | 0 | 106.50 MB | Outpatient claims details & dates |
| **Test Labels** | 1,353 | 1 | 1,353 | 0 | 0.01 MB | Evaluation cohort provider list |
| **Beneficiary (Test)** | 34,640 | 25 | 2,050 | 0 | 6.90 MB | Test patient medical profiles |
| **Inpatient (Test)** | 9,974 | 30 | 31,610 | 0 | 2.30 MB | Test inpatient claims details |
| **Outpatient (Test)** | 125,576 | 27 | 373,710 | 0 | 25.80 MB | Test outpatient claims details |

---

## Feature Engineering Summary

We engineer 61 provider-level features categorized into 5 distinct behavioural domains:

1. **Volume (12 features):** Total claim counts, inpatient split, unique patient counts, and distinct attending physician counts.
2. **Financial (15 features):** Cumulative reimbursements, average claim sizes, patient deductible sums, and per-beneficiary billing ratios.
3. **Clinical (10 features):** Total hospitalization stay days, average diagnosis codes count, and chronic disease prevalence metrics.
4. **Behavioral (14 features):** Repeated diagnosis rates, patient recycling indices, and physician concentration metrics.
5. **Temporal (10 features):** Weekend claim billing ratios, consecutive claims time gaps, and active days ratios.

---

## Model Performance

Stratified cross-validation and holdout validation results:

| Model | CV ROC-AUC | Holdout ROC-AUC | CV F1 | Holdout F1 | CV Recall | Holdout Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | **0.9343** | **0.9567** | **0.6340** | **0.6783** | **69.67%** | **76.47%** |
| **Random Forest (300)** | 0.9352 | — | 0.5679 | — | 85.18% | — |
| **XGBoost (Optuna)** | 0.9296 | — | 0.5736 | — | 82.41% | — |
| **Logistic Regression** | 0.8940 | — | 0.5799 | — | 78.85% | — |

> [!NOTE]
> Stacking Ensemble uses XGBoost, LightGBM (DART), and CatBoost meta-tuned via Logistic Regression. Holdout performance exceeding CV score verifies high generalization capacity.

---

## Top 10 Fraud Signals (SHAP)

1. **TotalReimbursement (0.141):** Highest contributor. Excess billing value relative to size benchmarks.
2. **MaxDiagCodes (0.134):** Billing maximum allowable diagnosis codes (upcoding signature).
3. **TotalDeductible (0.075):** Copay volume anomalies indicating duplicate billing.
4. **TotalHospitalDays (0.074):** Inflated inpatient stays (phantom bed billing).
5. **MaxClaimAmt (0.059):** Outlier claims indicating single major billing fraud.
6. **InpatientClaims (0.047):** High ratio of inpatient treatments (which have higher baseline payouts).
7. **AvgNumProcCodes (0.039):** Density of billed procedures per visit.
8. **RepeatedDiagRatio (0.036):** Repetitive copy-pasting of primary diagnosis codes.
9. **AvgUniqueProcCodes (0.031):** Diversity of procedures billed.
10. **ReimbPerBeneficiary (0.030):** Billing excess values on a small patient cohort.

---

## Risk Tier Framework

| Operational Risk Tier | Risk Score Range | Decision Threshold | Audit Action |
| :--- | :---: | :---: | :--- |
| **Critical Risk** | &ge; 0.70 | F1-Optimal (0.865) | Immediate payment suspension + Special Investigation audit |
| **High Risk** | 0.50 - 0.69 | F2-Optimal (0.597) | Pre-payment claims audit + on-site clinical review |
| **Watch List** | 0.30 - 0.49 | N/A | Quarterly profile monitoring + peer volume comparison |
| **Low Risk** | < 0.30 | N/A | Standard automated claims processing |

---

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Training & Evaluation Pipeline
```bash
python run_pipeline.py
```

### Step 3: Run Interactive UI Dashboard
```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
.
├── Data/                             # Raw CSV data files (git-ignored)
├── run_pipeline.py                   # Full model training and evaluation script
├── streamlit_app.py                  # Streamlit dashboard application
├── requirements.txt                  # Python dependencies
├── packages.txt                      # System dependencies for Streamlit Cloud
├── README.md                         # Documentation
├── .gitignore                        # Git exclusions file
├── pipeline_summary.json             # Evaluation run metadata outputs
├── model_results.csv                 # Model metrics table
├── shap_importance.csv               # SHAP values summary
└── Tharun Kumar V_Submission.csv     # Test set predictions output file
```

---

## Submission Format

The submission file `Tharun Kumar V_Submission.csv` contains predictions for the 1,353 test providers. It has the following columns:
- **Provider:** Provider unique identifier.
- **Probability:** Model-predicted probability of fraud (0.0 to 1.0).
- **Predicted_Class:** Class prediction ('Yes' for fraud, 'No' for safe) mapped using the F1-Optimal threshold of 0.865.

---

## References

1. CMS Medicare claims research data.
2. US Government Accountability Office (GAO) healthcare fraud reports.
3. NHCAA: Financial impact of healthcare billing schemes.
4. SHAP (SHapley Additive exPlanations) research papers.
5. Scikit-learn, XGBoost, and LightGBM API documentations.

---

## Confidentiality Note

This codebase is a professional case study and solution platform for healthcare fraud detection. All dataset contents, outputs, and details are protected under standard non-disclosure agreements.
