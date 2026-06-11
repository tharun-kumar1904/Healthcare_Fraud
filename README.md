# 🏥 Healthcare Provider Fraud Detection
### Sagility Data Science Case Study Submission
**Author:** Tharun | **Submitted:** 2024

---

## 📋 Problem Statement

Predict potentially fraudulent healthcare providers using Medicare insurance claims data. 
Provider fraud (upcoding, ghost billing, duplicate claims) costs the US insurance industry 
**$300+ billion annually**. This solution builds an end-to-end ML pipeline to flag 
high-risk providers for audit.

---

## 📁 Project Structure

```
Healthcare_Fraud_Detection/
├── Healthcare_Fraud_Detection.ipynb   # Complete analysis notebook (Phases 1–9)
├── streamlit_app.py                   # Interactive dashboard application
├── Tharun_Submission.csv              # Final test predictions
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── best_model.pkl                     # Trained Random Forest model
├── top_features.pkl                   # Selected 35 features
└── data/
    └── Case Study/
        ├── Training Data/
        │   ├── Train-*.csv            # Provider fraud labels
        │   ├── Train_Beneficiary*.csv # Patient demographics
        │   ├── Train_Inpatient*.csv   # Hospital admission claims
        │   └── Train_Outpatient*.csv  # Outpatient visit claims
        └── Unseen Data/
            └── (same structure)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook Healthcare_Fraud_Detection.ipynb
```

### 3. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 📊 Datasets

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| Train Labels | 5,410 | 2 | Provider fraud ground truth |
| Beneficiary (Train) | 138,556 | 25 | Patient KYC & health conditions |
| Inpatient (Train) | 40,474 | 30 | Hospital admission claims |
| Outpatient (Train) | 517,737 | 27 | Outpatient visit claims |
| Test (Unseen) | 1,353 | 1 | Providers for prediction |

---

## ⚙️ Solution Pipeline

| Phase | Description | Key Output |
|-------|-------------|------------|
| 1. Data Understanding | Profile all datasets, ER diagram | Dataset summary, quality report |
| 2. Data Management | Clean, merge, preprocess | Master merged dataset |
| 3. EDA | Deep statistical analysis | 10+ visualizations with insights |
| 4. Feature Engineering | Create 53 provider-level features | Feature matrix |
| 5. Feature Selection | MI + RF importance → top 35 | Final feature set |
| 6. Model Building | 5 models, 5-fold CV, SMOTE | Model comparison table |
| 7. Interpretability | SHAP analysis | Fraud explanation reports |
| 8. Prediction | Test set predictions | Tharun_Submission.csv |
| 9. Business Recs | Fraud patterns, prevention | Strategy document |

---

## 🤖 Model Performance (5-Fold Cross-Validation)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|-------|---------|--------|----|--------|-----------|
| **Random Forest** ⭐ | **0.9318** | 0.6381 | **0.5950** | **0.7925** | 0.4762 |
| Logistic Regression | 0.9310 | **0.6826** | 0.5596 | 0.8300 | 0.4221 |
| CatBoost | 0.9298 | 0.6661 | 0.5871 | 0.7292 | 0.4913 |
| LightGBM | 0.9272 | 0.6635 | 0.5957 | 0.6858 | 0.5266 |
| XGBoost | 0.9165 | 0.6443 | 0.5567 | 0.7668 | 0.4369 |

**Best Model: Random Forest** — chosen for highest ROC-AUC and strong recall (79.25%) 
which is critical in fraud detection to minimize missed fraud cases.

---

## 🔑 Top Fraud Indicators (SHAP)

1. **TotalReimbursement** — Inflated total billing amounts
2. **TotalHospitalDays** — Extended inpatient stays (ghost billing)
3. **TotalDeductible** — Excessive deductible patterns
4. **RepeatPatientRatio** — Recycling patients for duplicate claims
5. **ClaimsPerBeneficiary** — Excessive billing per patient
6. **InpatientRatio** — Preference for high-cost inpatient DRG codes
7. **AvgChronicCondCount** — Upcoding via complex chronic profiles
8. **PhysicianConcentration** — Physician ring/syndicate detection

---

## 🛡️ Business Recommendations

### Risk Tiering Framework
| Tier | Probability | Action |
|------|-------------|--------|
| 🔴 Critical | ≥ 0.70 | Payment hold + immediate investigation |
| 🟠 High | 0.50–0.69 | Enhanced auditing + site visits |
| 🟡 Watch | 0.30–0.49 | Quarterly review |
| 🟢 Low | < 0.30 | Standard processing |

### Top Fraud Patterns
- **Upcoding:** Billing complex chronic conditions to justify expensive procedures
- **Ghost Billing:** High inpatient claims for short or non-existent stays  
- **Repeat Billing:** Same patients across multiple claims (phantom services)
- **Physician Rings:** Small physician groups billing through one provider entity

---

## 📤 Submission File Format

`Tharun_Submission.csv` contains:
- `Provider` — Provider ID
- `Probability` — Fraud probability (0–1)
- `Predicted_Class` — Yes (Fraud) / No (Legitimate)

---

## 📚 References

1. CMS Medicare Fraud Resources: https://www.cms.gov/About-CMS/Components/CPI/index
2. FBI Healthcare Fraud Overview: https://www.fbi.gov/investigate/white-collar-crime/health-care-fraud
3. SHAP Documentation: https://shap.readthedocs.io
4. Imbalanced-learn (SMOTE): https://imbalanced-learn.org
5. US Healthcare Fraud Statistics: OIG HHS Annual Report 2023

---

*This submission is confidential and prepared exclusively for Sagility's Data Science evaluation. 
The dataset has not been published or shared externally.*
