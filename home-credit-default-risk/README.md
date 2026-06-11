# Credit Risk Scoring System
## Industrial ML Project — Day 1

### Problem Statement
Predict probability of loan default for 307,511 
applicants using real Home Credit dataset.

### Key Results
| Model | AUC |
|-------|-----|
| Logistic Regression | 0.6206 |
| Random Forest | 0.6887 |
| XGBoost | 0.7411 |
| LightGBM baseline | 0.7616 |
| LightGBM tuned | 0.7643 |

### Key Concepts Demonstrated
- Imbalanced classification (8.1% default rate)
- SMOTE inside Pipeline (prevents data leakage)
- Threshold tuning for business context
- SHAP explainability (GDPR compliance)
- Honest cross-validation (no leakage)

### Tech Stack
Python, LightGBM, XGBoost, SHAP, 
Scikit-learn, Imbalanced-learn

### Dataset
Home Credit Default Risk — Kaggle
https://www.kaggle.com/competitions/home-credit-default-risk