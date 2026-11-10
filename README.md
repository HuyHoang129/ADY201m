# Bank Term Deposit Subscription Prediction — A Comparative Study of Five Machine Learning Models on Imbalanced Telemarketing Data

Course project — FPT University, Ho Chi Minh City.

**Team members:** Duong Minh Khang · Nguyen Huy Hoang · Nguyen Minh Khoi

**Supervisor:** Le Nhat Tung | **Course:** ADY201m | **Semester:** Third

---

## 1. Overview

This project studies whether a bank can identify in advance which customers will subscribe to a term deposit, using the UCI Bank Marketing dataset (41,188 records, only 11.27% positive).

Five classifiers — Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM — are compared under an identical **leakage-aware pipeline**: stratified train/validation/test split (64/16/20), mutual-information feature selection, scaling, and SMOTE + Tomek Links fitted only inside each cross-validation training fold.

Two parallel scenarios are evaluated:

- **Scenario A (post-contact reference):** keeps `duration` — upper bound, not deployable.
- **Scenario B (deployable, pre-contact):** excludes `duration` — used for customer ranking.

## 2. Key results

| Item | Result |
|---|---|
| Selected model (by mean 5-fold CV PR-AUC, Scenario B) | LightGBM (tuned) |
| Decision threshold (validation-selected, Recall ≥ 0.50 check) | 0.617 |
| Held-out test PR-AUC / F1 / Recall / Precision | 0.4658 / 0.5071 / 0.5399 / 0.4781 |
| ROC-AUC | 0.8049 |
| Ranked calling: top 20% | captures 607/928 subscribers (**65.41%**) while avoiding ~80% of calls |

## 3. Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/                      # UCI Bank Marketing dataset (link or CSV)
├── ADY201m.ipynb              # full pipeline: EDA, 5 models, tuning, threshold
└── Bank_Term_Deposit_Subscription_Prediction....pdf   # final report (PDF)

```

## 4. How to run

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in `notebooks/` in order. All random states are fixed with `SEED = 42` for reproducibility.

## 5. Data

- Source: UCI Machine Learning Repository — [Bank Marketing dataset], compiled from a Portuguese bank telemarketing campaign (May 2008 – Nov 2010).
- 41,188 rows × 20 input features + binary target `y` (11.27% positive).
- No missing values; "unknown" is kept as a separate category.

## 6. Report

Bank_Term_Deposit_Subscription_Prediction__A_Comparative_Study_of_Five_Machine_Learning_Models_on_Imbalanced_Telemarketing_Data.pdf in the repository root.
