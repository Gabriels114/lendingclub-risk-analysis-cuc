# Loan Default Prediction – Report (Draft)

**Course:** Data Mining – Universidad de la Costa  
**Assignment:** Evaluation II – LendingClub Loan Default Prediction  
**Team Members:** _Add names here_

## 1. Dataset & Business Context
We predict whether a loan will be **Charged Off** (default) at origination to help minimize financial losses (reduce **false positives**) while avoiding rejecting reliable borrowers (reduce **false negatives**).

**Target:** `default = 1` if `loan_status == 'Charged Off'`, else `0` (Fully Paid).  
**Class balance:** Highly imbalanced (~80/20 in the full dataset).

## 2. Cleaning & Feature Engineering
- Parsed `term` → `term_months` and `emp_length` → `emp_length_years`.
- Derived `credit_age_years`, `issue_year`, `issue_month` from dates.
- Dropped noisy/high‑cardinality text columns (`title`, `emp_title`, `address`) and redundant (`grade`) keeping `sub_grade`.
- Imputation: median (numeric) / most frequent (categorical).

## 3. EDA & Correlation
- Descriptive statistics for numeric features.
- Class balance chart.
- Correlation heatmap (numeric) to identify potential multicollinearity.

## 4. Feature Selection
- **Mutual Information (MI)** to select top‑K predictive features as a guardrail against noise and multicollinearity.

## 5. Preprocessing & Split
- `ColumnTransformer` with scaling for numeric and one‑hot encoding for categorical.
- `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.

## 6–8. Models, Hyperparameters & Tuning
- **Logistic Regression** (`C` grid; class_weight='balanced').
- **Random Forest** (`n_estimators`, `max_depth`, `min_samples_*`, `max_features`; class_weight='balanced_subsample').
- **HistGradientBoosting** (`max_depth`, `learning_rate`, `max_iter`, `l2_regularization`).

Tuning via `RandomizedSearchCV` (3‑fold stratified) on the training set; select best estimator per model by ROC‑AUC.

## 9. Evaluation (Quick demo run on 3k subsample)
The full notebook runs on the entire dataset with full tuning. Below is a **quick demo** trained here on a 3k subsample (to fit time limits), reporting **class 1 (default)** metrics:

| Model   |   Precision(1) |   Recall(1) |    F1(1) |   ROC-AUC |
|:--------|---------------:|------------:|---------:|----------:|
| LogReg  |       0.336032 |   0.674797  | 0.448649 |  0.731281 |
| RF      |       0.533333 |   0.0650407 | 0.115942 |  0.708459 |

**Metrics:** Precision/Recall/F1 by class, Confusion Matrix, and ROC‑AUC. For business use, consider threshold calibration to trade off precision vs. recall on class 1.

## Discussion
1. **Best balance:** In preliminary subsample tests, Logistic Regression yielded higher recall on defaults, while Random Forest favored precision. After full tuning on the entire dataset (notebook), pick the model (or calibrated ensemble) that best meets risk policy.
2. **Effect of tuning:** RF depth/trees control overfitting vs. variance; LR `C` trades bias for variance; HGB learning rate/iterations impact convergence and generalization.
3. **Conclusion for LendingClub:** Use the selected model with thresholding and risk tiers: e.g., send high‑risk scores to **manual review**, auto‑approve low‑risk, and require **additional documentation** for the middle band. Monitor drift and retrain periodically.

---
**Artifacts generated:**  
- Notebook: `LendingClub_Default_Prediction.ipynb`  
- Quick metrics JSON: `lc_outputs/mini_results.json`

