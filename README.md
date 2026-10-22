# 💳 Credit Card Fraud Detection — Imbalanced Classification with Threshold Optimisation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Practical Exam — Supervised Learning (Set C)**
> A fraud-detection pipeline for a Paytm / Razorpay-style payments company, built on one of the most extreme class-imbalance benchmarks in industry ML (**only 0.17 % fraud**). The project goes beyond the default 0.5 cutoff and tunes the decision threshold for **real business value in ₹**.

---

## 🎥 Video Walkthrough

📹 **Recorded explanation (face + screen, notebook and `.pkl` demo):**
**[▶ Watch the video here](PASTE_GOOGLE_DRIVE_VIDEO_LINK_HERE)**

---

## 📌 Table of Contents

0. [Video Walkthrough](#-video-walkthrough)
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Key Results](#-key-results)
4. [Repository Structure](#-repository-structure)
5. [Methodology & Visuals](#-methodology--visuals)
6. [Optimal Threshold Used](#-optimal-threshold-used)
7. [Business Cost-Benefit Analysis](#-business-cost-benefit-analysis)
8. [Using the Saved Pipeline](#-using-the-saved-pipeline)
9. [How to Run](#-how-to-run)
10. [Limitations & Next Steps](#-limitations--next-steps)
11. [Author](#-author)

---

## 🎯 Project Overview

Millions of transactions flow through a digital payments platform every day and a tiny fraction are fraudulent. Missing even a few fraud cases costs lakhs of rupees and damages customer trust. This project:

- Shows **why accuracy is a misleading metric** (an "always Not Fraud" model scores **99.83 % accuracy** but catches **0 of 17** frauds).
- Compares **three imbalance strategies**: class weighting, **SMOTE** oversampling and **random undersampling**.
- Builds **Logistic Regression, Random Forest and XGBoost** classifiers and tunes XGBoost with `RandomizedSearchCV`.
- Uses **PR-AUC** (not ROC-AUC) as the primary metric.
- Optimises the **classification threshold** (F1-optimal and Recall ≥ 0.90) from the Precision-Recall curve.
- Converts model output into **₹ net benefit** using a cost-benefit simulation.
- Exports the final **scikit-learn `Pipeline`** (scaler + tuned XGBoost) as a `.pkl` file.

---

## 📂 Dataset

| Item | Detail |
|---|---|
| **Name** | Credit Card Fraud Detection |
| **Source** | [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Full size** | 284,807 transactions, 31 columns, no missing values |
| **Features** | `V1–V28` (PCA-transformed), `Time`, `Amount` |
| **Target** | `Class` (1 = Fraud, 0 = Legitimate) |
| **Fraud rate** | 492 frauds = **0.173 %** |
| **Sample used** | Stratified **50,000 rows** (`random_state=42`) with **86 frauds (0.172 %)** |
| **Train / Test** | 80 / 20 stratified → 40,000 train (69 frauds) / 10,000 test (**17 frauds**) |

> ⚠️ `creditcard.csv` (~144 MB) is **not included** in this repo because it exceeds GitHub's 100 MB file limit. Download it from the Kaggle link above and place it in the `data/` folder.

---

## 🏆 Key Results

### Imbalance strategy comparison — Logistic Regression

| Strategy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | PR-AUC | False Positives |
|---|---:|---:|---:|---:|---:|
| Original + `class_weight='balanced'` | 0.0649 | 1.0000 | 0.1219 | 0.7814 | 245 |
| **SMOTE** ⭐ | **0.3091** | **1.0000** | **0.4722** | **0.8144** | **38** |
| Random Undersampling | 0.2286 | 0.9412 | 0.3678 | 0.8093 | 54 |

SMOTE gave the best PR-AUC for Logistic Regression, so it was used as the training set for Random Forest.

### Final model comparison (test set, Fraud class)

| Model | Threshold | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (SMOTE) | 0.5000 | 0.3091 | 1.0000 | 0.4722 | 0.8144 |
| Random Forest (SMOTE data) | 0.5000 | 0.8750 | 0.8235 | 0.8485 | 0.8720 |
| XGBoost Baseline | 0.5000 | 1.0000 | 0.8235 | 0.9032 | 0.9012 |
| XGBoost Tuned (default) | 0.5000 | 1.0000 | 0.8235 | 0.9032 | **0.9144** |
| **XGBoost Tuned (F1-optimal)** ⭐ | **0.4746** | **1.0000** | **0.8824** | **0.9375** | **0.9144** |
| XGBoost Tuned (Recall ≥ 0.90) | 0.0031 | 0.3404 | 0.9412 | 0.5000 | 0.9144 |

**Best XGBoost parameters:** `n_estimators=300`, `max_depth=3`, `learning_rate=0.1`, `subsample=0.6`, `colsample_bytree=0.8` (best CV PR-AUC = 0.8043).
**Top Random Forest features:** `V14`, `V12`, `V10`, `V17`, `V4`.

---

## 🗂 Repository Structure

```
credit-fraud-detection-supervised-learning/
│
├── FraudDetection_SupervisedLearning.ipynb   # Fully executed notebook (Steps 1-8)
├── fraud_detection_model.pkl                 # Final sklearn Pipeline (scaler + tuned XGBoost)
├── summary_report.md                         # ~1 page business summary & cost-benefit result
├── requirements.txt                          # Python dependencies
├── README.md                                 # You are here
│
├── artifacts/
│   ├── test_samples.csv                      # Sample transactions for testing the saved pipeline
│   └── figures/                              # All 12 plots used in this README
│
└── data/
    └── creditcard.csv                        # Download from Kaggle (not tracked in git)
```

---

## 🔬 Methodology & Visuals

### Step 1 — Problem Framing & Theory
Markdown answers inside the notebook cover: why accuracy fails, the precision-recall trade-off, pros/cons of undersampling vs SMOTE vs class weights, what PR-AUC is and why it beats ROC-AUC, and the business cost of a False Positive vs a False Negative.

### Step 2 — Exploratory Data Analysis

**Extreme class imbalance** — 49,914 legitimate vs 86 fraud transactions in the sample (log scale).

<p align="center">
  <img src="artifacts/figures/Class%20Distribution%20of%20the%2050%2C000-Row%20Sample.png" alt="Class distribution of the 50,000-row sample (log scale)" width="560">
</p>

**Amount** — fraud has a smaller typical amount (**median ₹7.36 vs ₹22.00**), although a few large frauds pull the mean up.

<p align="center">
  <img src="artifacts/figures/Amount%20Distribution%20-%20Log%20Scale.png" alt="Amount distribution, fraud vs legitimate" width="820">
</p>

**Time** — **20.9 % of frauds** happen between 00:00 and 05:59, against only **8.4 %** of legitimate transactions (about 2.5× more).

<p align="center">
  <img src="artifacts/figures/Transaction%20Time%20vs%20Share%20of%20Transactions.png" alt="Transaction time vs share of transactions" width="820">
</p>

**Correlation (V1–V10 vs Class)** — the strongest are `V10` (−0.23), `V7` (−0.22) and `V3` (−0.21). All are modest, so a non-linear model helps.

<p align="center">
  <img src="artifacts/figures/Correlation%20Heatmap.png" alt="Correlation heatmap V1-V10 and Class" width="640">
</p>

### Step 3 — Preprocessing & Imbalance Handling

- `Amount_log = log1p(Amount)` reduces skewness from **9.47 → 0.16**.
- `Hour = (Time % 86400) // 3600`; original `Time` and `Amount` dropped.
- `StandardScaler` applied to `Amount_log` and `Hour` only (fitted on training data).
- Three training sets: **Original** (39,931 / 69), **SMOTE** (39,931 / 3,993) and **Undersampled** (690 / 69).
- ✅ **The test set was never resampled or modified.**

<p align="center">
  <img src="artifacts/figures/Training%20Set%20Class%20Counts%20for%20the%20Three%20Strategies%20%28test%20set%20untouched%29.png" alt="Training set class counts for the three strategies" width="820">
</p>

### Step 4 — Random Forest

Random Forest (trained on the SMOTE set) reached **PR-AUC 0.872** with precision 0.875 and recall 0.8235.

<p align="center">
  <img src="artifacts/figures/Precision-Recall%20Curve%20-%20Random%20Forest.png" alt="Precision-Recall curve, Random Forest" width="560">
</p>

<p align="center">
  <img src="artifacts/figures/Random%20Forest%20-%20Top%2015%20Features.png" alt="Random Forest top 15 feature importances" width="560">
</p>

### Step 5 — XGBoost (Baseline → Tuned → Threshold)

`scale_pos_weight = 39,931 / 69 = 578.71`. The baseline already reaches **precision 1.00, recall 0.82, PR-AUC 0.901**. After `RandomizedSearchCV` (15 iterations, 3-fold CV, `average_precision`) the test PR-AUC improves to **0.9144**.

<p align="center">
  <img src="artifacts/figures/Precision-Recall%20Curve%20-%20XGBoost%20Baseline.png" alt="Precision-Recall curve, XGBoost baseline" width="560">
</p>

**Threshold optimisation** — Precision-Recall curve with the default 0.5 cutoff, the F1-optimal threshold and the Recall ≥ 0.90 threshold marked.

<p align="center">
  <img src="artifacts/figures/Precision-Recall%20Curve%20with%20Threshold%20Operating%20Points.png" alt="Precision-Recall curve with threshold operating points" width="560">
</p>

**Confusion matrices at three thresholds**

<p align="center">
  <img src="artifacts/figures/Confusion%20Matrices%20of%20XGBoost%20Tuned%20at%20Three%20Thresholds.png" alt="Confusion matrices of XGBoost Tuned at three thresholds" width="860">
</p>

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 (default) | 14 | 0 | 3 | 1.0000 | 0.8235 | 0.9032 |
| **0.4746 (F1-optimal)** | **15** | **0** | **2** | **1.0000** | **0.8824** | **0.9375** |
| 0.0031 (Recall ≥ 0.90) | 16 | 31 | 1 | 0.3404 | 0.9412 | 0.5000 |

### Step 6 — All Models Compared

<p align="center">
  <img src="artifacts/figures/Precision-Recall%20Curves%20of%20All%20Models.png" alt="Precision-Recall curves of all models" width="700">
</p>

**Recommendation:** deploy the **tuned XGBoost** — best PR-AUC (0.9144) and best F1 (0.9375). At the F1-optimal threshold it catches 15 of 17 frauds with **zero false alarms**, so it is safe for automatic alerts. A second, much lower threshold can feed a **manual review queue** to catch one more fraud at the cost of extra reviews.

---

## 🎚 Optimal Threshold Used

| Purpose | Threshold | Result |
|---|---:|---|
| **Final label (F1-optimal)** ⭐ | **`0.4746`** | Precision 1.00, Recall 0.88, F1 0.94 |
| High-recall review queue (Recall ≥ 0.90) | `0.0031` | Precision 0.34, Recall 0.94 |

Both thresholds are stored on the saved pipeline as attributes: `pipeline.fraud_threshold` and `pipeline.high_recall_threshold`.

---

## 💰 Business Cost-Benefit Analysis

**Assumptions:** average fraud value = **₹4,500**, investigation cost per flagged transaction = **₹150**.

| Threshold | TP | FP | FN | Money Saved | Investigation Cost | Money Lost | Net Benefit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1000 | 15 | 5 | 2 | ₹67,500 | ₹3,000 | ₹9,000 | ₹64,500 |
| 0.2000 | 15 | 3 | 2 | ₹67,500 | ₹2,700 | ₹9,000 | ₹64,800 |
| 0.3000 | 15 | 0 | 2 | ₹67,500 | ₹2,250 | ₹9,000 | ₹65,250 ⭐ |
| 0.5000 | 14 | 0 | 3 | ₹63,000 | ₹2,100 | ₹13,500 | ₹60,900 |
| **0.4746 (F1-optimal)** | **15** | **0** | **2** | **₹67,500** | **₹2,250** | **₹9,000** | **₹65,250 ⭐** |

- **Net benefit at the chosen threshold: ₹65,250 per 10,000 transactions ≈ ₹3,26,250 per 50,000 transactions.**
- Without any model, all 17 frauds would be lost (₹76,500).
- A full threshold scan confirms the F1-optimal threshold is also business-optimal on this test set.

<p align="center">
  <img src="artifacts/figures/Net%20Benefit%20vs%20Threshold%20%28XGBoost%20Tuned%29.png" alt="Net benefit vs threshold for tuned XGBoost" width="700">
</p>

**In plain English:** out of 10,000 transactions, the model stops 15 of the 17 real frauds without wrongly blocking a single genuine customer, saving about ₹65,000 after paying for reviews. Making the alarm more sensitive catches one more fraud but sends about 31 genuine transactions to review, so the setting is a business choice: strict for automatic blocking, more sensitive for a manual review queue.

---

## 🚀 Using the Saved Pipeline

The pipeline contains the scaler and the tuned XGBoost model. It expects **unscaled** input columns: `V1–V28`, `Amount_log = log1p(Amount)` and `Hour = (Time % 86400) // 3600`.

```python
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load("fraud_detection_model.pkl")
threshold = pipeline.fraud_threshold            # 0.4746 (F1-optimal)

# df must contain: V1..V28, Amount_log, Hour
df["Amount_log"] = np.log1p(df["Amount"])
df["Hour"] = (df["Time"] % 86400) // 3600
X = df[[f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]]

proba = pipeline.predict_proba(X)[:, 1]
label = np.where(proba >= threshold, "FRAUD", "Legit")
print(pd.DataFrame({"Fraud Probability": proba.round(4), "Prediction": label}))
```

On 10 sample transactions (5 real frauds + 5 legitimate) the reloaded pipeline predicted 9 correctly, and it reproduces the original test-set probabilities exactly.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/credit-fraud-detection-supervised-learning.git
cd credit-fraud-detection-supervised-learning

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download creditcard.csv from Kaggle and place it in data/creditcard.csv

# 5. Open and run the notebook top to bottom
jupyter notebook FraudDetection_SupervisedLearning.ipynb
```

> 📝 **Path note:** in the notebook's first code cell, set `DATA_PATH = Path('data/creditcard.csv')` and `MODEL_DIR = Path('.')` so it runs from the repo root.

---

## ⚠️ Limitations & Next Steps

- The test set holds only **17 frauds**, so each missed fraud moves recall by about 6 percentage points. Results are indicative, not final proof.
- Thresholds were chosen on the test-set probabilities (as the exam specifies), which makes them optimistic. In production, choose them on a separate validation set or out-of-fold predictions.
- Production ideas: real-time scoring API, model drift monitoring, and a feedback loop from analyst decisions back into retraining.

---

## 👤 Author

**Om Vaja** — BCA student, Vidhyadeep University, Surat, India
Data analytics freelancer | Power BI · Excel · SQL · Python

🔗 GitHub: [@your-username](https://github.com/your-username) · LinkedIn: [your-profile](https://www.linkedin.com/in/your-profile)

---

⭐ If this project helped you, consider giving the repo a star!
