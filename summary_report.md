# 📝 Summary Report: Credit Card Fraud Detection

## 🎯 Business Problem and Why Accuracy Fails
A digital payments company processes millions of transactions, and only **0.17%** are fraudulent. A model that labels every transaction "Not Fraud" scores **99.83% accuracy** yet catches **0 frauds** (Recall = 0, Precision undefined). A missed fraud costs about ₹4,500 plus a chargeback and lost trust, while a false alarm costs only about ₹150 of analyst time. So I judged models on **PR-AUC**, Precision and Recall for the Fraud class, never on accuracy or ROC-AUC (which stays near 0.999 even for weak models).

## ⚖️ Imbalance Handling Strategy
I trained Logistic Regression on three training sets: original with `class_weight='balanced'`, **SMOTE** (fraud raised to 10% of majority) and **random undersampling** (also 10%). The test set was never resampled. **SMOTE won** (PR-AUC 0.8144, recall 1.00 with 38 false alarms), while class weights produced 245 false alarms and undersampling missed one fraud and discarded about 98% of the legitimate data. SMOTE was therefore used for Random Forest (PR-AUC 0.8720). XGBoost used its own `scale_pos_weight` (578.7) on the original data.

## 🏆 Recommended Model and Threshold
I recommend the **tuned XGBoost** pipeline (scaler + model) with the **F1-optimal threshold of 0.4746**.

| Model | Threshold | Precision | Recall | F1 | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (SMOTE) | 0.5000 | 0.3091 | 1.0000 | 0.4722 | 0.8144 |
| Random Forest | 0.5000 | 0.8750 | 0.8235 | 0.8485 | 0.8720 |
| XGBoost Baseline | 0.5000 | 1.0000 | 0.8235 | 0.9032 | 0.9012 |
| **XGBoost Tuned (F1-optimal)** | **0.4746** | **1.0000** | **0.8824** | **0.9375** | **0.9144** |
| XGBoost Tuned (Recall ≥ 0.90) | 0.0031 | 0.3404 | 0.9412 | 0.5000 | 0.9144 |

It has the best PR-AUC (0.9144, up from 0.9012 after tuning) and the best F1 (0.9375): it catches **15 of 17 frauds with zero false alarms**. The Recall ≥ 0.90 threshold catches one more fraud (16 of 17) but adds 31 false alarms, so it suits a manual review queue rather than automatic blocking.

## 💰 Cost-Benefit Result
Using ₹4,500 per fraud and ₹150 per investigated transaction, the model at threshold 0.4746 gives on the 10,000-transaction test set:
- Money saved: **₹67,500** (15 frauds caught)
- Investigation cost: **₹2,250** (15 reviews)
- Money lost: **₹9,000** (2 frauds missed)
- **Net benefit: ₹65,250**

Scaled to **50,000 transactions, the net benefit is about ₹3,26,250**. The business-optimal threshold equals the F1-optimal one here (thresholds 0.3 and 0.4746 both give ₹65,250), while 0.5 earns less (₹60,900) because it misses a fraud. This is based on only 17 test frauds, so it is indicative, not guaranteed.

## 🚀 Production Improvements
- **Real-time scoring:** serve the pickled pipeline behind a low-latency API and add a manual-review tier using the lower threshold.
- **Drift monitoring:** track PR-AUC, fraud rate and feature distributions (`Amount_log`, `Hour`, V-features) and alert on drift.
- **Feedback loop:** feed analyst decisions and chargebacks back as labels and retrain on a schedule.
- **Better validation:** choose thresholds on a separate time-based validation set, not the test set, and use far more than 17 frauds.
- **Cost-aware thresholds:** recompute the threshold whenever fraud value or review cost changes.
