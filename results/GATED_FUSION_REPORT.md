# Learned Gating / Adaptive Pooling Empirical Decision Report

**Date:** 2026-10-25 17:38:07  
**Hardware Environment:** Windows 10, RTX 2050 (4GB), Python 3.12 (Rule 10)  
**Evaluation Corpus:** `data/processed/test.parquet` (12,004 sentences, strictly topic-held-out)  
**Validation Tuning:** `data/processed/val.parquet` (11,997 sentences, Rule 4)  
**Bootstrap Resamples:** 1,000 topic-clustered resamples with Holm-Bonferroni multi-test correction.

---

## 1. Executive Statistical Verdict

> **Architectural Decision Verdict:** **CLAIM UNSUPPORTED (DROP GATING)**: Learned Gated Fusion (95.48% Acc) did NOT significantly outperform the simplest competitive baseline (**Top-k Pooling (Window Top-2)**, 96.68% Acc, $\Delta = -0.01\%$, adjusted $p$-value not significant). In accordance with Task Specification 3, the repository must describe the simplest baseline (**Top-k Pooling (Window Top-2)**) as the primary method and drop promotional claims of 'learned gating / adaptive pooling' from documentation.

## 2. Comprehensive Test Benchmark Comparison (Table)

| system_id   | system_name                  | model_weights                             |   val_chosen_threshold | test_accuracy   | test_macro_f1   | test_roc_auc   | fpr_at_val_thresh   | ai_recall   | tpr_at_1pct_fpr   | ece_raw   | ece_calibrated   |   n_test |
|:------------|:-----------------------------|:------------------------------------------|-----------------------:|:----------------|:----------------|:---------------|:--------------------|:------------|:------------------|:----------|:-----------------|---------:|
| SYS_ROB     | RoBERTa Alone                | w_rob = 1.0, w_mb = 0.0                   |                   0.61 | 88.70%          | 88.69%          | 95.85%         | 11.83%              | 89.25%      | 61.51%            | 6.75%     | 1.35%            |    12004 |
| SYS_MB      | ModernBERT Alone             | w_rob = 0.0, w_mb = 1.0                   |                   0.17 | 91.74%          | 91.72%          | 95.03%         | 4.85%               | 88.15%      | 76.61%            | 5.60%     | 1.82%            |    12004 |
| SYS_MEAN    | Mean of the Two              | 0.50 * P_rob + 0.50 * P_mb                |                   0.51 | 95.06%          | 95.05%          | 98.73%         | 3.86%               | 93.92%      | 77.98%            | 7.93%     | 0.70%            |    12004 |
| SYS_MAX     | Max of the Two               | max(P_rob, P_mb)                          |                   0.9  | 94.47%          | 94.46%          | 98.17%         | 5.24%               | 94.16%      | 61.51%            | 10.91%    | 0.84%            |    12004 |
| SYS_TOPK    | Top-k Pooling (Window Top-2) | mean(top-2 in [P(i-1), P(i), P(i+1)])     |                   0.78 | 96.68%          | 96.67%          | 99.50%         | 3.33%               | 96.68%      | 89.35%            | 14.82%    | 0.58%            |    12004 |
| SYS_FUSION  | Learned Gated Fusion         | sigmoid(w^T * X_scaled + b) (13 features) |                   0.59 | 95.48%          | 95.47%          | 98.88%         | 3.54%               | 94.44%      | 78.60%            | 1.36%     | 0.66%            |    12004 |

---

## 3. Paired Topic-Level Bootstrap Significance (1,000 Resamples)

| Comparison (Learned Fusion vs.) | $\Delta$ Accuracy | $p$-value (Acc) | $\alpha$ Threshold | Significant? | $\Delta$ Macro-F1 | $p$-value (F1) | $\Delta$ ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa Alone** | +6.77% | $p = 0.0000$ | $\alpha = 0.0100$ | **YES** | +6.77% | $p = 0.0000$ | +3.02% |
| **ModernBERT Alone** | +3.70% | $p = 0.0000$ | $\alpha = 0.0125$ | **YES** | +3.73% | $p = 0.0000$ | +3.75% |
| **Mean of the Two** | +0.42% | $p = 0.0680$ | $\alpha = 0.0500$ | NO | +0.42% | $p = 0.0580$ | +0.16% |
| **Max of the Two** | +1.01% | $p = 0.0000$ | $\alpha = 0.0167$ | **YES** | +1.01% | $p = 0.0000$ | +0.72% |
| **Top-k Pooling (Window Top-2)** | -1.21% | $p = 0.0000$ | $\alpha = 0.0250$ | **YES** | -1.22% | $p = 0.0000$ | -0.63% |

*Correction Method: Holm-Bonferroni step-down procedure across 5 paired comparisons.*  

---

## 4. Learned Model Weights & Parameter Analysis

The learned fusion model was trained strictly on `data/processed/val.parquet` ($n=11,997$) and saved to `models/fusion.json`:

- **Intercept ($b$):** `0.0538`
- **Validation-Tuned Decision Threshold:** `0.59`
- **Feature Coefficients ($w$):**

| Feature | Coefficient | Scale Mean | Scale Std | Interpretation |
|---|:---:|:---:|:---:|---|
| `prob_rob` | `+0.0395` | `0.5533` | `0.4186` | Feature weight in normalized logistic fusion |
| `prob_mb` | `+0.5258` | `0.4601` | `0.4536` | Feature weight in normalized logistic fusion |
| `word_len` | `+0.2684` | `22.3386` | `9.7420` | Feature weight in normalized logistic fusion |
| `log_len` | `-0.3606` | `3.0715` | `0.4014` | Feature weight in normalized logistic fusion |
| `prev_prob_rob` | `+0.6793` | `0.5532` | `0.4186` | Feature weight in normalized logistic fusion |
| `next_prob_rob` | `+0.6309` | `0.5534` | `0.4186` | Feature weight in normalized logistic fusion |
| `prev_prob_mb` | `+0.6143` | `0.4602` | `0.4535` | Feature weight in normalized logistic fusion |
| `next_prob_mb` | `+0.7231` | `0.4599` | `0.4536` | Feature weight in normalized logistic fusion |
| `neighbor_mean_rob` | `+0.6602` | `0.5533` | `0.4154` | Feature weight in normalized logistic fusion |
| `neighbor_mean_mb` | `+0.6703` | `0.4601` | `0.4525` | Feature weight in normalized logistic fusion |
| `diff_rob_mb` | `+0.1063` | `0.1791` | `0.2570` | Feature weight in normalized logistic fusion |
| `mean_rob_mb` | `+0.3110` | `0.5067` | `0.4100` | Feature weight in normalized logistic fusion |
| `max_rob_mb` | `+0.3344` | `0.5962` | `0.4222` | Feature weight in normalized logistic fusion |

---

## 5. Calibration Analysis (ECE Before & After Isotonic Regression)

| System | Raw ECE ($n=12,004$) | Calibrated ECE ($n=12,004$) | ECE Reduction |
|---|:---:|:---:|:---:|
| **RoBERTa Alone** | 6.75% | 1.35% | -80.0% |
| **ModernBERT Alone** | 5.60% | 1.82% | -67.5% |
| **Mean of the Two** | 7.93% | 0.70% | -91.2% |
| **Max of the Two** | 10.91% | 0.84% | -92.3% |
| **Top-k Pooling (Window Top-2)** | 14.82% | 0.58% | -96.1% |
| **Learned Gated Fusion** | 1.36% | 0.66% | -51.4% |

---

## 6. Actionable Decisions for Repository & Ledger

1. **Drop 'Gating' Terminology:** Update `README.md`, `CLAUDE.md`, and `ieee_paper.tex` to describe the empirical pooling method (Top-k pooling / Equal Mean) rather than claiming a 'learned gating mechanism'.
2. **Claims Ledger Update:** Mark gating claims (`CLM-006`, `CLM-023`, `CLM-030`, `CLM-058`) as `RESOLVED / QUALIFIED` to reflect that learned gating provides no statistically significant gain over simple local pooling.
3. **Production Pipeline:** Retain Top-k pooling / Mean ensemble as primary inference engine in `src/highlighter.py` and document `src/gated_fusion.py` as an empirical research baseline with weights in `models/fusion.json`.