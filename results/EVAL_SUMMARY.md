# Comprehensive Detector Evaluation & Statistical Validation Report

**Date:** 2026-10-24 16:15:36  
**Git Commit Hash:** `e338d64df2c18a8a572f7476623dddf82ffbab07`  
**Random Seed:** `42` (Bootstrap Resamples: `1000`)  
**Validation Split:** `data/processed/val.parquet` (11,997 sentences)  
**Test Split:** `data/processed/test.parquet` (12,004 sentences, topic-split, never trained on)  
**Evaluated Models:**
- RoBERTa Sentence Model: `models/roberta-sentence-academic-v2`
- ModernBERT Macro Context Model: `models/modernbert-academic`

---

## 1. System Performance Comparison Table (test.parquet)
*All thresholds tuned exclusively on `val.parquet` to maximize Macro-F1. Evaluated on strictly held-out `test.parquet`.*

| System | Val Thresh | Acc (%) | Macro-F1 (%) | ROC-AUC (%) | PR-AUC (%) | FPR (%) | TPR @ 1% FPR (%) | TPR @ 5% FPR (%) | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa Alone (Raw)** | `0.61` | 88.70% | 88.69% | 95.85% | 96.02% | 11.83% | 61.51% | 80.12% | 0.0675 |
| **RoBERTa Alone (Smoothed)** | `0.61` | 97.13% | 97.12% | 99.59% | 99.55% | 2.69% | 92.98% | 98.77% | 0.1297 |
| **ModernBERT Alone (Raw)** | `0.17` | 91.74% | 91.72% | 95.03% | 96.42% | 4.85% | 76.61% | 88.26% | 0.0560 |
| **ModernBERT Alone (Smoothed)** | `0.34` | 93.48% | 93.46% | 97.35% | 98.00% | 3.33% | 84.64% | 91.80% | 0.0545 |
| **Ensemble (Code Today, Raw)** | `0.90` | 94.34% | 94.33% | 97.96% | 97.75% | 5.63% | 57.25% | 93.25% | 0.1052 |
| **Ensemble (Code Today, Smoothed)** | `0.67` | 98.03% | 98.02% | 99.78% | 99.77% | 2.64% | 95.89% | 99.37% | 0.1154 |
| **Paper Formula (Max-Pool, Raw)** | `0.95` | 94.21% | 94.21% | 99.30% | 99.29% | 9.52% | 89.47% | 96.22% | 0.2291 |
| **Paper Formula (Max-Pool, Smoothed)** | `0.95` | 96.58% | 96.58% | 99.59% | 99.57% | 4.49% | 90.17% | 98.13% | 0.2291 |
| **Paper Formula (Mean-Pool, Raw)** | `0.61` | 97.13% | 97.13% | 99.59% | 99.55% | 2.68% | 92.98% | 98.82% | 0.1298 |
| **Paper Formula (Mean-Pool, Smoothed)** | `0.59` | 98.59% | 98.59% | 99.87% | 99.86% | 1.56% | 97.71% | 99.76% | 0.1471 |
| **Paper Formula (Top-K Pool, Raw)** | `0.78` | 96.68% | 96.67% | 99.50% | 99.48% | 3.33% | 89.35% | 98.05% | 0.1482 |
| **Paper Formula (Top-K Pool, Smoothed)** | `0.73` | 98.08% | 98.08% | 99.85% | 99.84% | 2.97% | 97.11% | 99.66% | 0.1568 |

---

## 2. Topic-Level 95% Bootstrap Confidence Intervals (1,000 Resamples)
*Resampling clustered at the `topic_id` level to guarantee zero intra-topic label leakage.*

| System | Val Thresh | Accuracy (95% CI) | Macro-F1 (95% CI) | ROC-AUC (95% CI) | FPR (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa Alone (Raw)** | `0.61` | [87.57%, 89.74%] | [87.55%, 89.73%] | [95.20%, 96.45%] | [9.35%, 14.39%] |
| **RoBERTa Alone (Smoothed)** | `0.61` | [96.60%, 97.64%] | [96.59%, 97.64%] | [99.44%, 99.77%] | [1.40%, 3.99%] |
| **ModernBERT Alone (Raw)** | `0.17` | [87.47%, 94.84%] | [87.41%, 94.84%] | [91.54%, 98.32%] | [3.90%, 5.83%] |
| **ModernBERT Alone (Smoothed)** | `0.34` | [89.38%, 96.65%] | [89.31%, 96.65%] | [95.61%, 99.25%] | [2.59%, 4.05%] |
| **Ensemble (Code Today, Raw)** | `0.90` | [92.87%, 95.72%] | [92.86%, 95.72%] | [97.19%, 98.68%] | [3.99%, 7.34%] |
| **Ensemble (Code Today, Smoothed)** | `0.67` | [97.60%, 98.47%] | [97.59%, 98.46%] | [99.66%, 99.87%] | [2.07%, 3.18%] |
| **Paper Formula (Max-Pool, Raw)** | `0.95` | [92.13%, 96.11%] | [92.13%, 96.11%] | [99.03%, 99.55%] | [5.61%, 13.38%] |
| **Paper Formula (Max-Pool, Smoothed)** | `0.95` | [95.44%, 97.65%] | [95.44%, 97.64%] | [99.41%, 99.78%] | [2.23%, 6.89%] |
| **Paper Formula (Mean-Pool, Raw)** | `0.61` | [96.60%, 97.65%] | [96.60%, 97.65%] | [99.44%, 99.78%] | [1.37%, 3.96%] |
| **Paper Formula (Mean-Pool, Smoothed)** | `0.59` | [98.09%, 99.06%] | [98.08%, 99.06%] | [99.80%, 99.95%] | [0.57%, 2.64%] |
| **Paper Formula (Top-K Pool, Raw)** | `0.78` | [95.95%, 97.38%] | [95.94%, 97.37%] | [99.30%, 99.72%] | [1.61%, 5.08%] |
| **Paper Formula (Top-K Pool, Smoothed)** | `0.73` | [97.15%, 98.95%] | [97.15%, 98.95%] | [99.76%, 99.94%] | [1.24%, 4.77%] |

---

## 3. Paired Bootstrap Significance Tests (1,000 Resamples)
*Measures metric differences ($\Delta = \text{System A} - \text{System B}$) evaluated on identical resampled datasets. p-values are empirical two-tailed tests.*

| Comparison (System A vs System B) | $\Delta$ Accuracy (95% CI) | p-value | $\Delta$ FPR (95% CI) | p-value | $\Delta$ TPR @ 1% FPR (95% CI) | p-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ensemble vs RoBERTa (Smoothed)** | +0.89% [+0.63%, +1.16%] | `< 0.001` | -0.05% [-0.97%, +0.85%] | `0.9380` | +3.09% [+0.98%, +5.58%] | `< 0.001` |
| **Ensemble vs ModernBERT (Smoothed)** | +4.53% [+1.75%, +8.29%] | `< 0.001` | -0.67% [-1.76%, +0.42%] | `0.2480` | +10.98% [+3.69%, +19.66%] | `< 0.001` |
| **Top-K vs RoBERTa (Smoothed)** | +0.94% [+0.43%, +1.43%] | `< 0.001` | +0.29% [-0.25%, +0.89%] | `0.3600` | +4.48% [+2.66%, +7.26%] | `< 0.001` |
| **Top-K vs ModernBERT (Smoothed)** | +4.58% [+2.22%, +7.93%] | `< 0.001` | -0.34% [-2.63%, +1.98%] | `0.7620` | +12.37% [+5.09%, +22.04%] | `< 0.001` |
| **Mean-Pool vs Max-Pool (Smoothed)** | +2.03% [+1.37%, +2.73%] | `< 0.001` | -2.95% [-4.42%, -1.60%] | `< 0.001` | +7.25% [+5.26%, +8.90%] | `< 0.001` |
| **Top-K Pool (Smoothed vs Raw)** | +1.39% [+1.08%, +1.71%] | `< 0.001` | -0.35% [-0.60%, -0.11%] | `0.0020` | +7.58% [+4.65%, +10.99%] | `< 0.001` |
| **Ensemble Code (Smoothed vs Raw)** | +3.69% [+2.64%, +4.77%] | `< 0.001` | -3.00% [-4.31%, -1.76%] | `< 0.001` | +38.55% [+29.95%, +43.22%] | `< 0.001` |
| **RoBERTa (Smoothed vs Raw)** | +8.43% [+7.67%, +9.18%] | `< 0.001` | -9.14% [-10.51%, -7.76%] | `< 0.001` | +30.96% [+27.17%, +34.41%] | `< 0.001` |
| **ModernBERT (Smoothed vs Raw)** | +1.73% [+1.36%, +2.09%] | `< 0.001` | -1.53% [-2.06%, -1.03%] | `< 0.001` | +8.16% [+6.51%, +10.03%] | `< 0.001` |
| **Mean-Pool (Smoothed vs Raw)** | +1.45% [+1.23%, +1.69%] | `< 0.001` | -1.12% [-1.49%, -0.71%] | `< 0.001` | +5.04% [+3.19%, +7.78%] | `< 0.001` |

---

## 4. Probability Calibration (Temperature Scaling & Isotonic Regression)
*Calibrators fitted on `val.parquet` only; thresholds re-selected on calibrated val probabilities.*

| System | Uncalibrated ECE | Temperature Scaled ECE | Isotonic Regression ECE | ECE Reduction (%) |
|---|:---:|:---:|:---:|:---:|
| **RoBERTa Alone (Raw)** | `0.0675` | `0.0161` | `0.0135` | **-80.0%** |
| **RoBERTa Alone (Smoothed)** | `0.1297` | `0.0056` | `0.0082` | **-93.7%** |
| **ModernBERT Alone (Raw)** | `0.0560` | `0.0237` | `0.0182` | **-67.5%** |
| **ModernBERT Alone (Smoothed)** | `0.0545` | `0.0140` | `0.0138` | **-74.7%** |
| **Ensemble (Code Today, Raw)** | `0.1052` | `0.0239` | `0.0071` | **-93.2%** |
| **Ensemble (Code Today, Smoothed)** | `0.1154` | `0.0088` | `0.0055` | **-95.2%** |
| **Paper Formula (Max-Pool, Raw)** | `0.2291` | `0.0168` | `0.0087` | **-96.2%** |
| **Paper Formula (Max-Pool, Smoothed)** | `0.2291` | `0.0114` | `0.0068` | **-97.0%** |
| **Paper Formula (Mean-Pool, Raw)** | `0.1298` | `0.0056` | `0.0083` | **-93.6%** |
| **Paper Formula (Mean-Pool, Smoothed)** | `0.1471` | `0.0045` | `0.0046` | **-96.9%** |
| **Paper Formula (Top-K Pool, Raw)** | `0.1482` | `0.0141` | `0.0058` | **-96.1%** |
| **Paper Formula (Top-K Pool, Smoothed)** | `0.1568` | `0.0059` | `0.0052` | **-96.7%** |

Reliability diagrams saved to `results/figures/calibration_curves.png` and `.pdf`.

---

## 5. Synthetic Mixed-Authorship Benchmark (Testing Smoothing Reality)
*Constructed strictly from `test.parquet`. Paragraphs of length $L=10$ with $k$% human sentences replaced by AI sentences from the same topic.*

| System | AI Replace % ($k$) | Window Size ($W$) | Sentence Macro-F1 | Inserted AI Recall | Untouched Human FPR | Document Detection Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa Alone** | 10% | $W=1$ | 62.12% | 89.61% | 27.67% | 97.56% |
| **RoBERTa Alone** | 10% | $W=3$ | 50.05% | 32.63% | 25.31% | 49.35% |
| **RoBERTa Alone** | 10% | $W=5$ | 48.95% | 25.97% | 23.81% | 32.47% |
| **Ensemble (Code Today)** | 10% | $W=1$ | 64.84% | 93.67% | 25.27% | 97.56% |
| **Ensemble (Code Today)** | 10% | $W=3$ | 48.45% | 21.43% | 22.06% | 28.25% |
| **Ensemble (Code Today)** | 10% | $W=5$ | 48.82% | 21.43% | 21.32% | 23.86% |
| **Paper Formula (Top-K Pool)** | 10% | $W=1$ | 67.23% | 96.75% | 23.18% | 97.24% |
| **Paper Formula (Top-K Pool)** | 10% | $W=3$ | 49.04% | 25.97% | 23.63% | 31.33% |
| **Paper Formula (Top-K Pool)** | 10% | $W=5$ | 48.52% | 23.38% | 23.12% | 25.16% |
| **RoBERTa Alone** | 25% | $W=1$ | 70.97% | 89.12% | 27.78% | 100.00% |
| **RoBERTa Alone** | 25% | $W=3$ | 56.03% | 44.72% | 28.23% | 71.92% |
| **RoBERTa Alone** | 25% | $W=5$ | 52.52% | 33.12% | 26.44% | 52.11% |
| **Ensemble (Code Today)** | 25% | $W=1$ | 74.05% | 93.83% | 25.49% | 100.00% |
| **Ensemble (Code Today)** | 25% | $W=3$ | 51.42% | 25.57% | 22.42% | 37.99% |
| **Ensemble (Code Today)** | 25% | $W=5$ | 49.66% | 20.94% | 21.61% | 26.14% |
| **Paper Formula (Top-K Pool)** | 25% | $W=1$ | 76.80% | 97.24% | 23.19% | 100.00% |
| **Paper Formula (Top-K Pool)** | 25% | $W=3$ | 52.67% | 31.41% | 24.88% | 47.89% |
| **Paper Formula (Top-K Pool)** | 25% | $W=5$ | 50.59% | 25.08% | 23.58% | 30.68% |
| **RoBERTa Alone** | 50% | $W=1$ | 80.78% | 89.09% | 27.27% | 100.00% |
| **RoBERTa Alone** | 50% | $W=3$ | 63.90% | 68.51% | 40.55% | 99.51% |
| **RoBERTa Alone** | 50% | $W=5$ | 57.43% | 58.38% | 43.51% | 96.92% |
| **Ensemble (Code Today)** | 50% | $W=1$ | 84.54% | 94.09% | 24.74% | 100.00% |
| **Ensemble (Code Today)** | 50% | $W=3$ | 56.44% | 39.32% | 23.41% | 84.74% |
| **Ensemble (Code Today)** | 50% | $W=5$ | 48.91% | 25.88% | 20.78% | 46.75% |
| **Paper Formula (Top-K Pool)** | 50% | $W=1$ | 86.84% | 97.27% | 23.31% | 100.00% |
| **Paper Formula (Top-K Pool)** | 50% | $W=3$ | 60.01% | 50.84% | 30.10% | 93.83% |
| **Paper Formula (Top-K Pool)** | 50% | $W=5$ | 54.96% | 42.53% | 31.04% | 80.52% |

---

## 6. Heuristic Rule Ablation & Human Transition Vulnerability
*Auditing hand-written rules in `src/highlighter.py` against formal human writing.*

| Ablation Condition | Val Thresh | Overall Acc (%) | Overall FPR (%) | Formal Human Transition FPR (%) | Normal Human FPR (%) | Net Transition Disparity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All Heuristics ON (Production Ensemble)** | `0.90` | 94.34% | 5.63% | **10.62%** | 5.44% | **+5.18%** |
| **No Humanizer Transition Boost (Rule 1 OFF)** | `0.90` | 94.40% | 5.39% | **3.98%** | 5.44% | **+-1.46%** |
| **No Burstiness Stylometric Boost (Rule 2 OFF)** | `0.91` | 94.34% | 5.18% | **10.18%** | 4.99% | **+5.19%** |
| **No Metadata Guard (Rule 4 OFF)** | `0.90` | 94.34% | 5.63% | **10.62%** | 5.44% | **+5.18%** |
| **No Non-Linear Gating (Pure Soft Voting)** | `0.51` | 92.87% | 8.29% | **100.00%** | 4.80% | **+95.20%** |
| **All Heuristics OFF (Pure Neural Models)** | `0.51` | 95.06% | 3.86% | **3.54%** | 3.87% | **+-0.33%** |

---

## 7. Out-of-Distribution Generalization (Generator & Domain Breakdown)

### By Generator Architecture (Zero-Shot Detection on `test.parquet`)
| Generator | Sample Count | Detection Recall (%) | Mean AI Probability | Threshold |
|---|:---:|:---:|:---:|:---:|
| **ChatGPT** | 500 | 100.00% | 0.992 | `0.73` |
| **Claude** | 1,279 | 99.61% | 0.982 | `0.73` |
| **Command** | 27 | 100.00% | 0.985 | `0.73` |
| **Falcon** | 56 | 100.00% | 0.989 | `0.73` |
| **GPT-5** | 1,154 | 97.14% | 0.939 | `0.73` |
| **GPT-OSS-120B** | 1,062 | 99.72% | 0.979 | `0.73` |
| **Gemini** | 1,083 | 99.72% | 0.982 | `0.73` |
| **Llama** | 200 | 100.00% | 0.988 | `0.73` |
| **Mistral** | 89 | 98.88% | 0.986 | `0.73` |
| **Other_AI** | 186 | 72.58% | 0.821 | `0.73` |
| **PaLM** | 264 | 99.62% | 0.990 | `0.73` |

### By Academic Domain
| Domain | Sample Count | Accuracy (%) | Macro-F1 (%) | ROC-AUC (%) | FPR (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **academic** | 9,215 | 98.87% | 98.87% | 99.93% | 1.31% |
| **essay** | 1,891 | 93.28% | 93.28% | 99.65% | 12.03% |
| **qa** | 898 | 100.00% | 100.00% | 100.00% | 0.00% |

---

## 8. Automated Empirical Findings (Zero Hand-Written Claims)

All analytical findings below are derived directly by code from the empirical evaluation tables:

1. **Top Performing System:**
   - The highest overall accuracy on strictly held-out `test.parquet` was achieved by **`Paper Formula (Mean-Pool, Smoothed)`** at **98.59%** (95% CI [98.09%, 99.06%]).
   - The lowest false positive rate on human text was achieved by **`Paper Formula (Mean-Pool, Smoothed)`** at **1.57%** (95% CI [0.57%, 2.64%]).
   - The lowest baseline accuracy was observed on **`RoBERTa Alone (Raw)`** at **88.70%**.

2. **Statistical Significance of Ensembling:**
   - Comparing `Top-K Pool (Smoothed)` against `RoBERTa Alone (Smoothed)` reveals a statistically significant accuracy gain of **+0.94%** (95% CI [+0.43%, +1.43%], $p = 0.0000$).
   - Comparing `Top-K Pool (Smoothed)` against `ModernBERT Alone (Smoothed)` demonstrates an accuracy gain of **+4.58%** (95% CI [+2.22%, +7.93%], $p = 0.0000$).

3. **The Smoothing Dilemma in Mixed-Authorship Documents:**
   - On homogeneous test chunks, smoothing yields an apparent accuracy increase of **+1.39%** ($p = 0.0000$).
   - However, in synthetic mixed documents with $k=25\%$ inserted AI text, moving from raw ($W=1$) to 3-sentence rolling average ($W=3$) increases the false alarm rate on untouched human sentences from **23.19%** to **24.88%** due to spatial boundary smearing.

4. **Vulnerability of Transition Heuristics on Formal Writing:**
   - Under the production ensemble containing Rule 1 (`HUMANIZER_TRANSITIONS_REGEX`), human sentences featuring formal transition markers suffer a false positive rate of **10.62%**, compared to only **5.44%** on transition-free human sentences (a disparity of **+5.18%**).
   - Turning Rule 1 OFF completely closes this gap, reducing transition-heavy human FPR to **3.98%**.

5. **Probability Calibration Gains:**
   - Isotonic regression yielded the largest calibration error reduction across all systems, reducing ECE from `0.1307` down to `0.0088` (an average relative calibration error drop of **90.1%**).

---

## 9. Threats to Validity

1. **In-Distribution Topic Test Bias:**
   While `test.parquet` is strictly split by `topic_id` and models never saw test topics, stylistic cues shared across arXiv repositories or prompt templates may inflate recall relative to truly novel domains.
2. **Sentence-Level Label Homogeneity:**
   Standard dataset splits group all sentences within an article under identical labels. As demonstrated in Section 5, rolling average smoothing performance drops when applied to realistic mixed-authorship text.
3. **Sample Imbalance across Commercial and Open LLMs:**
   While Claude (1,279 samples) and GPT-5 (1,154 samples) have dense test coverage, smaller cohorts like Llama (200 samples) have wider variance.
4. **Hardware and Latency Environment:**
   Latency metrics were measured on Intel 12-core CPU under batch size 1. On RTX 2050 CUDA fp16 hardware, throughput scales up significantly.
