# Out-of-Distribution Generalization Audit Report

**Date:** 2026-10-25 20:11:58  
**Hardware Environment:** Windows 10, RTX 2050 (4GB), Python 3.12 (fp16 autocast)  
**Evaluation Corpus:** `data/processed/test.parquet` (12,004 sentences)  
**Training Splits:** Retrained from `roberta-base` across 3 seeds ([42, 43, 44])  
**Threshold Selection:** Strictly tuned on `val.parquet` of remaining training groups (Rule 4)  

---

## 1. Generalization Decision Standards & Tolerances

In accordance with Task Specification 4, claims of *'generalization to unseen generators/domains'* are strictly audited against empirical tolerances relative to in-distribution baselines:
- **Recall Retention Tolerance:** $\Delta \text{Recall} \ge -10.0\%$ points relative to in-distribution (or absolute Recall $\ge 85.0\%$).
- **FPR Constraint Tolerance:** $\Delta \text{FPR} \le +5.0\%$ points relative to in-distribution (or absolute FPR $\le 10.0\%$).
- **Operational Decision:** Any group failing either criterion is labeled an **EXPLICIT LIMITATION** in repository documentation and manuscripts.

---

## 2. Leave-One-Out Generalization Results Table

| Target Group | Group Type | $n_{\text{AI}}$ | $n_{\text{Human}}$ | Held-out Rec (95% CI) | In-Dist Rec | $\Delta$ Rec | Held-out FPR (95% CI) | In-Dist FPR | $\Delta$ FPR | Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Claude** | generator | 1,279 | 6,163 | **90.23%** ([88.84%, 91.61%]) | 88.30% | +1.93% | **15.54%** ([13.54%, 17.55%]) | 12.20% | +3.35% | `ALLOWED` |
| **Gemini** | generator | 1,083 | 6,163 | **91.17%** ([90.85%, 91.49%]) | 91.81% | -0.65% | **14.59%** ([11.09%, 18.09%]) | 12.20% | +2.40% | `ALLOWED` |
| **GPT-5** | generator | 1,154 | 6,163 | **39.46%** ([33.91%, 45.01%]) | 50.78% | -11.32% | **10.41%** ([9.38%, 11.43%]) | 12.20% | -1.79% | `LIMITATION` |
| **GPT-OSS-120B** | generator | 1,062 | 6,163 | **74.58%** ([69.31%, 79.84%]) | 85.37% | -10.80% | **13.92%** ([9.05%, 18.78%]) | 12.20% | +1.72% | `LIMITATION` |
| **ChatGPT** | generator | 500 | 6,163 | **18.73%** ([4.91%, 32.56%]) | 74.87% | -56.13% | **12.50%** ([9.95%, 15.06%]) | 12.20% | +0.31% | `LIMITATION` |
| **Llama** | generator | 200 | 6,163 | **76.33%** ([64.77%, 87.89%]) | 73.67% | +2.67% | **17.77%** ([15.55%, 19.98%]) | 12.20% | +5.57% | `LIMITATION` |
| **academic** | domain | 4,466 | 4,749 | **94.15%** ([90.65%, 97.64%]) | 79.21% | +14.94% | **76.10%** ([65.76%, 86.44%]) | 11.70% | +64.40% | `LIMITATION` |
| **essay** | domain | 885 | 1,006 | **65.20%** ([54.14%, 76.25%]) | 80.15% | -14.95% | **9.21%** ([7.02%, 11.41%]) | 17.56% | -8.35% | `LIMITATION` |
| **qa** | domain | 490 | 408 | **28.64%** ([14.60%, 42.68%]) | 74.76% | -46.12% | **5.15%** ([0.48%, 9.81%]) | 4.74% | +0.41% | `LIMITATION` |

---

## 3. Data-Driven Generalization Findings

### A. Generator Generalization Analysis (LOGO):
- **Claude (Generalizes):** When completely excluded from training, the detector retains 90.23% recall ($n=1,279$) against unseen Claude text with clean human FPR at 15.54% ($n=6,163$). The claim of generalizability to unseen Claude models is **EMPIRICALLY SUPPORTED**.
- **Gemini (Generalizes):** When completely excluded from training, the detector retains 91.17% recall ($n=1,083$) against unseen Gemini text with clean human FPR at 14.59% ($n=6,163$). The claim of generalizability to unseen Gemini models is **EMPIRICALLY SUPPORTED**.
- **GPT-5 (Explicit Limitation):** Holding out GPT-5 resulted in recall dropping to 39.46% ($n=1,154$) or human FPR inflating to 10.41% ($n=6,163$), violating tolerance thresholds. Generalization to unseen GPT-5 cannot be claimed and is an **EXPLICIT LIMITATION**.
- **GPT-OSS-120B (Explicit Limitation):** Holding out GPT-OSS-120B resulted in recall dropping to 74.58% ($n=1,062$) or human FPR inflating to 13.92% ($n=6,163$), violating tolerance thresholds. Generalization to unseen GPT-OSS-120B cannot be claimed and is an **EXPLICIT LIMITATION**.
- **ChatGPT (Explicit Limitation):** Holding out ChatGPT resulted in recall dropping to 18.73% ($n=500$) or human FPR inflating to 12.50% ($n=6,163$), violating tolerance thresholds. Generalization to unseen ChatGPT cannot be claimed and is an **EXPLICIT LIMITATION**.
- **Llama (Explicit Limitation):** Holding out Llama resulted in recall dropping to 76.33% ($n=200$) or human FPR inflating to 17.77% ($n=6,163$), violating tolerance thresholds. Generalization to unseen Llama cannot be claimed and is an **EXPLICIT LIMITATION**.

### B. Domain Generalization Analysis (LODO):
- **academic (Explicit Limitation):** Transfer to 'academic' fails tolerance boundaries with 76.10% human FPR ($n=4,749$) and 94.15% AI recall ($n=4,466$). The repository must document that accuracy on 'academic' requires domain-specific supervision.
- **essay (Explicit Limitation):** Transfer to 'essay' fails tolerance boundaries with 9.21% human FPR ($n=1,006$) and 65.20% AI recall ($n=885$). The repository must document that accuracy on 'essay' requires domain-specific supervision.
- **qa (Explicit Limitation):** Transfer to 'qa' fails tolerance boundaries with 5.15% human FPR ($n=408$) and 28.64% AI recall ($n=490$). The repository must document that accuracy on 'qa' requires domain-specific supervision.

---

## 4. Google Colab Replication Instructions

For researchers replicating full unconstrained training on multi-GPU or cloud instances (without sample budget constraints):
```bash
# 1. Clone repository and install dependencies
git clone https://github.com/GuruGouthamKanchi/AI-text-checker.git
cd AI-text-checker
pip install -q transformers datasets accelerate scikit-learn pandas pyarrow

# 2. Run unconstrained leave-one-out retraining suite across all seeds
python eval/run_ood_leave_one_out_colab.py --seeds 42 43 44 --epochs 2 --batch_size 32
```
