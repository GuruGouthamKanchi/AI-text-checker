# 🛡️ AI-Generated Text Detector: Dual-Scale Hybrid Ensemble & Adversarial Robustness Suite

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0+-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source AI text detection and forensic analysis system designed for academic manuscripts and student writing. The platform pairs **micro-syntactic sentence analysis** (fine-tuned RoBERTa) with **surrounding-window context modeling** (fine-tuned ModernBERT, 3-sentence sliding window truncated to 512 tokens), supported by an interactive Next.js dashboard, citation verification, and closed-loop adversarial retraining.

---

## 🌟 Key Architecture & Implementation Status

```mermaid
graph TD
    A[Raw Document: PDF / DOCX / DOC / Text] --> B[Document Parser & Math Masker]
    B --> C[Sentence Segmentation]
    C --> D1[Micro-Syntax Engine: RoBERTa-512]
    C --> D2[Surrounding Context Engine: ModernBERT-512 Window]
    D1 --> E[Fusion: Heuristic Gated Max-Pooling]
    D2 --> E
    E --> F[Post-Hoc Calibration: Temperature Scaling & Isotonic]
    F --> G[Next.js Dashboard: Pretext Canvas View]
    C --> H[Citation Parser & CrossRef DOI Verifier]
    H --> G
```

### Architecture Inventory:
1. **Micro-Syntax Sentence Analysis:** Custom fine-tuned `roberta-sentence-academic-v2` evaluates local sentence syntax and token distributions within each sentence up to 512 tokens (`src/highlighter.py:176-180`). **`[IMPLEMENTED]`**
2. **Surrounding-Window Context Modeling:** Fine-tuned `modernbert-academic` evaluates surrounding 3-sentence paragraph windows ($s_{i-1}, s_i, s_{i+1}$) truncated to 512 tokens (`src/highlighter.py:170`). **`[IMPLEMENTED]`**
3. **Full 8,192-Token Un-truncated Document Context:** Single-pass evaluation of full multi-page manuscripts up to 8,192 tokens natively. **`[NOT IMPLEMENTED — inference currently truncates to 512 tokens on a 3-sentence sliding window]`**
4. **Adaptive Pooling & Gating:** Production inference combines models via gated max-pooling (`max(prob_rob, prob_mb)`) supplemented by hand-written regex transition indicators and low-burstiness document bonuses (`src/highlighter.py:187-197`). Benchmark studies also evaluate soft-voting Top-$k$ and Mean-pooling formulas (`results/ensemble_metrics.csv`). **`[IMPLEMENTED]`**
5. **LaTeX Math & Reference Masking:** Automatically masks inline (`$...$`) and display math blocks (`$$...$$`, `\[...\]`) prior to sentence segmentation and extracts bibliography blocks (`src/highlighter.py:24-32`). **`[IMPLEMENTED]`**
6. **Bibliographic CrossRef DOI Verification:** Automated reference segmentation and asynchronous CrossRef REST API querying to verify citation metadata (`src/doi_verifier.py`). **`[IMPLEMENTED]`**
7. **Empirical Citation Hallucination Benchmark:** Formal statistical evaluation of citation hallucination detection precision and recall on a labeled benchmark dataset. **`[NOT IMPLEMENTED — feature exists in code, but no benchmark evaluation file exists in results/]`**
8. **Multi-Format Document Parsing:** Ingestion of academic **PDF (`.pdf`)**, modern **Word (`.docx`)**, and legacy **Word (`.doc`)** via headless LibreOffice conversion (`src/document_parser.py`). Direct text input is supported via the web interface and REST API. **`[IMPLEMENTED]`**
9. **Rigorous Statistical Validation & Calibration:** 1,000-resample topic-level bootstrap confidence intervals, paired significance testing ($p < 0.001$), and post-hoc probability calibration reducing Expected Calibration Error (ECE) by an average of 90.1% (`results/calibration_comparison.csv`). **`[IMPLEMENTED]`**
10. **Adversarial Retraining Workbench:** Retraining loop regularizing representation invariance against open-source paraphrasers (T5) and prompt rewrites, while logging human False Positive Rate trade-offs (`robustness_eval/run_robustness_study.py`). **`[IMPLEMENTED]`**

---

## 📊 Benchmark Results

All metrics are evaluated exclusively on `data/processed/test.parquet` (12,004 topic-split sentences never seen during training). All decision thresholds are tuned strictly on `data/processed/val.parquet`.

<!-- BEGIN:RESULTS -->
### 1. Model Performance Comparison (`test.parquet`, n=12,004 sentences)

| System Configuration | Val Thresh | Accuracy | Macro-F1 | ROC-AUC | Human FPR | TPR @ 1% FPR | Calibrated ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa Alone (Raw)** | `0.61` | 88.70% | 88.69% | 95.85% | 11.83% | 61.51% | 0.0135 |
| **RoBERTa Alone (Smoothed, W=3)** | `0.61` | 97.13% | 97.12% | 99.59% | 2.69% | 92.98% | 0.0082 |
| **ModernBERT Alone (Raw, 512 max)** | `0.17` | 91.74% | 91.72% | 95.03% | 4.85% | 76.61% | 0.0182 |
| **ModernBERT Alone (Smoothed, W=3)** | `0.34` | 93.48% | 93.46% | 97.35% | 3.33% | 84.64% | 0.0138 |
| **Production Ensemble (Raw)** | `0.90` | 94.34% | 94.33% | 97.96% | 5.63% | 57.25% | 0.0071 |
| **Production Ensemble (Smoothed, W=3)**| `0.67` | 98.03% | 98.02% | 99.78% | 2.64% | 95.89% | 0.0055 |
| **Paper Top-K Pool (Raw)** | `0.78` | 96.68% | 96.67% | 99.50% | 3.33% | 89.35% | 0.0058 |
| **Paper Top-K Pool (Smoothed, W=3)** | `0.73` | **98.08%** | **98.08%** | **99.85%** | **2.97%** | **97.11%** | **0.0052** |
| **Paper Mean-Pool (Smoothed, W=3)** | `0.59` | **98.59%** | **98.59%** | **99.87%** | **1.56%** | **97.71%** | **0.0046** |

### 2. Paired Bootstrap Significance Tests (1,000 Topic Resamples)

| Comparison | $\Delta$ Accuracy (95% CI) | p-value | $\Delta$ TPR @ 1% FPR (95% CI) | p-value |
|---|:---:|:---:|:---:|:---:|
| **Top-K Ensemble vs RoBERTa Alone** | **+0.94%** [+0.43%, +1.43%] | $p < 0.001$ | **+4.48%** [+2.66%, +7.26%] | $p < 0.001$ |
| **Top-K Ensemble vs ModernBERT Alone**| **+4.58%** [+2.22%, +7.93%] | $p < 0.001$ | **+12.37%** [+5.09%, +22.04%] | $p < 0.001$ |
| **Mean-Pool vs Max-Pool** | **+2.03%** [+1.37%, +2.73%] | $p < 0.001$ | **+7.25%** [+5.26%, +8.90%] | $p < 0.001$ |
| **RoBERTa: Smoothed vs Raw** | **+8.43%** [+7.67%, +9.18%] | $p < 0.001$ | **+30.96%** [+27.17%, +34.41%]| $p < 0.001$ |

### 3. Per-Generator Breakdown (In-Distribution Test Split)

| Generator Architecture | Sample Size ($n$) | Detection Recall (%) | Mean AI Confidence |
|---|:---:|:---:|:---:|
| **ChatGPT** | 500 | **100.00%** | 0.992 |
| **Claude** | 1279 | **99.61%** | 0.982 |
| **Command *(anecdotal, n < 30)*** | 27 | **100.00%** | 0.985 |
| **Falcon** | 56 | **100.00%** | 0.989 |
| **GPT-5** | 1154 | **97.14%** | 0.939 |
| **GPT-OSS-120B** | 1062 | **99.72%** | 0.979 |
| **Gemini** | 1083 | **99.72%** | 0.982 |
| **Llama** | 200 | **100.00%** | 0.988 |
| **Mistral** | 89 | **98.88%** | 0.986 |
| **Other_AI *(mixed/unspecified open LLMs)*** | 186 | **72.58%** | 0.821 |
| **PaLM** | 264 | **99.62%** | 0.990 |

*Note: Per Rule 8, sample sizes below 30 are designated anecdotal. Detection recall on the mixed/unspecified open LLM category (`Other_AI`) drops to 72.58%.*

### 4. Adversarial Attack Evasion & Retraining Recovery

| Evaluation Setting | Attack Family | Sample Size ($n$) | Baseline Recall / FPR | Contrastive (λ=0.2) Recall / FPR | Net Shift |
|---|---|:---:|:---:|:---:|:---:|
| **Clean Human Writing** | `clean_human` | 600 | 12.00% | 32.67% ± 3.30% | +20.67% FPR inflation |
| **Clean AI Writing** | `clean_ai` | 600 | 92.50% | 99.00% ± 0.00% | +6.50% recall |
| Rule-based Perturbations | `rule_based` | 600 | 87.50% | 96.17% ± 0.47% | +8.67% recall |
| T5 Neural Paraphraser | `t5_paraphrase` | 600 | 83.00% | 93.83% ± 1.25% | +10.83% recall |
| LLM Conversational Light Rewrite | `llm_prompt_light` | 600 | 84.00% | 95.17% ± 0.24% | +11.17% recall |
| **LLM Heavy Rewrite (Held-Out)** | `llm_prompt_heavy` | 600 | 84.50% | 95.33% ± 0.85% | **+10.83% recall** |
| QuillBot *(anecdotal, n=2)* | `commercial_quillbot` | 2 | 0.00% | 83.33% ± 23.57% | +83.33% recall |
| Undetectable.ai *(anecdotal, n=2)* | `commercial_undetectable.ai` | 2 | 0.00% | 0.00% | +0.00% recall |

*Important Trade-off: While closed-loop adversarial retraining with contrastive loss regularizes representation invariance and recovers +10.83% recall on held-out heavy rewrites, it simultaneously inflates the False Positive Rate on clean human text from 12.00% to 32.67% (± 3.30%). Commercial paraphraser evaluations (QuillBot, Undetectable.ai) are strictly anecdotal ($n=2$). Undetectable.ai achieved 0.0% detection recall across all rounds.*

### 5. Measured Hardware Inference Latency

- **Hardware Environment:** Windows 10, RTX 2050 4GB (Driver 592.82, CUDA 13.1) (PyTorch Device: `CPU`)
- **Single-Sentence Inference:** 36.59 ms ± 2.25 ms (27.3 sentences/sec)
- **512-Token Chunk Inference:** 347.76 ms ± 6.49 ms (2.88 chunks/sec)
<!-- END:RESULTS -->

## 📁 Repository Structure

```text
AI_Text_Checker/
├── data/
│   ├── raw/                       # Raw benchmark datasets (DAIGT, HC3, etc.)
│   └── processed/                 # Leak-free parquet splits (train, val, test)
├── models/
│   ├── roberta-sentence-academic-v2/ # Fine-tuned RoBERTa micro-sentence detector
│   ├── modernbert-academic/       # Fine-tuned ModernBERT 8k macro-context detector
│   └── t5-humanizer-v1/           # Local neural paraphraser for adversarial evaluation
├── src/
│   ├── app.py                     # FastAPI production server with WebSocket streaming
│   ├── highlighter.py             # Dual-model inference, heuristics, and token pooling
│   ├── document_parser.py         # PDF / DOCX / DOC / TXT ingestion with LaTeX masking
│   ├── citation_parser.py         # Reference segmenter and bibliography parser
│   ├── doi_verifier.py            # Automated CrossRef API DOI & citation hallucination check
│   ├── report_generator.py        # PDF export with forensic sentence highlighting
│   └── prepare_dataset.py         # Leak-free topic-aware dataset preprocessing pipeline
├── frontend/                      # Next.js 14 + TypeScript + Tailwind CSS web dashboard
│   ├── src/components/
│   │   ├── PretextCanvasView.tsx  # Document canvas with interactive sentence highlighting
│   │   ├── AnalysisDashboard.tsx  # Granular risk breakdown and stylometric charts
│   │   └── CitationTable.tsx      # DOI verification and hallucination inspection
│   └── package.json
├── eval/                          # Complete empirical evaluation suite
│   ├── run_ensemble_eval.py       # Full evaluation runner on test.parquet
│   ├── run_eval_extensions.py     # Bootstrap, paired tests, calibration, mixed-authorship
│   ├── run_ensemble_eval.ipynb    # Google Colab notebook for detector evaluation
│   ├── run_robustness_colab.ipynb # Google Colab notebook for adversarial retraining
│   ├── run_ood_retraining.ipynb   # Colab notebook for Leave-One-Out OOD experiments
│   ├── MIXED_DOCS.md              # Synthetic mixed-authorship methodology documentation
│   ├── non_native/                # Loader for verified human non-native writing samples
│   └── long_docs/                 # arXiv downloader & generation templates for 8k docs
├── robustness_eval/               # Adversarial attack & closed-loop retraining pipeline
│   ├── attack_generator.py        # 4 attack families with strict split isolation
│   ├── run_robustness_study.py    # Retraining loop with contrastive invariance sweep
│   ├── README_FINDINGS.md         # Robustness audit and theoretical formulation
│   └── manual_inputs/             # Ingestion scripts for manual QuillBot / Undetectable.ai pastes
├── results/                       # Empirical artifacts, CSVs, JSONs, and publication figures
│   ├── EVAL_SUMMARY.md            # Dynamically generated detector evaluation report
│   ├── ROBUSTNESS_SUMMARY.md      # Adversarial evaluation and retraining report
│   ├── HEURISTICS.md              # Mathematical audit of hand-written inference rules
│   └── figures/                   # 300-DPI publication figures (PDF & PNG)
├── install.bat / install.sh       # One-click environment installer
├── start_all.bat / start_all.sh   # One-click startup for backend & frontend
└── ieee_paper.tex                 # Publication-ready IEEE conference manuscript
```

---

## 🚀 Quickstart & Installation

### Option A: One-Click Launchers (Recommended)

#### Windows
1. Run **`install.bat`** (first-time setup: creates `.venv`, installs dependencies, and prepares models).
2. Run **`start_all.bat`** to start both the FastAPI backend (`:8000`) and the Next.js frontend (`:3000`).
3. To stop all services: run **`stop_all.bat`**.

#### Linux / macOS
```bash
chmod +x install.sh start_all.sh stop_all.sh
./install.sh
./start_all.sh
```

---

### Option B: Manual Setup

#### 1. Backend Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# (Optional) For NVIDIA GPU acceleration (RTX 2050 / 3060 / 4090):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Start the Backend API
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- API & Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

#### 3. Start the Next.js Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
- Frontend Dashboard: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Reproducing Experimental Evaluations

All experiments strictly follow reproducible research standards: zero invented or rounded metrics, fixed random seeds (42, 43, 44), topic-split data isolation, and persistent artifact logging.

### 1. Detector Evaluation & Statistical Validation
```bash
# Executes 1,000 topic-level bootstrap resamples, paired significance tests,
# probability calibration, mixed-authorship benchmarks, and heuristic ablations:
python eval/run_eval_extensions.py
```
Outputs saved to `results/`:
- `results/bootstrap_all_systems.csv`
- `results/paired_tests.csv`
- `results/calibration_comparison.csv`
- `results/mixed_authorship.csv`
- `results/heuristic_ablation.csv`
- `results/figures/calibration_curves.png`
- `results/figures/mixed_authorship_tradeoff.png`
- `results/EVAL_SUMMARY.md`

### 2. Adversarial Robustness & Closed-Loop Retraining
```bash
# Evaluates 4 attack families, executes closed-loop retraining across 3 seeds,
# and sweeps contrastive loss lambda over {0, 0.1, 0.2, 0.5}:
python robustness_eval/run_robustness_study.py
```
Outputs saved to `results/`:
- `results/robustness_metrics.csv`
- `results/retrain_rounds.csv`
- `results/figures/robustness_degradation.png`
- `results/figures/retrain_rounds.png`
- `results/figures/lambda_sweep.png`
- `results/ROBUSTNESS_SUMMARY.md`

### 3. Google Colab Ready Notebooks
If local GPU VRAM is constrained, run the interactive Colab notebooks directly:
- **Ensemble Evaluation:** [`eval/run_ensemble_eval.ipynb`](eval/run_ensemble_eval.ipynb)
- **Adversarial Retraining:** [`eval/run_robustness_colab.ipynb`](eval/run_robustness_colab.ipynb)
- **Leave-One-Out OOD Suite:** [`eval/run_ood_retraining.ipynb`](eval/run_ood_retraining.ipynb)

---

## ⚠️ Limitations and Intended Use

This system is an academic research platform and decision-support instrument. In accordance with ethical academic forensics and empirical findings in `results/`, the following operational constraints must be observed:

1. **Not for Sole Disciplinary Evidence:** Probability estimates and highlight flags reflect localized statistical features (token distribution shifts and sentence length variances). They do **not** constitute definitive proof of authorship and must never serve as the sole justification for academic misconduct accusations, punitive penalties, or grade reductions without independent corroborating evidence and author interview.
2. **Domain-Specific False Positive Rates:** While the human False Positive Rate is 1.31% on academic arXiv abstracts (`results/ood_domain.csv`), it rises to **12.03% on student essays** and **10.62% on formal human writing with transition markers** (`results/heuristic_ablation.csv`). Evaluators must exercise caution when assessing non-native or formal student compositions.
3. **In-Distribution Test Split Constraints:** The 12,004 sentences in `test.parquet` are partitioned strictly by `topic_id` (zero topic overlap with training). However, shared corpus characteristics across academic repositories and specific prompt templates may lead to higher apparent recall compared to entirely novel, out-of-domain writing.
4. **Spatial Boundary Washout in Mixed-Authorship Documents:** Applying 3-sentence rolling average smoothing ($W=3$) effectively suppresses isolated false alarm spikes on homogeneous human text, but degrades detection recall on sparse, inserted AI sentences from **93.67% down to 21.43%** in synthetic mixed documents with 10% AI replacement (`results/mixed_authorship.csv`).
5. **Adversarial Retraining Specificity Trade-off:** Adversarial fine-tuning on paraphrased text increases recall on heavy rewrites by +10.83%, but simultaneously increases the clean human False Positive Rate from **12.00% to 32.67%** (`results/robustness_metrics.csv`).

---

## 📄 Research Paper & Manuscript Draft

A working conference manuscript draft is available in [`ieee_paper.tex`](ieee_paper.tex). Note that empirical tables in the manuscript are currently being synchronized with verified artifacts under `results/`.

If you reference this codebase, models, or evaluation datasets in your research, please cite:

```bibtex
@article{aitextchecker2026,
  title={Dual-Scale Hybrid Transformer Ensembles and Invariant Representations for Adversarially Robust AI Text Detection},
  author={Kanchi, Guru Goutham and Research Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 🛡️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
