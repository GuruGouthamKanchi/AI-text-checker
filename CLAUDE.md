# CLAUDE.md: Complete Codebase Guide & System Index

This file is the official entry point and comprehensive system index for Claude (Claude Code, Claude Projects, and Anthropic API models) to navigate and understand the entire `AI-text-checker` repository.

---

## 1. Repository Purpose & Architecture Overview

**Core Mission:** Open-source detection and forensic analysis of AI-generated academic manuscripts and student essays, combining sentence-level syntax classification with surrounding-window context modeling.

### Dual-Scale Ensemble Architecture:
- **Micro-Syntax Engine (RoBERTa):** Evaluates local sentence syntax and token distributions up to 512 tokens using fine-tuned `roberta-sentence-academic-v2`. **`[IMPLEMENTED]`**
- **Surrounding-Window Context Engine (ModernBERT):** Evaluates surrounding 3-sentence paragraph windows ($s_{i-1}, s_i, s_{i+1}$) truncated to 512 tokens using fine-tuned `modernbert-academic` (`src/highlighter.py:170`). **`[IMPLEMENTED]`**
- **Full 8,192-Token Document Processing:** Single-pass evaluation of full multi-page documents up to 8,192 tokens natively. **`[NOT IMPLEMENTED — inference currently truncates to 512 tokens on a 3-sentence window]`**
- **Adaptive Pooling & Gating:** Production inference applies gated max-pooling (`max(prob_rob, prob_mb)`) with heuristic transition regex indicators and document-level burstiness bonuses (`src/highlighter.py:187-197`). Empirical metrics for Raw, Smoothed, Top-$k$, and Mean-Pool configurations are documented programmatically in [results/EVAL_SUMMARY.md](results/EVAL_SUMMARY.md) and [results/ensemble_metrics.csv](results/ensemble_metrics.csv). **`[IMPLEMENTED]`**
- **Reference Guardrail:** Automated bibliographic reference segmentation and CrossRef API DOI lookup (`src/doi_verifier.py`). **`[IMPLEMENTED; empirical hallucination benchmark NOT IMPLEMENTED]`**

---

## 2. Complete Directory Map & File Inventory

```text
AI_Text_Checker/
├── CLAUDE.md                      <- Master guide and index for Claude
├── README.md                      <- Main project presentation and benchmark summary
├── requirements.txt               <- Python dependencies (PyTorch, Transformers, FastAPI)
├── install.bat / install.sh       <- Automated environment setup scripts
├── start_all.bat / start_all.sh   <- Single-click launchers for backend & frontend
├── stop_all.bat / stop_all.sh     <- Single-click process cleanup
├── ieee_paper.tex                 <- Working IEEE conference manuscript draft (under revision)
│
├── src/                           <- Core Backend & Inference Engine
│   ├── app.py                     # FastAPI REST API with WebSocket streaming & report exports
│   ├── highlighter.py             # Dual-model inference, heuristic rules, and token pooling
│   ├── document_parser.py         # Multi-format document parser (PDF, DOCX, DOC)
│   ├── citation_parser.py         # Bibliography segmenter and citation extractor
│   ├── doi_verifier.py            # Automated CrossRef DOI verification for hallucinations
│   ├── report_generator.py        # PDF report generator with forensic highlighting
│   ├── clean_text.py              # Text normalization, LaTeX math masking, sentence splitting
│   ├── dedupe.py                  # Exact duplicate and MinHash LSH fuzzy deduplication
│   ├── split.py                   # Leak-free topic-aware dataset partitioning
│   ├── prepare_dataset.py         # End-to-end dataset preprocessing orchestrator
│   ├── closed_loop_engine.py      # Adversarial evasion monitoring and closed-loop routing
│   ├── tex_parser.py              # LaTeX source parser preserving math blocks
│   └── tex_compiler.py            # LaTeX to PDF compilation utility
│
├── frontend/                      <- Next.js 14 Interactive Web Dashboard
│   ├── package.json               # Node.js dependencies
│   ├── src/app/                   # Next.js App Router (pages and layouts)
│   └── src/components/
│       ├── PretextCanvasView.tsx  # Interactive canvas with sentence-level risk highlighting
│       ├── AnalysisDashboard.tsx  # Granular stylometric charts and risk breakdown
│       └── CitationTable.tsx      # DOI verification table with clickable metadata
│
├── eval/                          <- Comprehensive Empirical Evaluation Suite
│   ├── run_ensemble_eval.py       # Primary detector evaluation runner on test.parquet
│   ├── run_eval_extensions.py     # 1,000-resample bootstrap CIs, paired tests, calibration
│   ├── run_ensemble_eval.ipynb    # Google Colab notebook for detector evaluation
│   ├── run_robustness_colab.ipynb # Google Colab notebook for adversarial retraining
│   ├── run_ood_retraining_colab.py# Colab Leave-One-Out OOD retraining script
│   ├── run_ood_retraining.ipynb   # Colab notebook for Leave-One-Out OOD experiments
│   ├── MIXED_DOCS.md              # Synthetic mixed-authorship benchmark methodology
│   ├── non_native/                # Loader and README for verified non-native writing samples
│   └── long_docs/                 # arXiv paper downloader and 8k-token prompt templates
│
├── robustness_eval/               <- Adversarial Attack & Closed-Loop Retraining Suite
│   ├── attack_generator.py        # 4 attack families (T5, LLM light/heavy, Rule-based, Commercial)
│   ├── run_robustness_study.py    # Retraining loop with contrastive invariance sweep
│   ├── README_FINDINGS.md         # Robustness audit and theoretical formulation
│   └── manual_inputs/             # Ingestion scripts for manual QuillBot / Undetectable.ai pastes
│
├── results/                       <- Verified Empirical Benchmarks, Figures, & Reports
│   ├── EVAL_SUMMARY.md            # Dynamically generated detector evaluation report
│   ├── ROBUSTNESS_SUMMARY.md      # Adversarial evaluation and retraining report
│   ├── HEURISTICS.md              # Mathematical audit of hand-written inference rules
│   ├── bootstrap_all_systems.csv  # 1,000-resample 95% bootstrap CIs for all 12 systems
│   ├── paired_tests.csv           # Paired differences in accuracy, FPR, TPR@1% with p-values
│   ├── calibration_comparison.csv # Uncalibrated vs Temperature Scaled vs Isotonic Regression
│   ├── mixed_authorship.csv       # Mixed document evaluation across window sizes 1, 3, 5
│   ├── ood_generator.csv          # Zero-shot detection recall across 11 LLM generators
│   ├── ood_domain.csv             # Performance across Academic, Essay, and Q&A domains
│   ├── heuristic_ablation.csv     # Heuristic rule ablations and transition-heavy human FPR
│   ├── latency_detailed.json      # Single-sentence and 512-token chunk latency benchmarks
│   └── figures/                   # 300-DPI publication figures (PDF & PNG)
│
├── data/                          <- Dataset Directory (Gitignored)
│   ├── raw/                       # Raw downloads (DAIGT, HC3)
│   └── processed/                 # Leak-free parquet splits (train.parquet, val.parquet, test.parquet)
│
└── models/                        <- Fine-Tuned Model Weights (Gitignored)
    ├── roberta-sentence-academic-v2/ # 125M parameter sentence detector
    ├── modernbert-academic/       # 149M parameter 8k context detector
    └── t5-humanizer-v1/           # 220M parameter neural paraphraser
```

---

## 3. Key Components & Code Contracts

### 1. Backend Inference (`src/highlighter.py`)
- **Primary Function:** `run_predictions(sentences: list[str]) -> tuple[list[dict], bool]`
- **Inputs:** A segmented list of sentences.
- **Outputs:** A list of dictionaries containing `{"sentence": str, "score": float, "label": int}` and a simulation boolean flag.
- **Rules & Heuristics:**
  - `HUMANIZER_TRANSITIONS_REGEX`: Matches 54 formal transition markers.
  - `is_low_burstiness`: Document-level sentence length variance check ($\sigma < 4.0$).
  - Adaptive Non-Linear Pooling: Combines ModernBERT and RoBERTa via gated max-pooling.

### 2. Document Parsing (`src/document_parser.py`)
- Handles `.pdf`, `.docx`, and legacy `.doc` (via background LibreOffice conversion). Raw text inputs are ingested directly via the web UI and REST API.
- Masks LaTeX math formulas (`$...$`, `$$...$$`, `\[...\]`) to prevent sentence split distortion.
- Extracts body text separately from the `References` / `Bibliography` section.

### 3. Citation Guardrail (`src/citation_parser.py` & `src/doi_verifier.py`)
- Regex parses IEEE, APA, MLA citation formats.
- Queries the official CrossRef REST API (`https://api.crossref.org/works/`) to verify whether DOIs and titles exist in real-world academic literature.

### 4. Evaluation Suite (`eval/run_eval_extensions.py`)
- Evaluates 12 system configurations on `test.parquet` (12,004 rows).
- 1,000 topic-level bootstrap resamples.
- Calibrates probabilities using Temperature Scaling and Isotonic Regression.
- Evaluates synthetic mixed documents ($k \in \{10, 25, 50\}\%$ AI needles).

### 5. Adversarial Robustness (`robustness_eval/run_robustness_study.py`)
- Evaluates 4 attack families: T5 Paraphrase, LLM Light Rewrite, LLM Heavy Rewrite (Held-out), and Rule-based.
- Retraining loop across 3 random seeds (42, 43, 44) sweeping contrastive invariance weight $\lambda \in \{0.0, 0.1, 0.2, 0.5\}$.

---

## 4. Execution Cheat Sheet

| Task | Command |
|---|---|
| **Start Backend API** | `uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload` |
| **Start Frontend Dashboard** | `cd frontend && npm run dev` |
| **Run Detector Evaluation** | `python eval/run_eval_extensions.py` |
| **Run Adversarial Retraining** | `python robustness_eval/run_robustness_study.py` |
| **Run Preprocessing Pipeline** | `python src/prepare_dataset.py` |

---

## 5. How to Read Entire Repository via GitIngest
Any Claude model or user can read this entire repository in one prompt by replacing `github.com` with `gitingest.com`:
👉 **`https://gitingest.com/GuruGouthamKanchi/AI-text-checker`**
This produces a complete, token-optimized textual snapshot containing all directories, files, and code.

---

## 6. Direct Monolithic Markdown Digest Link
To load the full repository codebase directly in token-optimized Markdown format:
👉 **`https://raw.githubusercontent.com/GuruGouthamKanchi/AI-text-checker/main/CLAUDE_DIGEST.md`**
This file compiles all 185 source files into fenced Markdown code blocks, minimizing token consumption for Claude's BPE tokenizer.
