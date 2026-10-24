# Robustness Evaluation Suite: Architecture & Existing Capabilities

**Date:** 2026-10-24  
**Audit Scope:** `robustness_eval/` directory and components

---

## 1. Executive Overview

The `robustness_eval/` package implements an adversarial testing, text humanization, and closed-loop robustness evaluation framework for the AI Text Checker. It provides tools to generate semantic-preserving adversarial text perturbations, measure detector score degradation (evasion rate), fine-tune models on adversarial examples, and serve humanization APIs via FastAPI.

---

## 2. Directory Structure & Module Breakdown

```text
robustness_eval/
├── __init__.py                  # Package exports for rewriters, harness, humanizer, and config
├── config.py                    # RobustnessConfig dataclass (thresholds, boost targets, seeds)
├── router.py                    # FastAPI APIRouter (/robustness/evaluate, /humanize, /closed-loop-humanize)
├── eval_closed_loop.py          # Benchmark script for AccaHumanize-CL dual-agent closed loop
├── eval_tex_pipeline.py         # LaTeX AST parser benchmark preserving equations, tables, & citations
├── finetune_appendix.py         # Retraining loop (pure PyTorch) on mixed clean + adversarial examples
├── requirements.txt             # Minimal dependencies (fastapi, uvicorn, pydantic)
├── config.py                    # Default hyperparameters and thresholds
├── datasets/
│   ├── __init__.py
│   ├── adversarial_pairs.py     # AdversarialPair dataclass & AdversarialDataset JSONL handler
│   ├── build_dataset.py         # Dataset generator creating adversarial pairs from parquet splits
│   └── adversarial_v1.jsonl     # Sample dataset of 10 adversarial pairs
├── rewriters/
│   ├── __init__.py              # Combined rewrite orchestration pipeline
│   ├── vocabulary_rewriter.py   # Regex-based replacement of overused LLM terms (delve, leverage, etc.)
│   ├── burstiness_rewriter.py   # Active-voice restructuring and sentence length variation
│   └── perplexity_rewriter.py   # Replacement of predictable discourse transitions (furthermore, etc.)
├── humanizer/
│   ├── __init__.py
│   ├── humanizer_engine.py      # Neural T5 Seq2Seq paraphraser (models/t5-humanizer-v1) + rule fallback
│   ├── train_humanizer.py       # Fine-tuning script for T5 paraphraser on Google Colab/local
│   └── train_humanizer_colab.ipynb # Colab notebook for T5 humanizer training
├── eval/
│   ├── __init__.py
│   ├── harness.py               # AdversarialHarness query engine evaluating detector evasion
│   └── report.py                # RobustnessReport metrics generator (ARI, flip rate, score drop)
└── reports/
    ├── robustness_report_v1.json # Pre-computed benchmark report JSON
    └── robustness_report_v1.md   # Pre-computed benchmark summary Markdown
```

---

## 3. Existing Adversarial Transformation Methods

The existing code contains four primary transformation mechanisms:

### A. Vocabulary Rewriting (`rewriters/vocabulary_rewriter.py`)
- **Phrase Map:** 24 multi-word regex substitutions targeting hallmark LLM scholarly phrases:
  - `"delve into"` $\rightarrow$ `"explore"`
  - `"it is important to note that"` $\rightarrow$ `"notably,"`
  - `"serves as a testament to"` $\rightarrow$ `"demonstrates"`
  - `"in the rapidly evolving landscape of"` $\rightarrow$ `"in modern"`
  - `"plays a pivotal role in"` $\rightarrow$ `"is key to"`
- **Single-Word Vocabulary Map:** 19 LLM over-represented terms mapped to varied human synonyms (`"delve"`, `"leverage"`, `"comprehensive"`, `"crucial"`, `"testament"`, `"tapestry"`, `"paramount"`, `"meticulous"`).
- Preserves LaTeX lock tokens (`__TEX_LOCK_...__`).

### B. Burstiness & Active Voice Restructuring (`rewriters/burstiness_rewriter.py`)
- Converts passive participle clause starts into direct active human voice (`"Implemented"` $\rightarrow$ `"I built"`, `"Optimized"` $\rightarrow$ `"I optimized"`, `"Collaborated on"` $\rightarrow$ `"I worked on"`).
- Strips repetitive heading patterns and simplifies percentage reduction phrases.
- Restructures passive academic definitions into natural predictive phrasing.

### C. Perplexity Perturbation (`rewriters/perplexity_rewriter.py`)
- Substitutes predictable transitions with less frequent syntactic variants:
  - `"furthermore"` $\rightarrow$ `"in addition"`
  - `"moreover"` $\rightarrow$ `"alongside this"`
  - `"consequently"` $\rightarrow$ `"resulting in"`
  - `"therefore"` $\rightarrow$ `"thus"`
  - `"however"` $\rightarrow$ `"though"`

### D. Neural Seq2Seq Paraphrasing (`humanizer/humanizer_engine.py`)
- Uses fine-tuned local weights in `models/t5-humanizer-v1` (`AutoModelForSeq2SeqLM`).
- Employs tone-conditioned generation (`academic`, `resume`, `casual`) with beam search (`num_beams=4`), context sliding window, and repetition penalties.
- Token-locking guardrails extract math and citation placeholders before paraphrasing and restore them afterwards.

---

## 4. Evaluation Harness & Reporting Capabilities

- **`AdversarialHarness` (`eval/harness.py`):**
  - Evaluates both `original_text` and `rewritten_text` against detector endpoints (`run_analysis_pipeline` or custom sentence-level models).
  - Computes `score_drop = max(0.0, orig_score - rewritten_score)`.
  - Determines evasion: `evaded = (orig_score >= threshold) and (rewritten_score < threshold)`.
- **`RobustnessReport` (`eval/report.py`):**
  - Generates comprehensive markdown tables and JSON statistics.
  - Computes **Adversarial Resilience Index (ARI)**:
    $$\text{ARI} = 1.0 - \left(0.5 \times \text{FlipRate} + 0.5 \times \frac{\text{MeanScoreDrop}}{100.0}\right)$$
  - Categorizes resilience into tiers: Highly Resilient ($\ge 0.85$), Moderate Resilience ($0.65 - 0.85$), Vulnerable ($0.45 - 0.65$), and Critical Vulnerability ($< 0.45$).

---

## 5. Retraining Pipeline (`finetune_appendix.py`)

- **Current Implementation:** Pure PyTorch training loop on `models/modernbert-academic` using cross-entropy loss over a mix of original AI text, rewritten adversarial text (labeled 1), and human baseline samples (labeled 0).
- **Gaps to Address for Current Task:**
  1. Currently hardcoded with small toy sample lists rather than the real topic-split parquet datasets (`train.parquet`, `val.parquet`, `test.parquet`).
  2. Does not implement the min-max / contrastive variant loss ($\lambda \in \{0, 0.1, 0.2, 0.5\}$).
  3. Does not implement multi-round closed-loop adversarial retraining or held-out family evaluation across multiple random seeds ($42 + 3$ seeds).

---

## 6. Integration With Main Detector

- Exposes FastAPI endpoints in `robustness_eval/router.py`:
  - `POST /robustness/evaluate`: Returns original score, humanized score, delta, resilience verdict, and explanation.
  - `POST /robustness/humanize/stream`: SSE real-time streaming endpoint for sentence-by-sentence neural humanization.
  - `POST /robustness/closed-loop-humanize/stream`: Dual-agent closed loop streaming with token locking.
- Registered into the main FastAPI application in `src/app.py`.
