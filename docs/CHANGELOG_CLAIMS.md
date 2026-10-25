# Claims Changelog (`docs/CHANGELOG_CLAIMS.md`)

This changelog records every textual modification, qualification, and deletion applied in accordance with the **Project Rules** and the resolution plan in `results/claims_ledger.csv`.

---

## 1. Overview of Changes

| Category | Total Resolved | Primary Actions Taken |
|---|:---:|---|
| **Promotional & Marketing Terms** | 6 | Removed unverified superlatives ("enterprise-grade", "state-of-the-art", "publication-ready"). |
| **Architecture Specifications** | 12 | Accurately documented 3-sentence sliding window (truncated to 512 tokens); removed non-existent 8k full-document inference and theoretical gating equations; marked inventory as `[IMPLEMENTED]` or `[NOT IMPLEMENTED]`. |
| **Empirical Performance Numbers** | 15 | Replaced hard-coded/fabricated numbers with values generated programmatically from `results/` (`EVAL_SUMMARY.md`, `paired_tests.csv`, `ood_generator.csv`). |
| **Robustness & Adversarial Claims** | 8 | Reported clean-human FPR inflation (+20.67%) in the same sentence as recall recovery (+10.83%); explicitly labeled commercial tools ($n=2$) as strictly anecdotal. |
| **Hardware & Latency Claims** | 3 | Removed unverified NVIDIA RTX 4090 latency claim; reported measured CPU latency on Windows 10 Intel i5-12450H CPU (347.8 ms per 512 tokens, 36.6 ms per sentence). |
| **Dataset & Domain Provenance** | 5 | Replaced fabricated corpus claims (PubMed, IEEE Xplore, Law, Philosophy) with actual test split composition (12,004 topic-split sentences across Academic, Essay, and QA domains). |
| **Frontend & Simulation Guardrails** | 8 | Removed fake DeBERTa multipliers (`* 0.98`) and hardcoded consensus strings; added strict production mode loud failure checks. |

---

## 2. Detailed Claim Resolutions

### CLM-001 (`README.md:10`)
- **Old Text:** `"An enterprise-grade, academically rigorous AI text detection and forensic analysis system."`
- **New Text:** `"An academically rigorous, open-source AI text detection and forensic analysis framework."`
- **Reason:** Project Rule 2. "Enterprise-grade" is an unquantified promotional marketing descriptor with no objective benchmark.
- **Ledger ID:** `CLM-001`

### CLM-002 (`README.md:10`)
- **Old Text:** `"pairs micro-syntactic sentence analysis (RoBERTa 512) with macro-contextual document representation (ModernBERT 8k)"`
- **New Text:** `"pairs micro-syntactic sentence analysis (RoBERTa 512) with local 3-sentence context window evaluation (ModernBERT truncated to 512 tokens)"`
- **Reason:** Project Rule 5. Code inspection of `src/highlighter.py:170` confirms ModernBERT input is truncated to 512 tokens on a 3-sentence sliding window; full 8k document context is not evaluated.
- **Ledger ID:** `CLM-002`

### CLM-003 (`README.md:10`)
- **Old Text:** `"backed by an interactive Next.js dashboard, citation verification, and closed-loop adversarial retraining"`
- **New Text:** `"backed by an interactive Next.js dashboard, automated CrossRef DOI lookup, and closed-loop adversarial retraining"`
- **Reason:** Project Rule 2. Citation verification is implemented via CrossRef API lookup, but no benchmark evaluation of hallucination detection recall exists in `results/`.
- **Ledger ID:** `CLM-003`

### CLM-005 (`README.md:32`)
- **Old Text:** `"Fine-tuned modernbert-academic processes up to 8,192 tokens natively, capturing discourse cohesion, paragraph flow, and macro-level stylistic uniformity."`
- **New Text:** `"Fine-tuned modernbert-academic processes a 3-sentence sliding window (truncated to 512 tokens in production inference), capturing inter-sentence transition context."`
- **Reason:** Project Rule 5. Truncation to 512 tokens in `src/highlighter.py:170` contradicts native 8k document processing claim.
- **Ledger ID:** `CLM-005`

### CLM-006 (`README.md:33`)
- **Old Text:** `"Adaptive Non-Linear Gating: Combines context-window sliding averages with top-k pooling, achieving 98.72% Accuracy and 99.87% ROC-AUC on strictly topic-held-out test benchmarks."`
- **New Text:** `"Ensemble Decision Pooling: Combines sentence-level predictions via Top-K ($K=3$, achieving 98.08% Accuracy, 99.85% ROC-AUC) and Mean-Pooling (achieving 98.59% Accuracy, 99.87% ROC-AUC) on the held-out test split at validation-tuned thresholds. Production heuristic gating in src/highlighter.py achieves 94.34% raw sentence accuracy."`
- **Reason:** Project Rules 1 & 4. 98.72% corresponded to threshold 0.50 in `ensemble_metrics.csv`; validation-tuned threshold 0.73 yields 98.08% in `results/EVAL_SUMMARY.md`.
- **Ledger ID:** `CLM-006`

### CLM-009 (`README.md:40`)
- **Old Text:** `"Direct support for PDF (.pdf), modern Word (.docx), plain text, and legacy Word (.doc) via automated background LibreOffice conversion."`
- **New Text:** `"Direct support for PDF (.pdf), modern Word (.docx), and legacy Word (.doc) via automated background LibreOffice conversion."`
- **Reason:** Project Rule 5. `src/document_parser.py:59` explicitly raises `DocumentParsingError` on plain text (`.txt`), which is tested as unsupported in `tests/test_extraction.py`.
- **Ledger ID:** `CLM-009`

### CLM-013 (`README.md:48`)
- **Old Text:** `"Evaluated against 4 reproducible attack families: open-source neural paraphrasers (T5), prompt rewrites (light & heavy conversational), rule-based perturbations, and commercial tools (QuillBot, Undetectable.ai)."`
- **New Text:** `"Evaluated against 4 reproducible attack families: open-source neural paraphrasers (T5, n=120), prompt rewrites (light & heavy conversational, n=120 each), rule-based perturbations (n=120), and commercial tools (QuillBot, Undetectable.ai; n=2 each, strictly anecdotal)."`
- **Reason:** Project Rule 8. Commercial tool sample size ($n=4$ total, $n=2$ per tool) is strictly anecdotal ($n < 30$).
- **Ledger ID:** `CLM-013`

### CLM-014 (`README.md:49`)
- **Old Text:** `"Closed-loop retraining with contrastive loss regularizes representation invariance, achieving +10.83% recall recovery on unseen heavy rewrite attacks."`
- **New Text:** `"Closed-loop retraining with contrastive loss regularizes representation invariance, achieving +10.83% recall recovery (from 65.00% to 75.83%; n=120) on unseen heavy rewrite attacks, while clean human text false positive rate simultaneously increases from 12.00% to 32.67% (+20.67%; n=150)."`
- **Reason:** Project Rules 1, 2, and prompt mandate. Clean-human FPR increase must be reported in the same sentence as recall recovery.
- **Ledger ID:** `CLM-014`

### CLM-018 (`README.md:82-90`)
- **Old Text:** Table 3 titled `"Out-of-Distribution Generalization across LLM Generators"`, omitting `Other_AI`.
- **New Text:** Table 3 renamed to `"Per-Generator Breakdown (In-Distribution Test Split)"` regenerated programmatically via `scripts/update_readme_tables.py`, including `Other_AI` ($n=186$, 72.58% recall) and reporting $n$ beside every generator.
- **Reason:** Project Rules 1, 3, 8. Renamed to reflect that test.parquet is an in-distribution held-out split, included all generators without cherry-picking, and added sample sizes.
- **Ledger ID:** `CLM-018`

### CLM-019 (`README.md:240`)
- **Old Text:** `"Publication-ready IEEE conference manuscript (ieee_paper.tex)."`
- **New Text:** `"Working manuscript draft (ieee_paper.tex) with empirical evaluation tables and architectural documentation."`
- **Reason:** Project Rule 2. Manuscript contained placeholder metadata and unverified metrics prior to audit.
- **Ledger ID:** `CLM-019`

### CLM-020 (`CLAUDE.md:9`)
- **Old Text:** `"Enterprise-grade and academically rigorous detection of AI-generated academic text, research papers, and essays."`
- **New Text:** `"Academically rigorous, open-source detection framework for AI-generated academic text and research papers."`
- **Reason:** Project Rule 2. Removed "enterprise-grade" promotional claim.
- **Ledger ID:** `CLM-020`

### CLM-022 (`CLAUDE.md:13`)
- **Old Text:** `"Macro-Context Engine (ModernBERT): Captures long-range document cohesion, discourse transitions, and semantic consistency up to 8,192 tokens (models/modernbert-academic)."`
- **New Text:** `"Context Window Engine (ModernBERT) [IMPLEMENTED - 3-SENTENCE WINDOW]: Evaluates local 3-sentence sliding window context truncated to 512 tokens in production (models/modernbert-academic). [NOT IMPLEMENTED]: Full 8,192 token document-level attention."`
- **Reason:** Project Rule 5. Added explicit `[IMPLEMENTED]` and `[NOT IMPLEMENTED]` status indicators to describe code as it exists today.
- **Ledger ID:** `CLM-022`

### CLM-023 (`CLAUDE.md:14`)
- **Old Text:** `"Adaptive Pooling: Non-linear combination of local 3-sentence sliding window context with Top-K pooling, yielding 98.72% Accuracy and 99.87% ROC-AUC on topic-held-out benchmarks."`
- **New Text:** `"Decision Pooling: Combines sentence and context predictions via max-pooling and heuristic gating. Top-K and Mean-Pooling evaluation metrics are documented in results/EVAL_SUMMARY.md."`
- **Reason:** Project Rule 1 and task instruction. Removed hardcoded performance numbers from prose in `CLAUDE.md`; pointed to `results/EVAL_SUMMARY.md`.
- **Ledger ID:** `CLM-023`

### CLM-025 (`CLAUDE.md:29`)
- **Old Text:** `"Publication-ready IEEE conference manuscript (ieee_paper.tex)"`
- **New Text:** `"Working manuscript draft (ieee_paper.tex)"`
- **Reason:** Project Rule 2. Corrected promotional claim.
- **Ledger ID:** `CLM-025`

### CLM-031 (`CLAUDE.md:120`)
- **Old Text:** `"Multi-format document parser (PDF, DOCX, DOC, TXT) / Handles LaTeX .tex directly"`
- **New Text:** `"Multi-format document parser (PDF, DOCX, DOC) / Handles LaTeX .tex directly"`
- **Reason:** Project Rule 5. Plain text `.txt` raises `DocumentParsingError` in `src/document_parser.py`.
- **Ledger ID:** `CLM-031`

### CLM-032 (`PROJECT_SUMMARY.md:23`)
- **Old Text:** `"PRIMARY PRODUCTION CLASSIFIER (8,192 Context Window, FlashAttention-2)"`
- **New Text:** `"CONTEXT WINDOW CLASSIFIER (3-Sentence Sliding Window, 512 Tokens Truncation)"`
- **Reason:** Project Rule 5. Accurately describes model context window as implemented.
- **Ledger ID:** `CLM-032`

### CLM-037 to CLM-040 (`PROJECT_SUMMARY.md:65-95`)
- **Old Text:** Phase 2 TF-IDF Baseline (97.35% Acc), Phase 3 DistilBERT (97.22% Acc, 5.28% FPR), Persuade Corpus essay FPR (31.4%), Phase 4 Document RoBERTa (98.75% Acc, 2.00% FPR).
- **New Text:** Removed unverified historical baseline claims. Replaced with reference to verified evaluation suite on `data/processed/test.parquet` in `results/EVAL_SUMMARY.md`.
- **Reason:** Project Rules 1 & 2. Metrics could not be reproduced from any evidence file under `results/`.
- **Ledger ID:** `CLM-037`, `CLM-038`, `CLM-039`, `CLM-040`

### CLM-042 to CLM-044 (`PROJECT_SUMMARY.md:105-120`)
- **Old Text:** `"Final Test Metrics (v2 Model): Overall Accuracy: 87.84% | Overall Recall: 91.85% | Human Academic FPR: 14.23%"`
- **New Text:** Updated to verified empirical metrics from `results/EVAL_SUMMARY.md`: RoBERTa alone raw (88.70% Acc, 89.14% Rec, 11.83% FPR) and smoothed (97.13% Acc, 97.20% Rec, 2.69% FPR); Hybrid Ensemble Top-K smoothed (98.08% Acc, 99.23% Rec, 2.97% FPR).
- **Reason:** Project Rule 1. Replaced outdated numbers with verified values from `results/EVAL_SUMMARY.md`.
- **Ledger ID:** `CLM-042`, `CLM-043`, `CLM-044`

### CLM-045 (`PROJECT_SUMMARY.md:125`)
- **Old Text:** `"Post-Processing Filters (Near-Zero FPR)"`
- **New Text:** `"Post-Processing Rule-Based Filters (Heuristic Specificity Adjustments)"`
- **Reason:** Project Rule 2. "Near-Zero FPR" is unsupported by empirical evidence (measured FPR is 1.56% - 2.97% on academic abstracts and 12.03% on essays).
- **Ledger ID:** `CLM-045`

### CLM-046 & CLM-047 (`PROJECT_SUMMARY.md:135-145`)
- **Old Text:** Human PDF (EJ1172284.pdf): 1.39% False Positive Rate; AI PDF (LLM_Architectures_and_Training_Survey.pdf): 51.1% AI Percentage.
- **New Text:** Qualified both single-document PDF demonstrations as exploratory case studies with sample size $n=1$, noting that single-document runs cannot establish general system error rates.
- **Reason:** Project Rule 8. Sample size $n=1$ is strictly anecdotal.
- **Ledger ID:** `CLM-046`, `CLM-047`

### CLM-052 to CLM-055 (`ieee_paper.tex:36-54`)
- **Old Text:** Abstract and Introduction claims: 98.6% macro-F1 score outperforming RoBERTa by 11.8%; baseline static transformer catastrophic failure; 12,000 papers from PubMed/IEEE; 97.8% precision post-paraphrasing.
- **New Text:** Rewritten to report 98.08% (Top-K) and 98.59% (Mean-Pool) accuracy on 12,004 topic-split test sentences; described 3-sentence sliding window (512 tokens); reported +10.83% recall recovery against heavy prompt rewrites alongside +20.67% clean human FPR increase.
- **Reason:** Project Rules 1, 2, 5. Replaced fabricated metrics and nonexistent corpus provenance with verified test data.
- **Ledger ID:** `CLM-052`, `CLM-053`, `CLM-054`, `CLM-055`

### CLM-056 (`ieee_paper.tex:74-92`)
- **Old Text:** Table I comparing cited papers with fabricated Macro F1 and Humanizer Resilience numbers (e.g., Gupta 90.1%, Blake 92.1%, Proposed Work 98.6%, 97.8%).
- **New Text:** Converted Table I to a qualitative taxonomy and architectural feature comparison (Venue, Context Window, Stylometric Features, Adversarial Evaluation, Released Artifacts) with zero fabricated numbers.
- **Reason:** Project Rules 1 & 6. Never invent metrics or attribute numbers to cited papers unless reported by those papers.
- **Ledger ID:** `CLM-056`

### CLM-057 & CLM-058 (`ieee_paper.tex:108-144`)
- **Old Text:** ModernBERT un-truncated 8192 document sequence representation; gating network equation $w_1 P_{MB} + w_2 P_{max} + (1 - w_1 - w_2) g$ where $(1 - w_1 - w_2) = 0$.
- **New Text:** Replaced with accurate mathematical formulation of ModernBERT 3-sentence sliding window (truncated to 512 tokens), sentence max-pooling, heuristic gating, and Top-K / Mean-Pool document aggregation.
- **Reason:** Project Rule 5. Architecture text must reflect code as implemented.
- **Ledger ID:** `CLM-057`, `CLM-058`

### CLM-060 to CLM-066 (`ieee_paper.tex:188-228`)
- **Old Text:**
  - 12,000 papers from PubMed/IEEE Xplore.
  - Table II: Fabricated baseline and ensemble metrics (0.4% FPR).
  - Table III: Fabricated humanizer resilience (Undetectable.ai 97.2%).
  - Contrastive loss ablation claiming +23.2% recovery boost.
  - Latency: 280 ms on NVIDIA RTX 4090 GPU.
  - Held-out Law and Philosophy domain generalization (97.4% F1).
- **New Text:**
  - Evaluated on 12,004 topic-split test sentences from `data/processed/test.parquet`.
  - Table II: Empirical metrics from `results/EVAL_SUMMARY.md` Table 1.
  - Table III: Empirical attack recovery from `results/robustness_metrics.csv` (reporting clean-human FPR increase +20.67% and labeling commercial tools $n=2$ anecdotal).
  - Contrastive loss advantage: measured +0.83% over cross-entropy on heavy rewrites.
  - Latency: measured on Intel i5-12450H CPU (347.8 ms per 512 tokens, 36.6 ms per sentence).
  - Domain generalization: reported Academic abstracts (1.31% FPR), Essays (12.03% FPR), and QA (0.00% FPR) from `results/ood_domain.csv`.
- **Reason:** Project Rules 1, 2, 3, 5, 6, 8, 10. Replaced all fabricated data with strictly verified experimental results.
- **Ledger ID:** `CLM-060`, `CLM-061`, `CLM-062`, `CLM-063`, `CLM-064`, `CLM-065`, `CLM-066`

### CLM-067 to CLM-069 (`frontend/src/app/page.tsx`)
- **Old Text:**
  - `detection_engine: "ModernBERT-base (8k Context Window)"` and badges.
  - `multi_model_consensus` with fake DeBERTa-v3 (`* 0.98`), fake RoBERTa (`* 1.02`), and hardcoded `"98.4% High Agreement"`.
  - `"Executing ModernBERT 8k Context Model..."`.
- **New Text:**
  - Updated detection engine strings to `"Hybrid Ensemble (ModernBERT 3-Sentence Context + RoBERTa Sentence)"`.
  - Replaced fake multi-model matrix with truthful Hybrid Ensemble Architecture metadata (Context Window: 3-Sent / 512 Tokens; Sentence Syntax: 512 Tokens; Decision Fusion: Max-Pool & Heuristics).
  - Updated status message to `"Executing ModernBERT Context Model (512 tokens)..."`.
- **Reason:** Project Rules 2 & 5. Removed simulated models, fake multipliers, and incorrect context window claims.
- **Ledger ID:** `CLM-067`, `CLM-068`, `CLM-069`

### CLM-071 to CLM-073 (`frontend/src/components/StylometricGauges.tsx`)
- **Old Text:**
  - Tooltips stating rigid human/AI thresholds ("Human academic papers typically score 70% – 95%").
  - Header badge `"100% Calibrated"`.
- **New Text:**
  - Qualified tooltip ranges as heuristic rules of thumb ("Heuristic rule-of-thumb: academic writing often exhibits 70% – 95% TTR").
  - Replaced badge with `"Heuristic Gauges"`.
- **Reason:** Project Rule 2. Gauges are heuristic diagnostics, not calibrated probability classifiers.
- **Ledger ID:** `CLM-071`, `CLM-072`, `CLM-073`

### CLM-074 & CLM-075 (`src/highlighter.py`)
- **Old Text:**
  - Docstring claiming `"Applies weighted soft-voting: 0.65 * ModernBERT + 0.35 * RoBERTa"`.
  - Simulation mode running unconditionally when weights are missing.
- **New Text:**
  - Updated docstring to describe adaptive max-pooling with heuristic gating and stylometric adjustments.
  - Added loud `FileNotFoundError` check when `STRICT_PRODUCTION=1` or `REQUIRE_MODELS=1` in accordance with Project Rule 7.
- **Reason:** Project Rules 5 & 7. Docstring now matches implementation; production fails loudly if weights are absent.
- **Ledger ID:** `CLM-074`, `CLM-075`

### CLM-076 & CLM-077 (`src/app.py`)
- **Old Text:**
  - Startup log: `"[FastAPI Startup] HYBRID ENSEMBLE ENGINE ACTIVE on {device}! (65% ModernBERT + 35% RoBERTa)"`.
  - Simulation fallback on startup without strict production check.
- **New Text:**
  - Updated log message: `"[FastAPI Startup] HYBRID ENSEMBLE ENGINE ACTIVE on {device}! (Max-pooling & heuristic fusion)"`.
  - Added loud `RuntimeError` check when `STRICT_PRODUCTION=1` or `REQUIRE_MODELS=1` in accordance with Project Rule 7.
- **Reason:** Project Rules 5 & 7. Accurately describes active fusion logic and enforces loud failure in production mode.
- **Ledger ID:** `CLM-076`, `CLM-077`

### CLM-087 & CLM-088 (`results/ROBUSTNESS_SUMMARY.md`)
- **Old Text:** QuillBot (+83.33%) and Undetectable.ai (0.00%) recall recovery reported without sample size or anecdotal qualifiers.
- **New Text:** Table and discussion explicitly label commercial tools as strictly anecdotal with sample size ($n=2$ per tool; $n=4$ total). Stated clean-human FPR inflation (+20.67%) in the same sentence as recall recovery.
- **Reason:** Project Rules 1, 8, and user prompt mandate.
- **Ledger ID:** `CLM-087`, `CLM-088`
