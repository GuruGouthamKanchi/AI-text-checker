# Master Technical Specification & Project Blueprint: AI-Generated Text Detector

This document serves as the master technical summary, architectural design, and blueprint of the **Academic AI-Generated Text Detector** project. It outlines every decision, parameter, mathematical implementation, and dataset schema configured from initialization to Phase 6.

---

## 1. Project Directory Structure
The repository is organized as follows:
```text
AI_Text_Checker/
│
├── data/
│   ├── raw/
│   │   └── daigt_v2_train.csv             # Kaggle DAIGT V2 dataset (101MB)
│   └── processed/
│       ├── train.parquet                 # Balanced, stratified training split (9.39MB)
│       ├── val.parquet                   # Balanced, stratified validation split (1.19MB)
│       └── test.parquet                  # Balanced, stratified test split (1.15MB)
│
├── models/
│   ├── baseline/                         # Saved TF-IDF vectorizer and Logistic Regression model
│   ├── modernbert-academic/              # Surrounding-context classifier (evaluated on 3-sentence window truncated to 512 tokens)
│   ├── roberta-sentence-academic/        # Original sentence model weights (v1)
│   └── roberta-sentence-academic-v2/     # Optimized domain-stratified sentence weights (v2)
│
├── src/
│   ├── data_loaders.py                   # HF/local dataset ingestion and schema standardization
│   ├── clean_text.py                     # Text normalization and character cleaning
│   ├── dedupe.py                         # Text similarity and deduplication functions
│   ├── split.py                          # Topic-aware split implementation (prevents leakage)
│   ├── baseline_model.py                 # TF-IDF + Logistic Regression training script
│   ├── train_distilbert_colab.py          # DistilBERT Colab training script (Phase 3)
│   ├── train_roberta_colab.py            # Document-level RoBERTa Colab training script (Phase 4)
│   ├── train_sentence_roberta.py         # Sentence-level Multi-LLM Colab training script (Phase 5)
│   └── highlighter.py                    # Inference engine, PDF parser, HTML reporter & widgets
│
├── requirements.txt                      # Project library dependencies
├── EJ1172284.pdf                         # Human-written test PDF journal paper (sample demonstration)
└── LLM_Architectures_and_Training_Survey.pdf  # Claude-generated test PDF survey paper (sample demonstration)
```

---

## 2. Ingested Datasets & Standardized Schema
To align the project with academic journal verification, we ingested four distinct datasets in [`src/data_loaders.py`](file:///d:/AI_Text_Checker/src/data_loaders.py). All loaders map raw columns to a unified schema:
1. `text` (str): Raw text content.
2. `label` (int): `0` = Human-written, `1` = AI-generated.
3. `source` (str): Identifier of the specific writing origin or generator model.
4. `topic_id` (str): Topic, prompt, or arXiv title identifier used for leak-free group splitting.

### Ingestion Source Breakdowns:
* **Academic Journals (arXiv)**:
  * `NicolaiSivesind/ChatGPT-Research-Abstracts`: 20,000 abstract pairs.
  * `coai/ai-text-detection-training`: 73,482 paragraphs. Contains original papers and AI paraphrases from Claude, Gemini, GPT, and open models.
* **Student Essays (DAIGT)**:
  * Kaggle `daigt_v2_train.csv` (101MB): 44,868 rows. Includes essays from the Persuade Corpus and AI generations (ChatGPT, Llama).
* **General Q&A Forums (HC3)**:
  * `Hello-SimpleAI/HC3`: 85,431 answers comparing Human responses with ChatGPT responses.

---

## 3. Exploratory Historical Baselines (Phases 2–4 Archival Notes)

*Note: Early exploratory models (TF-IDF, DistilBERT, RoBERTa document chunking) served as preliminary scaffolding. All official, verified empirical metrics are evaluated exclusively on `data/processed/test.parquet` and documented programmatically in [`results/EVAL_SUMMARY.md`](results/EVAL_SUMMARY.md).*

---

## 4. Sentence-Level Pipeline & Stratified Retraining (Phase 5)

To highlight individual sentences in the software, we transitioned the pipeline to sentence-level inputs.

### Preprocessing & Filtering (`src/prepare_dataset.py`):
1. **Sentence Segmentation**: Paragraphs are segmented into sentences using `nltk.sent_tokenize`.
2. **Punctuation & Length Filters**: Filters out sentences under 15 characters or under 3 words.
3. **Exact Deduplication**: Drops duplicate sentences.
4. **Topic-Aware Splits**: Groups sentence splits by paper title or prompt ID (`topic_id`) using `GroupKFold` equivalents, ensuring sentences from the same paper are not shared across training, validation, and test splits (0.0% data leakage).

### Domain-Stratified Balancing:
To balance academic style representation across writing types, the dataset preparation pipeline samples:
* **Academic Journals**: ~100,000 sentences
* **Student Essays**: ~10,000 sentences
* **Forum Q&As**: ~10,000 sentences
* **Total Processed Corpus**: ~120,000 balanced sentences across `train.parquet` (95k), `val.parquet` (12k), and `test.parquet` (12,004).

### Verified Empirical Performance (v2 Model on `test.parquet`, n=12,004):
Refer to [`results/EVAL_SUMMARY.md`](results/EVAL_SUMMARY.md) for full metrics:
* **RoBERTa Sentence Model (Raw)**: Accuracy: **88.70%** | Macro-F1: **88.69%** | Human FPR: **11.83%**
* **RoBERTa Sentence Model (Smoothed, W=3)**: Accuracy: **97.13%** | Macro-F1: **97.12%** | Human FPR: **2.69%**
* **Domain Disparity**: Human FPR is **1.31%** on academic papers, but rises to **12.03%** on student essays (`results/ood_domain.csv`).

---

## 5. Inference Engine & Visual Report (`src/highlighter.py`)

The local checker parses documents, scores sentences, and generates the verification dashboard.

### 1. Multi-Format Parser Integration:
* Integrates `PyMuPDF` (`fitz`) and `python-docx` for document parsing, with background LibreOffice conversion for legacy `.doc` files. Direct text is ingested via the REST API.

### 2. Post-Processing Context Smoothing & Empirical Trade-offs:
To smooth out isolated false alarms on homogeneous human text, a 3-sentence weighted rolling average is supported:
$$S_{\text{smooth}}(i) = 0.25 \cdot S(i-1) + 0.50 \cdot S(i) + 0.25 \cdot S(i+1)$$

*Critical Mixed-Authorship Caveat:* While smoothing lowers human False Positive Rates on homogeneous writing (from 11.83% to 2.69%), [`results/mixed_authorship.csv`](results/mixed_authorship.csv) demonstrates that in mixed documents with sparse AI insertions, smoothing degrades single-sentence AI needle recall from **93.67% down to 21.43%**.

### 3. Visual Confidence Thresholds:
* **High Confidence AI (Red)**: $\ge 80\%$ probability
* **Medium Confidence AI (Orange)**: $55\% - 80\%$ probability
* **Unflagged (Human Baseline)**: $< 55\%$ probability

*Qualitative Single-Document Demonstrations ($n=1$)*:
* Human PDF (`EJ1172284.pdf`): 356 of 361 sentences unflagged (5 orange highlights, 1.39% single-document FPR).
* AI Survey PDF (`LLM_Architectures_and_Training_Survey.pdf`): 95 of 186 sentences flagged as AI-generated.

### 4. Explainability Heuristic Widgets:
* **Lexical Diversity (MSTTR)**: Type-Token Ratio across sliding 100-word windows. Intended as a heuristic diagnostic rule, not a standalone classifier.
* **Structural Burstiness (CV)**: Coefficient of variation of sentence lengths ($\sigma_L / \mu_L$). Intended as an exploratory stylistic gauge.

---

## 6. Advanced Integrity Safeguards

* **LLM Writing Fingerprint Watchdog**: Analyses the density of known model-specific transition phrases (`src/highlighter.py:analyze_llm_signatures`).
* **Bibliographic Citation Auditor**: Segments academic references from bibliographies and queries the public **CrossRef REST API** asynchronously to verify metadata existence (`src/doi_verifier.py`). Quantitative benchmark evaluation of hallucination detection recall remains an active research extension.
