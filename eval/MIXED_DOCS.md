# Synthetic Mixed-Authorship Evaluation Protocol (`eval/MIXED_DOCS.md`)

This document details the methodology for constructing and evaluating synthetic mixed-authorship documents to rigorously test whether local smoothing (rolling averages) genuinely improves detection or merely exploits intra-document label homogeneity.

---

## 1. Problem Formulation: The "Smoothing Artifact" Hypothesis

In standard evaluation benchmarks (such as Table 1 in `EVAL_SUMMARY.md`), test datasets are organized in homogeneous chunks: all sentences in an AI document are AI-generated ($y_i = 1$), and all sentences in a human document are human-written ($y_i = 0$).

Under this condition, a 3-sentence rolling average:
$$\tilde{s}_i = \frac{1}{2W + 1} \sum_{j=i-W}^{i+W} s_j$$
acts as a powerful noise-reduction filter. Because all neighbors share the exact same ground-truth label, smoothing cancels out per-sentence variance and artificially inflates performance (jumping from 88.70% to 95.97% accuracy on RoBERTa).

### The Critical Flaw in Practice:
In real-world academic integrity and co-piloted writing scenarios, documents are **mixed-authorship**:
- A human student writes an entire essay but pastes in 1–2 AI-generated paragraphs or sentences.
- An author writes an academic paper and uses an LLM to rewrite specific sentences ($k = 10\%, 25\%, 50\%$).

Under mixed authorship, smoothing can cause **boundary smearing**:
1. It may dilute the AI signal of an isolated AI sentence below the detection threshold (**false negative**).
2. It smears the high AI score of an inserted sentence into the adjacent human sentences, falsely accusing human text (**elevated human False Positive Rate**).

---

## 2. Dataset Construction Methodology

The synthetic mixed-authorship dataset is constructed **strictly from `data/processed/test.parquet`** (Project Rule 3: topic-split, never seen in training).

### Algorithm:
1. **Topic Grouping:** Group human sentences ($y = 0$) by their `topic_id`.
2. **Document Segmentation:** Partition human sentences into contiguous paragraph blocks of length $L = 10$ sentences.
3. **AI Needle Injection:** For each replacement ratio $k \in \{10\%, 25\%, 50\%\}$:
   - Calculate number of sentences to replace: $m = \text{round}(L \times k / 100)$.
   - Choose $m$ distinct positions uniformly at random within the document using seed 42.
   - Sample $m$ AI sentences ($y = 1$) from the **same `topic_id`** (or same domain if topic pool is exhausted) to preserve contextual and topical coherence.
   - Replace the human sentences at the chosen positions with the sampled AI sentences.
4. **Ground-Truth Tracking:**
   - Untouched human sentences retain ground truth label $y = 0$.
   - Inserted AI sentences retain ground truth label $y = 1$.

---

## 3. Evaluated Window Configurations & Metrics

For each replacement level $k \in \{10\%, 25\%, 50\%\}$, we evaluate predictions across three smoothing regimes:
- **Window 1 (Raw / No Smoothing):** Evaluates raw model precision without spatial aggregation.
- **Window 3 (Standard 3-Sentence Rolling Average):** The baseline smoothing filter.
- **Window 5 (5-Sentence Rolling Average):** A wider smoothing filter to observe degradation trends.

### Measured Metrics:
1. **Sentence-Level Macro-F1:** Overall classification balance.
2. **Inserted AI Sentence Recall:** Fraction of AI needle sentences successfully flagged ($\ge \text{threshold}$).
3. **Untouched Human Sentence FPR:** Fraction of original human sentences falsely flagged as AI due to boundary smearing.
4. **Document-Level Detection Rate:** Fraction of mixed documents flagged as containing AI text ($\ge 1$ sentence flagged).

All empirical results are saved to `results/mixed_authorship.csv` and plotted in `results/figures/mixed_authorship_tradeoff.png`.
