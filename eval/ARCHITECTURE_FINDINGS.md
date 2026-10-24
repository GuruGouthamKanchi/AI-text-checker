# Architecture Findings: Current Inference Pipeline Analysis

**Date:** 2026-10-23  
**Files Inspected:**
- `src/highlighter.py` (Core inference, sentence segmentation, smoothing, heuristics)
- `src/app.py` (FastAPI backend service, document-level scoring, confidence tiers)
- `src/train_sentence_roberta.py` (Training script for RoBERTa sentence model)
- `src/train_modernbert_academic.py` (Training script for ModernBERT)
- `models/modernbert-academic/` (Architecture: `ModernBertForSequenceClassification`, max context 8192)
- `models/roberta-sentence-academic-v2/` (Architecture: `RobertaForSequenceClassification`, max context 514)

---

## 1. Executive Summary: What the Code Actually Does

While project documentation and manuscripts describe an attention-gated dual-branch pipeline ($P_{ensemble} = w_1 P_{MB} + w_2 P_{max} + (1 - w_1 - w_2) g$), **the current production code implements a heuristic rule-based fusion system operated at the sentence level**.

Key takeaways:
1. **ModernBERT does NOT process the full document at once.** ModernBERT is executed inside a sentence-by-sentence loop over a 3-sentence local sliding window (`prev_sent sent next_sent`), truncated to `max_length=512`.
2. **RoBERTa is strictly sentence-level.** RoBERTa evaluates each sentence in isolation (`max_length=512`).
3. **No neural gating network ($g$) exists in the code.** There are no gating layer weights ($v, b_g$). Instead, score fusion is performed via conditional `max()` logic paired with hardcoded heuristic boosts (+0.15 for humanizer transition regex matches, +0.10 for low document burstiness).
4. **Document-level score is a piecewise word-weighted sum**, not the paper's soft-voting formula.

---

## 2. Granularity: Sentence-Level vs. Document-Level

### A. Sentence-Level Operations
- **Segmentation (`src/highlighter.py:segment_sentences`):**
  - Text is split into sentences using `nltk.sent_tokenize`.
  - LaTeX display math (`$$...$$`, `\[...\]`) and inline math (`$...$`, `\(...\)`) are masked prior to tokenization and restored afterwards.
  - Bibliography blocks (`[1]`, `[2]`, etc.) are partitioned into individual reference entries.
- **RoBERTa Micro-Branch (`src/highlighter.py:run_predictions`):**
  - Input: Individual sentence string `sent`.
  - Tokenization: `rob_tokenizer(sent, truncation=True, max_length=512, return_tensors="pt")`.
  - Output: `prob_rob = float(torch.softmax(rob_logits, dim=-1).cpu().numpy()[0][1])`.
- **ModernBERT Branch (`src/highlighter.py:run_predictions`):**
  - Input: 3-sentence local window:
    ```python
    prev_sent = sentences[idx - 1] if idx > 0 else ""
    next_sent = sentences[idx + 1] if idx < len(sentences) - 1 else ""
    window_context = f"{prev_sent} {sent} {next_sent}".strip()
    ```
  - Tokenization: `mb_tokenizer(window_context if len(window_context) > len(sent) else sent, truncation=True, max_length=512, return_tensors="pt")`.
  - Output: `prob_mb = float(torch.softmax(mb_logits, dim=-1).cpu().numpy()[0][1])`.
  - **Note:** ModernBERT's native 8,192-token context window is not used here; it is clamped to 512 tokens on a 3-sentence window.

### B. Document-Level Operations
- **Structural Burstiness Calculation (`src/highlighter.py:153-155`):**
  - Document-level word count standard deviation is computed across all sentences:
    ```python
    sent_word_lens = [len([w for w in s.split() if w.strip()]) for s in sentences]
    burstiness_std = float(np.std(sent_word_lens)) if len(sent_word_lens) > 1 else 10.0
    is_low_burstiness = (burstiness_std < 4.0 and len(sentences) >= 5)
    ```
  - `is_low_burstiness` is a global document-level boolean flag passed into each sentence's fusion calculation.
- **Document Overall AI Percentage (`src/app.py:288-306`):**
  - The final document-level score reported in the API is a word-weighted piecewise aggregation across all post-smoothed sentences:
    ```python
    for item in processed_sentences:
        words = len([w for w in text.split() if w.strip()])
        doc_words_total += words
        if score >= 0.75:
            flagged_words += words * 1.0
        elif score >= 0.60:
            flagged_words += words * max(0.85, score)
        elif score >= 0.50:
            flagged_words += words * 0.50
        # score < 0.50 contributes 0.0

    overall_percentage = round(min(100.0, (flagged_words / doc_words_total) * 100.0), 1)
    ```

---

## 3. Score Fusion Logic in Code Today

In `src/highlighter.py` (lines 187–212), the fusion algorithm for each sentence combines `prob_mb` and `prob_rob` as follows:

```python
# 1. Base score is the maximum of the two models
base_score = max(prob_rob, prob_mb)

# 2. Check for humanizer transition regex matches (e.g., 'in addition', 'furthermore', 'delve', 'pivotal', 'realm', etc.)
has_humanizer_transition = bool(re.search(HUMANIZER_TRANSITIONS_REGEX, sent_lower))

# 3. Decision branches
if prob_mb is not None and prob_rob is not None:
    if has_humanizer_transition:
        # Boost up to 0.99 for suspected transition phrases
        prob_ai = min(0.99, max(base_score, 0.65) + 0.15)
    elif is_low_burstiness and base_score >= 0.35:
        # Boost by +0.10 if entire document has unnaturally uniform sentence length
        prob_ai = min(0.99, base_score + 0.10)
    elif prob_rob >= 0.38 or prob_mb >= 0.38:
        # If either model detects AI with moderate confidence, take max
        prob_ai = base_score
    else:
        # Only take 50/50 average if BOTH models predict < 0.38
        prob_ai = 0.50 * prob_mb + 0.50 * prob_rob

# 4. Metadata Guard Override
is_metadata = bool(re.search(r'^\s*(?:independent research|university|department|school|e-mail|email|july \d{4}...)\s*$', sent_lower))
if is_metadata:
    prob_ai = 0.02
```

### 3-Sentence Rolling Average Smoothing (`src/highlighter.py:smooth_predictions`):
After raw fusion across all sentences, a 3-sentence rolling average is applied:
- **First sentence ($i=0$):** $0.70 \times s_0 + 0.30 \times s_1$
- **Last sentence ($i=n-1$):** $0.70 \times s_{n-1} + 0.30 \times s_{n-2}$
- **Interior sentences ($0 < i < n-1$):** $0.25 \times s_{i-1} + 0.50 \times s_i + 0.25 \times s_{i+1}$

---

## 4. Complete Catalog of Thresholds Used in the Pipeline

| Pipeline Stage | Threshold | Condition / Action | Source Location |
|---|:---:|---|---|
| **Raw Binary Prediction** | `0.50` | `label = 1 if prob_ai >= 0.5 else 0` | `src/highlighter.py:216` |
| **Max-Pool Fallback Trigger** | `0.38` | If `prob_rob >= 0.38 or prob_mb >= 0.38`, use `max()`, else `0.5*mb + 0.5*rob` | `src/highlighter.py:194` |
| **Low Burstiness std** | `4.0` | `burstiness_std < 4.0` and `len(sentences) >= 5` flags document | `src/highlighter.py:155` |
| **Low Burstiness Boost** | `0.35` | If `is_low_burstiness and base_score >= 0.35`, boost by `+0.10` | `src/highlighter.py:192` |
| **Humanizer Transition Floor** | `0.65` | Transition match floors score at `0.65`, then adds `+0.15` (min `0.80`) | `src/highlighter.py:191` |
| **Metadata Override** | N/A | Regex match forces `prob_ai = 0.02` | `src/highlighter.py:211` |
| **Smoothed Sentence Label** | `0.60` | `label = 1 if avg_score >= 0.60 else 0` | `src/highlighter.py:773` |
| **App High Confidence Tier** | `0.75` | Highlight red; full 1.0 word weight | `src/app.py:249, 296` |
| **App Medium Confidence Tier**| `0.60` | Highlight orange; `max(0.85, score)` word weight | `src/app.py:252, 298` |
| **App Low Weight Trigger** | `0.50` | `0.50 <= score < 0.60` receives `0.50` word weight | `src/app.py:300` |
| **MSTTR High (Human)** | `68.0%` | Rich vocabulary label | `src/highlighter.py:247` |
| **MSTTR Medium** | `58.0%` | Standard vocabulary label | `src/highlighter.py:250` |
| **Burstiness CV High (Human)** | `45.0%` | Dynamic structure label | `src/highlighter.py:258` |
| **Burstiness CV Medium** | `30.0%` | Moderate variation label | `src/highlighter.py:261` |

---

## 5. Architectural Contrast: Code vs. Paper Formulation

| Feature | Paper Manuscript Claim | Current Codebase Reality |
|---|---|---|
| **ModernBERT Context** | 8,192 un-truncated tokens across full document | 3-sentence sliding window, truncated at 512 tokens |
| **RoBERTa Context** | 512-token sentence classifier | 512-token sentence classifier (`MAX_LENGTH=64` during fine-tuning) |
| **Gating Mechanism** | Non-linear gating layer $g = \sigma(v^T [P_{MB}; P_{max}] + b_g)$ | **None.** No gating weights exist; uses conditional max/regex heuristics |
| **Ensemble Formula** | $P = w_1 P_{MB} + w_2 P_{max} + (1 - w_1 - w_2) g$ | Heuristic rules: $\max(P_{rob}, P_{mb})$ + transition/burstiness bonuses |
| **Document Aggregation** | Gated combination of macro $P_{MB}$ and max sentence $P_{max}$ | Piecewise word-count weighted sum of smoothed sentence scores |
