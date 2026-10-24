# Comprehensive Audit of Hand-Written Heuristic Rules in `src/highlighter.py`

This document details every hand-written heuristic, regex pattern, threshold boost, and suppression rule implemented in the current production inference pipeline (`src/highlighter.py:run_predictions`).

---

## 1. Inventory of Hand-Written Rules

### Rule 1: Humanizer Transition Artifacts & Multi-Domain Paraphrase Regex
- **Variable Name:** `HUMANIZER_TRANSITIONS_REGEX`
- **Location:** `src/highlighter.py`, lines 150, 184, 189–191
- **Pattern Match:** Case-insensitive word-boundary regex matching 54 transition phrases and AI fingerprint ngrams:
  ```regex
  \b(in addition|furthermore|notably|it is worth|as a result|overall|additionally|thus|delve|testament|pivotal|underscores|tapestry|moreover|seamlessly|consequently|paramount|realm|beacon|thorough framework|serves as proof|complex challenges|play a key role|plays a key role|empirical evaluations demonstrate|increasingly focused on|as has been shown|gradient descent methods|generalization performance|synergistic optimization|key challenges|objective function|empirical risk minimization|important importance|future research into|crucial framework|mitigating oxidative stress|in modern oncology|therapeutic efficacy|clinical trials have|play a pivotal role|in modern quantum physics|mitigating phase noise|active phase stabilization|error rate scales linearly|in modern macroeconomics|in applied econometrics|stochastic general equilibrium|nominal rigidities|adaptive fiscal intervention|in summary|taken together)\b
  ```
- **Exact Algorithmic Effect:**
  If any phrase in `HUMANIZER_TRANSITIONS_REGEX` is detected in a sentence:
  $$\text{prob\_ai} = \min\left(0.99, \max(\text{base\_score}, 0.65) + 0.15\right)$$
  where $\text{base\_score} = \max(\text{prob\_rob}, \text{prob\_mb})$.
- **Vulnerability / Side Effect:**
  If a human author writes a completely human sentence that contains any formal transition such as *"In addition, we conducted five trials"* or *"Furthermore, empirical evaluations demonstrate that..."*, even if the neural models output $\text{prob\_rob} = 0.01$ and $\text{prob\_mb} = 0.01$, the score is immediately inflated:
  $$\text{prob\_ai} = \max(0.01, 0.65) + 0.15 = 0.80$$
  This causes severe false positive inflation on formal human writing and non-native essays that rely heavily on transition word signposting.

---

### Rule 2: Low-Burstiness Stylometric Bonus
- **Variable Name:** `is_low_burstiness`
- **Location:** `src/highlighter.py`, lines 152–155, 192–193
- **Mathematical Formulation:**
  - Let $L_1, L_2, \dots, L_m$ be the word lengths of all sentences in the document.
  - $\sigma_{\text{burst}} = \text{std}(L)$
  - Condition: $\sigma_{\text{burst}} < 4.0$ and $m \ge 5$.
- **Exact Algorithmic Effect:**
  If the document exhibits low sentence length variation ($\sigma_{\text{burst}} < 4.0$) and the neural base score is at least $0.35$:
  $$\text{prob\_ai} = \min(0.99, \text{base\_score} + 0.10)$$
- **Vulnerability / Side Effect:**
  Short, uniform human paragraphs (e.g., technical definitions, bullet point prose, or standardized testing essays) often naturally have low length variance ($\sigma < 4.0$), bumping borderline human sentences over the decision threshold.

---

### Rule 3: Non-Linear Gated Max-Pooling
- **Location:** `src/highlighter.py`, lines 187–197
- **Exact Algorithmic Effect:**
  When neither transition nor burstiness heuristics trigger:
  $$\text{prob\_ai} = \begin{cases} \max(\text{prob\_rob}, \text{prob\_mb}) & \text{if } \text{prob\_rob} \ge 0.38 \text{ or } \text{prob\_mb} \ge 0.38 \\ 0.50 \cdot \text{prob\_mb} + 0.50 \cdot \text{prob\_rob} & \text{otherwise} \end{cases}$$
- **Vulnerability / Side Effect:**
  By taking the maximum score whenever either model hits $0.38$, any single false alarm by one model (e.g. RoBERTa misclassifying a sentence) immediately overrides the calm confidence of the other model.

---

### Rule 4: Standalone Title and Affiliation Metadata Suppression
- **Variable Name:** `is_metadata`
- **Location:** `src/highlighter.py`, lines 205–211
- **Pattern Match:**
  ```regex
  ^\s*(?:independent research|university|department|school|e-mail|email|july \d{4}|june \d{4}|august \d{4}|january \d{4}|february \d{4}|march \d{4}|april \d{4}|may \d{4}|september \d{4}|october \d{4}|november \d{4}|december \d{4})\s*$
  ```
- **Exact Algorithmic Effect:**
  If a standalone sentence matches academic metadata or header formats:
  $$\text{prob\_ai} = 0.02$$
- **Rationale:**
  Prevents institutional affiliations, university names, and dates from being flagged as AI text.

---

## 2. Summary Table of Heuristic Effects

| Rule | Trigger Condition | Score Transformation | Primary Vulnerability |
|---|---|---|---|
| **Humanizer Transitions** | Word in `HUMANIZER_TRANSITIONS_REGEX` | $\min(0.99, \max(\text{base}, 0.65) + 0.15)$ | High FPR on formal human academic & non-native text |
| **Low Burstiness** | Document $\text{std}(\text{words}) < 4.0$ & $\ge 5$ sents | $\min(0.99, \text{base} + 0.10)$ if $\text{base} \ge 0.35$ | Penalizes disciplined, uniformly structured human paragraphs |
| **Gated Max-Pool** | Either model $\ge 0.38$ | $\max(\text{prob\_rob}, \text{prob\_mb})$ | Propagates single-model false positives |
| **Metadata Guard** | Standalone author/institution/date regex | $\text{prob\_ai} = 0.02$ | Could suppress AI-generated author blocks if cloned verbatim |
