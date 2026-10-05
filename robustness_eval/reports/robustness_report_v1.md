# Adversarial Robustness Evaluation Report

## Overview Metrics
- **Total Evaluations**: 1
- **Original Text Accuracy**: 100.0%
- **Rewritten Text Accuracy**: 0.0%
- **Accuracy Drop**: **100.0%**
- **Verdict Flip Rate (AI → Human)**: 100.0%
- **Mean Score Drop**: 50.0%
- **Adversarial Resilience Index (ARI)**: 0.00 / 1.00
- **Confidence Under Attack Signal**: **Critical Vulnerability (High Evasion Risk)**

## Rewrite Method Breakdown
| Rewrite Method | Samples | Orig Accuracy | Rewritten Accuracy | Accuracy Drop | Mean Score Drop | Flip Rate |
|---|---|---|---|---|---|---|
| combined_v1 | 1 | 100.0% | 0.0% | 100.0% | 50.0% | 100.0% |

## Sentence-Level Fooling Patterns
- **LLM Cliché Term Substitution (Vocabulary)** (Occurrences: 1, Impact: -50.0% AI score)
  *Description*: Replacing high-signal LLM indicator terms ('delve', 'crucial', 'leverage') with human synonyms.