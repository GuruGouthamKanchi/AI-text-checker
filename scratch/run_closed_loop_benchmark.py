import sys
import os
import time

sys.path.append(os.path.abspath("."))

from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, run_predictions, smooth_predictions
from src.closed_loop_engine import ClosedLoopHumanizer

raw_synthetic_paper = """
Architectures and Training Paradigms of Large Language Models: A Survey

AI Research Group
Department of Computer Science

Abstract—Large Language Models (LLMs) have become the dominant paradigm in natural language processing, driven by the Transformer architecture and its capacity to scale across data, parameters, and compute. In addition, it is crucial to delve into the complex multi-faceted paradigm of artificial intelligence integration within contemporary educational ecosystems. Consequently, stakeholders must carefully balance innovation with ethical guidelines. In this paper, we explore a thorough framework that serves as proof to the impact of synthetic neural networks.

1. INTRODUCTION
Furthermore, in modern computer science, notably, deep learning architectures play a key role in advancing computational intelligence. In addition, researchers have increasingly focused on optimizing parameter distributions across complex multi-dimensional search spaces. As has been shown in recent literature, the loss optimization trajectory follows standard empirical risk minimization. In addition, it is worth emphasising that standard gradient descent methods often hit local minima convergence. Thus, adaptive optimization is essential to ensure robust generalization performance across diverse benchmark datasets.

2. METHODOLOGY AND FRAMEWORK
To address these key challenges, we propose a synergistic optimization pipeline that integrates multi-head self-attention mechanisms with dynamic loss scaling. Delving into the realm of quantum computing reveals a plethora of transformative opportunities. As quantum bits exhibit superposition and entanglement, computational speedups across cryptographic domains become increasingly achievable. Overall, this study highlights the important importance of architectural optimization in deep learning systems. In addition, our empirical findings provide the way for future research into automated hyperparameter tuning and scalable neural optimization.

3. CONCLUSION
In conclusion, this study underscores the paramount importance of sustainable energy transition strategies. By leveraging advanced photovoltaic technologies, society can foster a greener future while driving economic growth seamlessly. Furthermore, exploring the nuanced interplay between macroeconomics and environmental sustainability provides crucial insights. Utilizing predictive modeling allows policymakers to design targeted interventions that foster long-term economic resilience.
"""

def evaluate_paper_score(doc_text: str):
    raw_paragraphs = split_into_smart_paragraphs(doc_text)
    sentences = []
    for p in raw_paragraphs:
        sentences.extend(segment_sentences(p))
        
    preds, is_sim = run_predictions(sentences)
    preds = smooth_predictions(preds)
    
    flagged_words = 0.0
    doc_words_total = 0
    flagged_high = 0
    flagged_mid = 0

    for item in preds:
        text = item["sentence"]
        score = item["score"]
        words = len([w for w in text.split() if w.strip()])
        doc_words_total += words
        
        if score >= 0.75:
            flagged_words += words * 1.0
            flagged_high += 1
        elif score >= 0.60:
            flagged_words += words * max(0.85, score)
            flagged_mid += 1
        elif score >= 0.50:
            flagged_words += words * 0.50

    if doc_words_total > 0:
        overall_pct = round(min(100.0, (flagged_words / doc_words_total) * 100.0), 1)
    else:
        overall_pct = 0.0
        
    return {
        "overall_percentage": overall_pct,
        "sentences_count": len(sentences),
        "words_count": doc_words_total,
        "flagged_high": flagged_high,
        "flagged_mid": flagged_mid,
        "predictions": preds
    }

print("=================================================================")
print("ADVERSARIAL CLOSED-LOOP BENCHMARK & RED-TEAMING EVALUATION")
print("=================================================================\n")

print("[Step 1] Running Baseline Analysis on Raw Synthetic Test Paper...")
res_raw = evaluate_paper_score(raw_synthetic_paper)
print(f"  Baseline Raw AI Score: {res_raw['overall_percentage']}%")
print(f"  Total Sentences: {res_raw['sentences_count']} | High Risk: {res_raw['flagged_high']} | Medium Risk: {res_raw['flagged_mid']}\n")

print("[Step 2] Passing Synthetic Paper Through Closed-Loop Humanizer (AccaHumanize-CL)...")
closed_loop = ClosedLoopHumanizer()

paragraphs = split_into_smart_paragraphs(raw_synthetic_paper)
humanized_paragraphs = []

for p in paragraphs:
    if len(p.strip()) == 0:
        continue
    events = list(closed_loop.process_closed_loop_stream(p, tone="academic", intensity=0.85, min_semantic_sim=0.45))
    h_sents = [e["humanized"] for e in events if e.get("type") == "step" and "humanized" in e]
    if h_sents:
        humanized_paragraphs.append(" ".join(h_sents))
    else:
        humanized_paragraphs.append(p)

humanized_paper = "\n\n".join(humanized_paragraphs)

with open("scratch/humanized_paper_test.txt", "w", encoding="utf-8") as f:
    f.write(humanized_paper)

print(f"  Humanized Paper Generated ({len(humanized_paper.split())} words)\n")

print("[Step 3] Running Adversarial Detection Analysis on Closed-Loop Humanized Paper...")
res_humanized = evaluate_paper_score(humanized_paper)

delta = res_raw['overall_percentage'] - res_humanized['overall_percentage']

out_lines = []
out_lines.append("=================================================================")
out_lines.append("ADVERSARIAL COMPARISON MATRIX (BEFORE VS AFTER CLOSED-LOOP)")
out_lines.append("=================================================================")
out_lines.append(f"Metric                        | Baseline Raw Synthetic | Humanized Closed-Loop")
out_lines.append(f"-----------------------------------------------------------------")
out_lines.append(f"Overall AI Detection Score    | {res_raw['overall_percentage']:6.1f}%                | {res_humanized['overall_percentage']:6.1f}%")
out_lines.append(f"Total Sentences               | {res_raw['sentences_count']:6d}                 | {res_humanized['sentences_count']:6d}")
out_lines.append(f"High-Risk AI Sentences        | {res_raw['flagged_high']:6d}                 | {res_humanized['flagged_high']:6d}")
out_lines.append(f"Medium-Risk AI Sentences      | {res_raw['flagged_mid']:6d}                 | {res_humanized['flagged_mid']:6d}")
out_lines.append(f"Total AI Flagged Sentences    | {res_raw['flagged_high']+res_raw['flagged_mid']:6d}                 | {res_humanized['flagged_high']+res_humanized['flagged_mid']:6d}")
out_lines.append(f"Adversarial Evasion Delta     | --                     | {delta:+6.1f}%")
out_lines.append("=================================================================")

report_str = "\n".join(out_lines)
print(report_str)

with open("scratch/closed_loop_benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(report_str)
