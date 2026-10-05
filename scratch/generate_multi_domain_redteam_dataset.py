import sys
import os
import json
import time

sys.path.append(os.path.abspath("."))

from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, run_predictions, smooth_predictions
from src.closed_loop_engine import ClosedLoopHumanizer

# Multi-Domain Synthetic Academic Papers
MULTI_DOMAIN_PAPERS = {
    "Computer Science & AI": """
Synergistic Neural Frameworks for Automated Deep Learning Optimization

AI Research Group
Department of Computer Science

Abstract—Large Language Models (LLMs) have become the dominant paradigm in natural language processing, driven by the Transformer architecture and its capacity to scale across data, parameters, and compute. In addition, it is crucial to delve into the complex multi-faceted paradigm of artificial intelligence integration within contemporary educational ecosystems. Consequently, stakeholders must carefully balance innovation with ethical guidelines. In this paper, we explore a thorough framework that serves as proof to the impact of synthetic neural networks.

1. INTRODUCTION
Furthermore, in modern computer science, notably, deep learning architectures play a key role in advancing computational intelligence. In addition, researchers have increasingly focused on optimizing parameter distributions across complex multi-dimensional search spaces. As has been shown in recent literature, the loss optimization trajectory follows standard empirical risk minimization. In addition, it is worth emphasising that standard gradient descent methods often hit local minima convergence. Thus, adaptive optimization is essential to ensure robust generalization performance across diverse benchmark datasets.
""",

    "Biomedical & Medicine": """
High-Precision CRISPR-Cas9 Gene Editing Protocols in Human Pluripotent Stem Cells

Biomedical Research Institute
Department of Molecular Genetics

Abstract—The rapid evolution of genomic engineering has transformed molecular medicine. In addition, it is important to explore into the complex challenges of targeted gene disruption to ensure patient safety. Consequently, researchers must evaluate off-target cleavage events using next-generation sequencing assays. In this paper, we explore a thorough framework for optimizing Cas9 ribonucleoprotein delivery into human stem cells.

1. INTRODUCTION
Furthermore, in contemporary biotechnology, notably, guide RNA design plays a key role in maximizing cleavage efficiency. In addition, clinicians have increasingly focused on non-viral electroporation methods across primary cell lineages. As has been shown in recent literature, non-homologous end joining repair trajectories follow predictable insertion-deletion profiles. Thus, high-fidelity Cas9 variants are essential to mitigate unintended genomic alterations.
""",

    "Physics & Quantum Computing": """
Fault-Tolerant Quantum Key Distribution Over Fiber-Optic Networks

Quantum Information Group
Department of Applied Physics

Abstract—Quantum key distribution (QKD) leverages fundamental principles of quantum mechanics to guarantee information-theoretic security. In addition, it is crucial to delve into the complex multi-faceted paradigm of decoy-state protocols within fiber networks. Consequently, experimentalists must balance secret key rates with channel attenuation constraints. In this paper, we explore a thorough framework that serves as proof of quantum advantage in metropolitan links.

1. INTRODUCTION
Furthermore, in modern quantum physics, notably, superconducting nanowire single-photon detectors play a key role in advancing quantum communication. In addition, physicists have increasingly focused on mitigating phase noise across polarization-encoded channels. As has been shown in recent literature, quantum bit error rate scales linearly with dark count probability. Thus, active phase stabilization is essential to ensure long-distance key generation.
""",

    "Economics & Social Science": """
Macroeconomic Propagation of Fiscal Policy In Stochastic DSGE Models

Institute for Economic Policy
Department of Economics

Abstract—In modern macroeconomics, dynamic stochastic general equilibrium (DSGE) models incorporate nominal rigidities to evaluate monetary policy. In addition, it is important to explore into the complex challenges of financial friction modeling during economic shocks. Consequently, policymakers must balance inflation targeting with employment growth. In this paper, we explore a thorough framework for Bayesian parameter calibration.

1. INTRODUCTION
Furthermore, in applied econometrics, notably, vector autoregression models play a key role in forecasting macroeconomic fluctuations. In addition, economists have increasingly focused on structural impulse response functions across heterogeneous household datasets. As has been shown in recent literature, consumption multipliers depend strongly on marginal propensity to consume. Thus, adaptive fiscal intervention is essential to sustain long-term economic stability.
"""
}

def evaluate_paper(doc_text: str):
    paragraphs = split_into_smart_paragraphs(doc_text)
    sentences = []
    for p in paragraphs:
        sentences.extend(segment_sentences(p))
    preds, _ = run_predictions(sentences)
    preds = smooth_predictions(preds)
    
    flagged_words = 0.0
    doc_words = 0
    flagged_high = 0
    flagged_mid = 0

    for item in preds:
        t = item["sentence"]
        s = item["score"]
        w = len([x for x in t.split() if x.strip()])
        doc_words += w
        if s >= 0.75:
            flagged_words += w * 1.0
            flagged_high += 1
        elif s >= 0.60:
            flagged_words += w * max(0.85, s)
            flagged_mid += 1
        elif s >= 0.50:
            flagged_words += w * 0.50

    pct = round(min(100.0, (flagged_words / doc_words) * 100.0), 1) if doc_words > 0 else 0.0
    return pct, len(sentences), flagged_high, flagged_mid, preds

print("=================================================================")
print("MULTI-DOMAIN ADVERSARIAL RED-TEAMING DATASET GENERATION")
print("=================================================================\n")

closed_loop = ClosedLoopHumanizer()
adversarial_dataset = []

for domain, paper_text in MULTI_DOMAIN_PAPERS.items():
    print(f"[Domain: {domain}] Running Baseline Analysis...")
    raw_pct, raw_sents, raw_h, raw_m, _ = evaluate_paper(paper_text)
    print(f"  Baseline Raw AI Score: {raw_pct}% ({raw_h} High / {raw_m} Mid out of {raw_sents} sents)")

    print(f"  Generating Closed-Loop Humanized Evasion Candidate...")
    paragraphs = split_into_smart_paragraphs(paper_text)
    humanized_paras = []

    for p in paragraphs:
        if not p.strip():
            continue
        events = list(closed_loop.process_closed_loop_stream(p, tone="academic", intensity=0.85, min_semantic_sim=0.45))
        h_sents = [e["humanized"] for e in events if e.get("type") == "step" and "humanized" in e]
        humanized_paras.append(" ".join(h_sents) if h_sents else p)

    humanized_text = "\n\n".join(humanized_paras)
    hum_pct, hum_sents, hum_h, hum_m, hum_preds = evaluate_paper(humanized_text)
    print(f"  Humanized Red-Team AI Score: {hum_pct}% ({hum_h} High / {hum_m} Mid out of {hum_sents} sents)")
    print(f"  Evasion Delta (Delta S): {raw_pct - hum_pct:+.1f}%\n")

    adversarial_dataset.append({
        "domain": domain,
        "raw_text": paper_text,
        "raw_score": raw_pct,
        "humanized_text": humanized_text,
        "humanized_score": hum_pct,
        "predictions": hum_preds
    })

with open("scratch/multi_domain_adversarial_dataset.json", "w", encoding="utf-8") as f:
    json.dump(adversarial_dataset, f, indent=2)

print("Saved scratch/multi_domain_adversarial_dataset.json successfully.")
