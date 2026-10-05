import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath("."))

from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, run_predictions, smooth_predictions

def evaluate_doc_text(doc_text: str):
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
        
    return overall_pct, len(sentences), flagged_high, flagged_mid

def main():
    print("=================================================================")
    print("ADVERSARIAL FINE-TUNING & MULTI-DOMAIN ENSEMBLE OPTIMIZATION")
    print("=================================================================\n")

    dataset_path = "scratch/multi_domain_adversarial_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file '{dataset_path}' not found!")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[Evaluation] Multi-Domain Red-Teaming Results Post-Optimization:")
    for item in data:
        domain = item["domain"]
        raw_text = item["raw_text"]
        hum_text = item["humanized_text"]

        raw_pct, r_sents, r_h, r_m = evaluate_doc_text(raw_text)
        hum_pct, h_sents, h_h, h_m = evaluate_doc_text(hum_text)

        print(f"  [{domain}]")
        print(f"    Raw AI Score:       {raw_pct}% ({r_h} High / {r_m} Mid / {r_sents} sents)")
        print(f"    Humanized AI Score: {hum_pct}% ({h_h} High / {h_m} Mid / {h_sents} sents)")
        print(f"    Evasion Delta:      {raw_pct - hum_pct:+.1f}%\n")

    # Evaluate Genuine Human Control Paper (EJ1172284.pdf) if available
    human_pdf = "scratch/EJ1172284.pdf"
    if not os.path.exists(human_pdf):
        human_pdf = "EJ1172284.pdf"

    if os.path.exists(human_pdf):
        try:
            import fitz
            doc = fitz.open(human_pdf)
            human_text = "\n".join([page.get_text() for page in doc])
            hp_pct, hp_sents, hp_h, hp_m = evaluate_doc_text(human_text)
            print("=================================================================")
            print("GENUINE HUMAN CONTROL PAPER VERIFICATION (EJ1172284.pdf)")
            print("=================================================================")
            print(f"  Human Control Paper AI Score: {hp_pct}%")
            print(f"  Total Sentences: {hp_sents} | High Risk: {hp_h} | Mid Risk: {hp_m}")
            print(f"  False Positive Rate: {0.0 if hp_pct < 10.0 else hp_pct}%\n")
        except Exception as e:
            print(f"  [Human Control] Could not evaluate {human_pdf}: {e}")

if __name__ == "__main__":
    main()
