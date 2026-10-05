import fitz
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences

doc = fitz.open('D:/AI_Text_Checker/old_only_roBERTa.pdf')
page3_text = doc[2].get_text()

lines = page3_text.splitlines()
body_lines = [l for l in lines if not l.startswith('ID: vpr:') and not l.startswith('Page ') and not l.startswith('VeriPaper')]
raw_text = "\n".join(body_lines)

paragraphs = split_into_smart_paragraphs(raw_text)
sentences = []
for p in paragraphs:
    sentences.extend(segment_sentences(p))

device = "cuda" if torch.cuda.is_available() else "cpu"

roberta_path = "models/roberta-sentence-academic-v2"
roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device)
roberta_model.eval()

modernbert_path = "models/modernbert-academic"
modernbert_tok = AutoTokenizer.from_pretrained(modernbert_path)
modernbert_model = AutoModelForSequenceClassification.from_pretrained(modernbert_path).to(device)
modernbert_model.eval()

out_lines = []
out_lines.append("================ EMPIRICAL ROOT CAUSE ANALYSIS ================")
out_lines.append(f"Total Sentences: {len(sentences)}\n")

roberta_probs = []
modernbert_probs = []
ensemble_linear = []
ensemble_max = []
ensemble_adaptive = []

for idx, sent in enumerate(sentences):
    inputs_r = roberta_tok(sent, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        prob_r = float(torch.softmax(roberta_model(**inputs_r).logits, dim=-1)[0, 1].item())
    
    inputs_m = modernbert_tok(sent, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        prob_m = float(torch.softmax(modernbert_model(**inputs_m).logits, dim=-1)[0, 1].item())
        
    lin_score = 0.65 * prob_m + 0.35 * prob_r
    max_score = max(prob_r, prob_m)
    # Adaptive score with humanizer boost if transition phrases detected
    has_humanizer_transition = bool(re.search(r'\b(in addition|furthermore|notably|it is worth|as a result|overall|additionally|thus)\b', sent.lower()))
    boost = 0.15 if has_humanizer_transition else 0.0
    adapt_score = min(0.99, max_score + boost)

    roberta_probs.append(prob_r)
    modernbert_probs.append(prob_m)
    ensemble_linear.append(lin_score)
    ensemble_max.append(max_score)
    ensemble_adaptive.append(adapt_score)

    clean_sent = sent.encode('ascii', 'ignore').decode('ascii')
    out_lines.append(f"S#{idx+1:02d} | RoBERTa: {prob_r:.4f} | ModernBERT: {prob_m:.4f} | Linear (65/35): {lin_score:.4f} | Max-Pool: {max_score:.4f} | Adapt: {adapt_score:.4f}")
    out_lines.append(f"     Text: {clean_sent[:80]}...\n")

out_lines.append("================ SUMMARY COMPARISON ================")
out_lines.append(f"RoBERTa Alone (Avg Score)      : {np.mean(roberta_probs)*100:.2f}% | Sentences >= 0.60: {sum(1 for p in roberta_probs if p>=0.60)}")
out_lines.append(f"ModernBERT Alone (Avg Score)   : {np.mean(modernbert_probs)*100:.2f}% | Sentences >= 0.60: {sum(1 for p in modernbert_probs if p>=0.60)}")
out_lines.append(f"Current Linear Ensemble (65/35): {np.mean(ensemble_linear)*100:.2f}% | Sentences >= 0.60: {sum(1 for p in ensemble_linear if p>=0.60)}")
out_lines.append(f"Max-Pooled Ensemble            : {np.mean(ensemble_max)*100:.2f}% | Sentences >= 0.60: {sum(1 for p in ensemble_max if p>=0.60)}")
out_lines.append(f"Adaptive Humanizer-Boosted     : {np.mean(ensemble_adaptive)*100:.2f}% | Sentences >= 0.60: {sum(1 for p in ensemble_adaptive if p>=0.60)}")

with open('scratch/root_cause_results.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))

print("Results written to scratch/root_cause_results.txt")
