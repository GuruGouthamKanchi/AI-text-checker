import fitz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, smooth_predictions

pdf_path = 'D:/AI_Text_Checker/old_only_roBERTa.pdf'
doc = fitz.open(pdf_path)
page3_text = doc[2].get_text()

# Extract main body text
lines = page3_text.splitlines()
# Filter out metadata lines like ID, Page 3 of 3, etc.
body_lines = [l for l in lines if not l.startswith('ID: vpr:') and not l.startswith('Page ') and not l.startswith('VeriPaper')]
raw_text = "\n".join(body_lines)

paragraphs = split_into_smart_paragraphs(raw_text)
sentences = []
for p in paragraphs:
    sentences.extend(segment_sentences(p))

print(f"Total sentences extracted: {len(sentences)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

roberta_path = "models/roberta-sentence-academic-v2"
roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device)
roberta_model.eval()

modernbert_path = "models/modernbert-academic"
modernbert_tok = AutoTokenizer.from_pretrained(modernbert_path)
modernbert_model = AutoModelForSequenceClassification.from_pretrained(modernbert_path).to(device)
modernbert_model.eval()

print("\n--- SENTENCE BY SENTENCE SCORES ---")
roberta_scores = []
modernbert_scores = []
ensemble_scores = []

for idx, sent in enumerate(sentences):
    # RoBERTa prediction
    inputs_r = roberta_tok(sent, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out_r = roberta_model(**inputs_r)
        prob_r = torch.softmax(out_r.logits, dim=-1)[0, 1].item()
    
    # ModernBERT prediction
    inputs_m = modernbert_tok(sent, truncation=True, max_length=8192, return_tensors="pt").to(device)
    with torch.no_grad():
        out_m = modernbert_model(**inputs_m)
        prob_m = torch.softmax(out_m.logits, dim=-1)[0, 1].item()
    
    # Current ensemble formula (65% ModernBERT + 35% RoBERTa)
    ens_score = 0.65 * prob_m + 0.35 * prob_r
    
    roberta_scores.append(prob_r)
    modernbert_scores.append(prob_m)
    ensemble_scores.append(ens_score)
    
    print(f"S#{idx+1:02d} | RoBERTa: {prob_r:.4f} | ModernBERT: {prob_m:.4f} | Ens (65/35): {ens_score:.4f} | {sent[:70]}...")

# Now test smooth_predictions on all 3 sets
preds_r = [{'sentence': s, 'score': sc} for s, sc in zip(sentences, roberta_scores)]
preds_m = [{'sentence': s, 'score': sc} for s, sc in zip(sentences, modernbert_scores)]
preds_e = [{'sentence': s, 'score': sc} for s, sc in zip(sentences, ensemble_scores)]

sm_r = smooth_predictions(preds_r)
sm_m = smooth_predictions(preds_m)
sm_e = smooth_predictions(preds_e)

flag_r_high = sum(1 for p in sm_r if p['score'] >= 0.75)
flag_r_mid  = sum(1 for p in sm_r if 0.60 <= p['score'] < 0.75)

flag_m_high = sum(1 for p in sm_m if p['score'] >= 0.75)
flag_m_mid  = sum(1 for p in sm_m if 0.60 <= p['score'] < 0.75)

flag_e_high = sum(1 for p in sm_e if p['score'] >= 0.75)
flag_e_mid  = sum(1 for p in sm_e if 0.60 <= p['score'] < 0.75)

print("\n--- AGGREGATE SUMMARY (AFTER SMOOTHING & THRESHOLD 0.75 / 0.60) ---")
print(f"RoBERTa Alone        : High={flag_r_high}, Mid={flag_r_mid}, Total Flagged={flag_r_high+flag_r_mid}/{len(sentences)}")
print(f"ModernBERT Alone     : High={flag_m_high}, Mid={flag_m_mid}, Total Flagged={flag_m_high+flag_m_mid}/{len(sentences)}")
print(f"Ensemble (65/35)     : High={flag_e_high}, Mid={flag_e_mid}, Total Flagged={flag_e_high+flag_e_mid}/{len(sentences)}")
