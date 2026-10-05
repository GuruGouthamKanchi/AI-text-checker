import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.app import run_analysis_pipeline

model_path = "models/t5-humanizer-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

test_samples = [
    "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features.",
    "Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development.",
    "Designed an end-to-end NLP data pipeline featuring streamlined text preprocessing, feature extraction, and sentence-level model evaluation using fine-tuned DistilBERT and RoBERTa models achieving 94.2% validation accuracy.",
    "SCS-CN Method The SCS-CN method, introduced by the Soil Conservation Service of the United States in 1969, is a widely used conceptual hydrological model for estimating direct runoff generated from daily rainfall depth."
]

print("=== TESTING PURE NEURAL T5 HUMANIZER (BYPASSING ALL RULE REWRITERS) ===")
for i, raw in enumerate(test_samples):
    prompt = f"humanize: {raw}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    humanized = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Rescore with RoBERTa detector
    res = run_analysis_pipeline(doc_text=humanized, filename=f"neural_{i}.txt", page_count=1)
    score = res["overall_ai_percentage"]
    
    print(f"\n[{i+1}] RAW INPUT : {raw}")
    print(f"    PURE T5   : {humanized}")
    print(f"    AI SCORE  : {score}%")
