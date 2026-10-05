from robustness_eval.humanizer import humanize_text

test_samples = [
    "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features.",
    "Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development.",
    "Designed an end-to-end NLP data pipeline featuring streamlined text preprocessing, feature extraction, and sentence-level model evaluation using fine-tuned DistilBERT and RoBERTa models achieving 94.2% validation accuracy.",
    "Engineered an autonomous 3-agent ETL pipeline performing extraction, transformation, and 3-way matching across 1,000+ monthly unstructured invoices, POs, and receipt datasets with a 98.5% accuracy rate."
]

print("=== REFACTORED PURE NEURAL ENGINE BENCHMARK ===")
for i, s in enumerate(test_samples):
    res = humanize_text(s, tone="resume")
    print(f"\n[{i+1}] INPUT : {s}")
    print(f"    OUTPUT: {res['humanized_text']}")
    print(f"    AI    : {res['original_ai_percentage']}% -> {res['humanized_ai_percentage']}% AI  ({res['resilience_verdict']})")
