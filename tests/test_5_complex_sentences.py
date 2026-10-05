from robustness_eval.humanizer.humanizer_engine import humanize_text

test_sentences = [
    ("1. Academic / Hydrology", "The SCS-CN method, introduced by the Soil Conservation Service of the United States in 1969, is a widely used conceptual hydrological model for estimating direct runoff generated from daily rainfall depth."),
    ("2. Microservices Architecture", "Engineered a high-throughput Python microservice enabling real-time stream ingestion, distributed feature transformation, and low-latency machine learning inference with under 150ms P99 SLA."),
    ("3. Autonomous ETL Pipeline", "Architected an autonomous 3-agent ETL pipeline performing multi-document extraction, automated schema mapping, and 3-way matching across 1,000+ monthly unstructured invoices with a 98.5% accuracy rate."),
    ("4. NLP & Machine Learning", "Designed an end-to-end NLP evaluation framework featuring streamlined text preprocessing, embedding extraction, and sentence-level model diagnostics using fine-tuned RoBERTa transformers achieving 94.2% validation accuracy."),
    ("5. AI Alignment & Ethics", "Furthermore, it is crucial to delve into the multifaceted challenges of artificial intelligence alignment to foster responsible development and mitigate potential existential risks.")
]

print("=== BENCHMARKING 5 COMPLEX TEST SENTENCES ===")
for category, text in test_sentences:
    res = humanize_text(text, tone="resume")
    print(f"\n[{category}]")
    print(f"INPUT : {res['original_text']}")
    print(f"OUTPUT: {res['humanized_text']}")
    print(f"SCORE : {res['original_ai_percentage']}% -> {res['humanized_ai_percentage']}% AI ({res['resilience_verdict']})")
