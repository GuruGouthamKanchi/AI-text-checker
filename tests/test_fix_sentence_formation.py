from robustness_eval.humanizer import humanize_text

sample = "Engineered an autonomous 3-agent ETL pipeline performing extraction, transformation, and 3-way matching across 1,000+ monthly unstructured invoices, POs, and receipt datasets with a 98.5% accuracy rate."

print("=== INPUT ===")
print(sample)
print()

res = humanize_text(sample, tone="resume")

print("=== HUMANIZED OUTPUT ===")
print(res["humanized_text"])
print()
print(f"AI Score: {res['humanized_ai_percentage']}%  ({res['resilience_verdict']})")
