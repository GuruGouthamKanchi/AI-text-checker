import requests

url = 'http://localhost:8000/analyze'
files = {'file': open('D:/AI_Text_Checker/old_only_roBERTa.pdf', 'rb')}
r = requests.post(url, files=files)
data = r.json()

print("================ ADVANCED MULTI-LAYER ENSEMBLE BENCHMARK ================")
print("Status code:", r.status_code)
print("Overall AI Percentage:", data.get("overall_ai_percentage"), "%")
print("Summary Breakdown:", data.get("summary"))

sentences = data.get("sentences", [])
print(f"Total Sentences: {len(sentences)}")
flagged_high = [s for s in sentences if s["confidence_tier"] == "high"]
flagged_mid = [s for s in sentences if s["confidence_tier"] == "medium"]
print(f"High Risk Sentences: {len(flagged_high)}")
print(f"Medium Risk Sentences: {len(flagged_mid)}")

print("\n--- SAMPLE HIGH RISK SENTENCES ---")
for s in flagged_high[:5]:
    print(f"[HIGH] Score={s['ai_probability']:.4f}: {s['text'][:80]}...")
