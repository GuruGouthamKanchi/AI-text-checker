import requests

url = 'http://localhost:8000/analyze'
files = {'file': open('D:/AI_Text_Checker/old_only_roBERTa.pdf', 'rb')}
r = requests.post(url, files=files)
data = r.json()

print("Status code:", r.status_code)
print("Overall AI Percentage:", data.get("overall_ai_percentage"), "%")
print("Summary Breakdown:", data.get("summary"))

sentences = data.get("sentences", [])
print(f"Total Sentences: {len(sentences)}")
flagged = [s for s in sentences if s["confidence_tier"] in ["high", "medium"]]
print(f"Flagged Sentences Count: {len(flagged)}")
for s in flagged:
    print(f"[{s['confidence_tier'].upper()}] Score={s['ai_probability']:.4f}: {repr(s['text'][:80])}")
