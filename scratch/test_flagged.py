import requests

url = 'http://localhost:8000/analyze'
files = {'file': open('LLM_Architectures_and_Training_Survey.pdf', 'rb')}
r = requests.post(url, files=files)
data = r.json()
sents = data['sentences']
for idx, s in enumerate(sents):
    score = s['ai_probability']
    tier = s['confidence_tier']
    if tier in ['high', 'medium'] or score >= 0.60:
        print(f"Sentence #{idx+1} [P{s.get('paragraph_index')}] ({tier}, score={score:.4f}):\n  {repr(s['text'])}\n")
