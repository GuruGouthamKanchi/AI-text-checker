import requests
import json

url = "http://127.0.0.1:8000/robustness/humanize/stream"
payload = {
    "text": "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features. Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development.",
    "tone": "resume"
}

print("=== TESTING REAL-TIME SSE STREAMING ENDPOINT ===")
print("Sending request to:", url)
print()

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print("STREAM CHUNK:", decoded)
