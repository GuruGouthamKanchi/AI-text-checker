import fitz
import re
import requests

BASE_URL = "http://localhost:8000"
pdf_path = "d:\\AI_Text_Checker\\LLM_Architectures_and_Training_Survey.pdf"

print(f"[Test] Uploading '{pdf_path}' to /analyze...")
with open(pdf_path, "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/analyze",
        files={"file": ("LLM_Architectures_and_Training_Survey.pdf", f, "application/pdf")}
    )

print(f"Analyze Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    sentences = data.get("sentences", [])
    high_sent = [s for s in sentences if s.get("confidence_tier") == "high"]
    med_sent = [s for s in sentences if s.get("confidence_tier") == "medium"]
    
    print(f"\n[Web App Breakdown]")
    print(f"Total Sentences: {len(sentences)}")
    print(f"High Confidence AI Sentences ({len(high_sent)}):")
    for s in high_sent:
        print(f"  - [{s.get('ai_probability')*100:.1f}%] {s.get('text')[:100]}")
        
    print(f"\nMedium Confidence AI Sentences ({len(med_sent)}):")
    for s in med_sent:
        print(f"  - [{s.get('ai_probability')*100:.1f}%] {s.get('text')[:100]}")
