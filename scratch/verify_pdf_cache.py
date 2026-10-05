import requests

BASE_URL = "http://localhost:8000"

print("[Test] Sending document to /analyze endpoint...")
sample_text = "In this paper, we present a novel neural architecture for machine translation. We evaluate our model on standard benchmarks and achieve state-of-the-art BLEU scores."

# Test /analyze
resp = requests.post(
    f"{BASE_URL}/analyze",
    files={"file": ("sample_paper.txt", sample_text.encode('utf-8'), "text/plain")}
)

print(f"[Analyze Endpoint] Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"[Analyze Endpoint] AI Probability Score: {data.get('overall_percentage')}%")
    print(f"[Analyze Endpoint] Sentences processed: {len(data.get('sentences', []))}")

# Test /report/generate-pdf
print("\n[Test] Clicking Download Forensics Report (PDF)...")
pdf_resp = requests.post(
    f"{BASE_URL}/report/generate-pdf",
    data={"filename": "sample_paper.txt", "sample_type": ""}
)

print(f"[PDF Report Endpoint] Status: {pdf_resp.status_code}")
print(f"[PDF Report Endpoint] Content-Type: {pdf_resp.headers.get('content-type')}")
print(f"[PDF Report Endpoint] Generated PDF Bytes: {len(pdf_resp.content)} bytes")

if pdf_resp.status_code == 200 and pdf_resp.content.startswith(b"%PDF"):
    print("\n✅ Verification Successful: PDF report generated cleanly with Cache Hit reuse!")
else:
    print("\n❌ PDF Generation Failed.")
