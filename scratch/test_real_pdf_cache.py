import requests
import os

BASE_URL = "http://localhost:8000"
pdf_path = "d:\\AI_Text_Checker\\EJ1172284.pdf"

if os.path.exists(pdf_path):
    print(f"[Test 1] Uploading PDF manuscript '{os.path.basename(pdf_path)}' to /analyze endpoint...")
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/analyze",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
        )

    print(f"[Analyze Endpoint Response] Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"[Analyze Endpoint Response] Sentences processed: {len(data.get('sentences', []))}")
        print(f"[Analyze Endpoint Response] Overall AI Score: {data.get('overall_percentage')}%")
    else:
        print(f"[Analyze Error] {resp.text}")

    print("\n[Test 2] Requesting Forensic PDF Report via /report?sample_type=human endpoint...")
    pdf_resp = requests.post(
        f"{BASE_URL}/report?sample_type=human"
    )

    print(f"[Report Endpoint Response] Status: {pdf_resp.status_code}")
    print(f"[Report Endpoint Response] Content-Type: {pdf_resp.headers.get('content-type')}")
    print(f"[Report Endpoint Response] PDF Size: {len(pdf_resp.content)} bytes")

    if pdf_resp.status_code == 200 and pdf_resp.content.startswith(b"%PDF"):
        print("\nVerification SUCCESSFUL: Forensic PDF Report generated cleanly!")
    else:
        print(f"\nVerification FAILED: Status {pdf_resp.status_code}, Body: {pdf_resp.text[:200]}")
else:
    print(f"Sample PDF not found at {pdf_path}")
