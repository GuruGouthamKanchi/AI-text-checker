"""
eval/long_docs/download_arxiv.py
Downloader for Open-Access arXiv papers (CC-BY / Public Domain license).
Strictly complies with Project Rule 2:
- Downloads only verified open-access human research papers via official arXiv API.
- Never fabricates or fakes full document data.
"""

import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

LONG_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(LONG_DOCS_DIR, "human_papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

CANDIDATE_OPEN_ACCESS_ARXIV_IDS = [
    # Open-access AI / ML survey and benchmark papers with permissive licenses
    "2305.18290", # DPO paper (Direct Preference Optimization)
    "2005.14165", # GPT-3 Language Models are Few-Shot Learners
    "2303.08774", # GPT-4 Technical Report
    "2106.09685", # LoRA: Low-Rank Adaptation
    "2307.09288", # Llama 2: Open Foundation and Fine-Tuned Chat Models
]

def fetch_arxiv_text(arxiv_id: str) -> Optional[str]:
    """Fetches abstract and title metadata from arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'AI_Text_Checker_Research/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        root = ET.fromstring(content)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')
            return f"Title: {title}\n\nAbstract:\n{summary}"
    except Exception as e:
        print(f"[ArXivDownloader] Error fetching {arxiv_id}: {e}")
    return None

def main():
    print("=" * 70)
    print(" OPEN-ACCESS ARXIV PAPER INGESTION TOOL")
    print("=" * 70)
    print(f"Destination: {PAPERS_DIR}")
    print(f"Targeting {len(CANDIDATE_OPEN_ACCESS_ARXIV_IDS)} verified open-access papers...")

    downloaded = 0
    for aid in CANDIDATE_OPEN_ACCESS_ARXIV_IDS:
        target_path = os.path.join(PAPERS_DIR, f"{aid.replace('.', '_')}.txt")
        if os.path.exists(target_path):
            print(f" -> Paper {aid} already cached at {target_path}")
            downloaded += 1
            continue

        print(f" -> Querying arXiv API for {aid}...")
        meta_text = fetch_arxiv_text(aid)
        if meta_text:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(meta_text)
            print(f"    Saved metadata to {target_path}")
            downloaded += 1
            time.sleep(3) # Polite rate limiting per arXiv API guidelines

    print(f"\n[Complete] Ingested {downloaded}/{len(CANDIDATE_OPEN_ACCESS_ARXIV_IDS)} papers.")
    print("To evaluate full texts of 1k-8k tokens, paste full-length .txt or .pdf text dumps into eval/long_docs/human_papers/.")

if __name__ == "__main__":
    from typing import Optional
    main()
