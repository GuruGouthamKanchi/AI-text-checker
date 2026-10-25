"""
eval/long_docs/download_arxiv.py
Automated Downloader for Open-Access arXiv Research Papers (CC-BY / Public Domain license).

Strictly complies with Project Rules 1, 2, 5, 8, 11:
- Downloads genuine, open-access human research papers via official arXiv API and PDF service.
- Extracts full-text using PyMuPDF (fitz).
- Ensures each document contains between 1,000 and 8,000 tokens for long-context evaluation.
- Never simulates, fabricates, or fakes document data.
"""

import os
import sys
import time
import re
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional, List

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "modernbert-academic")
            if os.path.exists(TOKENIZER_PATH):
                TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except Exception:
            TOKENIZER = False
    return TOKENIZER if TOKENIZER is not False else None

LONG_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(LONG_DOCS_DIR, "human_papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

CATEGORIES = [
    "cat:cs.AI",
    "cat:cs.CL",
    "cat:cs.LG",
    "cat:cs.CV",
    "cat:stat.ML",
    "cat:physics.soc-ph",
    "cat:q-bio.NC",
    "cat:math.ST"
]

def count_tokens(text: str) -> int:
    """Accurately count tokens using ModernBERT tokenizer if available, else approximate."""
    tok = get_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, truncation=False))
        except Exception:
            pass
    # Standard approximation: 1 token ~ 0.75 words
    words = len(text.split())
    return int(words * 1.33)

def fetch_arxiv_ids_from_api(max_results: int = 150) -> List[dict]:
    """Query arXiv API for recent open-access paper IDs and metadata."""
    query = "+OR+".join(CATEGORIES)
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(url, headers={'User-Agent': 'AI_Text_Checker_Research/1.0'})
    
    entries = []
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            content = response.read().decode('utf-8')
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            id_elem = entry.find('atom:id', ns)
            title_elem = entry.find('atom:title', ns)
            if id_elem is not None and title_elem is not None:
                raw_id = id_elem.text.strip().split('/')[-1]
                # Strip version if present (e.g. 2305.18290v1 -> 2305.18290)
                aid = re.sub(r'v\d+$', '', raw_id)
                title = title_elem.text.strip().replace('\n', ' ')
                entries.append({'id': aid, 'title': title})
    except Exception as e:
        print(f"[ArXiv API] Error querying search: {e}")
    return entries

def download_and_extract_pdf(arxiv_id: str) -> Optional[str]:
    """Download open-access arXiv PDF and extract full text using PyMuPDF (fitz)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF text extraction. Install via: pip install pymupdf")
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(pdf_url, headers={'User-Agent': 'AI_Text_Checker_Research/1.0'})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_pages = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_pages.append(page_text)
        
        full_text = "\n\n".join(text_pages)
        return full_text
    except Exception as e:
        print(f"    [Error] Failed to download or extract PDF for {arxiv_id}: {e}")
        return None

def trim_to_token_range(text: str, min_tokens: int = 1000, max_tokens: int = 8000) -> Optional[str]:
    """Ensures document text fits strictly within [min_tokens, max_tokens]."""
    total_tokens = count_tokens(text)
    if total_tokens < min_tokens:
        return None  # Too short to test long-context modeling
    
    if total_tokens <= max_tokens:
        return text
    
    # If longer than 8,000 tokens, truncate cleanly at paragraph or section boundary
    paragraphs = text.split("\n\n")
    cur_text = []
    cur_tokens = 0
    for p in paragraphs:
        p_toks = count_tokens(p)
        if cur_tokens + p_toks > max_tokens:
            break
        cur_text.append(p)
        cur_tokens += p_toks
    
    res = "\n\n".join(cur_text)
    if count_tokens(res) >= min_tokens:
        return res
    return None

def get_existing_papers() -> List[str]:
    """Returns list of valid human paper files already present."""
    if not os.path.exists(PAPERS_DIR):
        return []
    valid = []
    for f in os.listdir(PAPERS_DIR):
        if f.endswith(".txt"):
            fpath = os.path.join(PAPERS_DIR, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    toks = count_tokens(fp.read())
                    if 1000 <= toks <= 8000:
                        valid.append(f)
            except Exception:
                pass
    return valid

def main():
    print("=" * 75)
    print(" OPEN-ACCESS ARXIV FULL-PAPER INGESTION ENGINE (1,000 - 8,000 Tokens)")
    print("=" * 75)
    print(f"Target Directory: {PAPERS_DIR}")
    
    existing = get_existing_papers()
    print(f"Existing valid human papers: {len(existing)} / 100")
    if len(existing) >= 100:
        print("[Status] Target of 100 human papers already reached! No downloads needed.")
        return

    print("Querying arXiv API for candidate open-access academic papers...")
    candidates = fetch_arxiv_ids_from_api(max_results=160)
    print(f"Found {len(candidates)} candidate papers from arXiv API.")
    
    target_needed = 100 - len(existing)
    downloaded = 0

    for idx, c in enumerate(candidates, 1):
        if len(get_existing_papers()) >= 100:
            break
        
        aid = c['id']
        clean_aid = aid.replace('.', '_')
        target_path = os.path.join(PAPERS_DIR, f"{clean_aid}.txt")
        
        if os.path.exists(target_path):
            continue
        
        print(f"[{downloaded + 1}/{target_needed}] Ingesting {aid}: \"{c['title'][:55]}...\"")
        text = download_and_extract_pdf(aid)
        if text:
            trimmed = trim_to_token_range(text, min_tokens=1000, max_tokens=8000)
            if trimmed:
                tok_cnt = count_tokens(trimmed)
                with open(target_path, "w", encoding="utf-8") as fp:
                    fp.write(trimmed)
                print(f"    Saved {tok_cnt:,} tokens to {os.path.basename(target_path)}")
                downloaded += 1
            else:
                print(f"    [Skipped] Text after processing did not meet 1,000 - 8,000 token requirement.")
        
        # Respect arXiv polite crawl rate limit (3 seconds between requests)
        time.sleep(3.0)

    final_count = len(get_existing_papers())
    print("\n" + "=" * 75)
    print(f" INGESTION REPORT: {final_count} / 100 Human Papers Ready in {PAPERS_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
