import urllib.request
import urllib.parse
import json

journals = [
    "IEEE Transactions on Information Forensics and Security",
    "IEEE Transactions on Dependable and Secure Computing",
    "IEEE Transactions on Neural Networks and Learning Systems",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "IEEE Transactions on Knowledge and Data Engineering",
    "IEEE/ACM Transactions on Audio, Speech, and Language Processing"
]

results = []

for j in journals:
    q = f"{j} text detection adversarial"
    url = f"https://api.crossref.org/prefixes/10.1109/works?query={urllib.parse.quote(q)}&rows=5"
    req = urllib.request.Request(url, headers={'User-Agent': 'VeriPaperCurator/1.0 (mailto:scholar@veripaper.org)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            for it in items:
                container = it.get('container-title', [''])[0]
                title = it.get('title', [''])[0]
                doi = it.get('DOI', '')
                title_lower = title.lower()
                if any(w in title_lower for w in ['text', 'adversarial', 'detection', 'transformer', 'deepfake', 'nlp', 'language']):
                    authors = []
                    for a in it.get('author', []):
                        g = a.get('given', '')
                        f = a.get('family', '')
                        if f:
                            ini = " ".join([p[0] + "." for p in g.split() if p]) if g else ""
                            authors.append(f"{ini} {f}".strip())
                    results.append({
                        'journal': container,
                        'title': title,
                        'doi': doi,
                        'authors': authors[:4],
                        'year': it.get('created', {}).get('date-parts', [[0]])[0][0]
                    })
    except Exception as e:
        print(f"Error {j}: {e}")

with open("scratch/ieee_transactions_candidates.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Found {len(results)} IEEE Transactions papers!")
for p in results[:10]:
    print(f"- {p['title']} [{p['journal']}] DOI: {p['doi']}")
