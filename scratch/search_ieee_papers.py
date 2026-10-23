import urllib.request
import urllib.parse
import json

queries = [
    "AI-generated text detection",
    "ChatGPT detection",
    "machine-generated text detection",
    "deepfake text detection",
    "adversarial paraphrase detection",
    "stylometric text detection",
    "text watermarking detection",
    "Transformer text classification academic",
    "large language model text detection",
    "detecting ChatGPT academic text"
]

ieee_candidates = []

for q in queries:
    url = f"https://api.crossref.org/prefixes/10.1109/works?query={urllib.parse.quote(q)}&rows=10&sort=relevance"
    req = urllib.request.Request(url, headers={'User-Agent': 'VeriPaperIEEESearch/1.0 (mailto:scholar@veripaper.org)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            for it in items:
                title = it.get('title', [''])[0]
                doi = it.get('DOI', '')
                container = it.get('container-title', [''])[0]
                
                # Filter out obvious non-text or irrelevant items
                title_lower = title.lower()
                if not any(k in title_lower for k in ['text', 'chatgpt', 'language', 'nlp', 'stylometric', 'deepfake', 'academic', 'paraphrase', 'transformer']):
                    continue
                    
                authors = []
                for a in it.get('author', []):
                    given = a.get('given', '')
                    family = a.get('family', '')
                    if family:
                        initials = " ".join([p[0] + "." for p in given.split() if p]) if given else ""
                        authors.append(f"{initials} {family}".strip())
                        
                year = 2023
                for k in ['published-print', 'published-online', 'created']:
                    if k in it and 'date-parts' in it[k] and it[k]['date-parts'][0][0]:
                        year = it[k]['date-parts'][0][0]
                        break
                        
                ieee_candidates.append({
                    'doi': doi,
                    'title': title,
                    'journal': container,
                    'year': year,
                    'authors': authors,
                    'volume': it.get('volume', ''),
                    'issue': it.get('issue', ''),
                    'page': it.get('page', ''),
                    'type': it.get('type', '')
                })
    except Exception as e:
        print(f"Error for query '{q}': {e}")

# Deduplicate by DOI
seen_dois = set()
unique_ieee = []
for p in ieee_candidates:
    if p['doi'].lower() not in seen_dois:
        seen_dois.add(p['doi'].lower())
        unique_ieee.append(p)

with open("scratch/ieee_deep_search.json", "w", encoding="utf-8") as f:
    json.dump(unique_ieee, f, indent=2)

print(f"Discovered {len(unique_ieee)} unique IEEE papers on AI text detection & NLP!")
