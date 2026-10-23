import urllib.request
import urllib.parse
import json

prefixes = {
    'IEEE': ('10.1109', 'AI-generated text detection'),
    'ACM': ('10.1145', 'machine generated text detection'),
    'Elsevier': ('10.1016', 'ChatGPT text detection academic integrity'),
    'Springer Nature': ('10.1007', 'AI generated text detection higher education'),
    'Wiley': ('10.1002', 'stylometric text classification NLP'),
    'MDPI': ('10.3390', 'detecting AI generated text transformer'),
    'Taylor & Francis': ('10.1080', 'AI text detection academic integrity plagiarism'),
    'SAGE': ('10.1177', 'ChatGPT academic integrity artificial intelligence')
}

results = {}
for pub, (prefix, q) in prefixes.items():
    url = f"https://api.crossref.org/prefixes/{prefix}/works?query={urllib.parse.quote(q)}&rows=4&sort=relevance"
    req = urllib.request.Request(url, headers={'User-Agent': 'VeriPaperRefSearch/1.0 (mailto:scholar@veripaper.org)'})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            results[pub] = []
            for it in items:
                authors = []
                for a in it.get('author', []):
                    given = a.get('given', '')
                    family = a.get('family', '')
                    if family:
                        authors.append(f"{given} {family}".strip())
                year = 2023
                if 'published-print' in it and 'date-parts' in it['published-print']:
                    year = it['published-print']['date-parts'][0][0]
                elif 'published-online' in it and 'date-parts' in it['published-online']:
                    year = it['published-online']['date-parts'][0][0]
                elif 'created' in it and 'date-parts' in it['created']:
                    year = it['created']['date-parts'][0][0]
                    
                results[pub].append({
                    'doi': it.get('DOI', ''),
                    'title': it.get('title', [''])[0],
                    'container': it.get('container-title', [''])[0],
                    'year': year,
                    'authors': authors[:4],
                    'volume': it.get('volume', ''),
                    'issue': it.get('issue', ''),
                    'page': it.get('page', '')
                })
    except Exception as e:
        print(f"Error for {pub}: {e}")

with open("scratch/verified_references.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved verified_references.json successfully!")
