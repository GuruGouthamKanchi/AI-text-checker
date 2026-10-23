import urllib.request
import json

queries = [
    'https://api.crossref.org/works?query=AI-Generated+Text+Detection+Stylometric+Transformer&filter=from-pub-date:2024-01-01,prefix:10.1109&rows=25',
    'https://api.crossref.org/works?query=Machine-Generated+Text+Detection+Survey+Benchmark&filter=from-pub-date:2024-01-01,prefix:10.1109&rows=25',
    'https://api.crossref.org/works?query=AI+Text+Detection+Paraphrasing+Adversarial&filter=from-pub-date:2024-01-01,prefix:10.1109&rows=25',
    'https://api.crossref.org/works?query=IEEE+Transactions+AI-Generated+Text+Detection&filter=from-pub-date:2024-01-01,prefix:10.1109&rows=25'
]

seen_dois = set()
results = []

for q in queries:
    req = urllib.request.Request(q, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            for item in data.get('message', {}).get('items', []):
                doi = item.get('DOI', '').lower()
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)
                title = item.get('title', [''])[0]
                year_parts = item.get('issued', {}).get('date-parts', [[0]])[0]
                year = year_parts[0] if year_parts else 0
                if year < 2024:
                    continue
                container = item.get('container-title', [''])[0]
                authors = []
                for a in item.get('author', []):
                    given = a.get('given', '')
                    family = a.get('family', '')
                    if given and family:
                        authors.append(f"{given[0]}. {family}")
                    elif family:
                        authors.append(family)
                page = item.get('page', '')
                volume = item.get('volume', '')
                results.append({
                    'doi': doi,
                    'title': title,
                    'journal': container,
                    'year': year,
                    'authors': authors,
                    'page': page,
                    'volume': volume
                })
    except Exception as e:
        print('Query error:', e)

print(f"Total retrieved >= 2024: {len(results)}")
with open('scratch/found_2024_plus_candidates.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

for i, r in enumerate(results[:25]):
    print(f"[{i+1}] {r['year']} | {r['doi']} | {r['journal'][:35]} | {r['title']}")
