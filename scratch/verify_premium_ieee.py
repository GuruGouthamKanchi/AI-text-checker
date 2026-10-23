import urllib.request
import json

dois = [
    '10.1109/tdsc.2025.3624268',
    '10.1109/tnnls.2026.3727304',
    '10.1109/tifs.2022.3155975',
    '10.1109/qpain69676.2026.11545549',
    '10.1109/compsac65507.2025.00094',
    '10.1109/icitsif69060.2026.11609156',
    '10.1109/ispras64596.2024.10899150',
    '10.1109/ACCESS.2023.3294090'
]

details = []
for doi in dois:
    url = 'https://api.crossref.org/works/' + doi
    req = urllib.request.Request(url, headers={'User-Agent': 'VeriPaperCurator/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read().decode())['message']
            authors = []
            for a in d.get('author', []):
                given = a.get('given', '')
                family = a.get('family', '')
                if family:
                    ini = " ".join([p[0] + "." for p in given.split() if p]) if given else ""
                    authors.append(f"{ini} {family}".strip())
                    
            year = 2023
            for k in ['published-print', 'published-online', 'created']:
                if k in d and 'date-parts' in d[k] and d[k]['date-parts'][0][0]:
                    year = d[k]['date-parts'][0][0]
                    break
                    
            details.append({
                'doi': doi,
                'title': d.get('title', [''])[0],
                'journal': d.get('container-title', [''])[0],
                'authors': authors,
                'year': year,
                'volume': d.get('volume', ''),
                'issue': d.get('issue', ''),
                'page': d.get('page', '')
            })
    except Exception as e:
        print(f"Error {doi}: {e}")

with open("scratch/ieee_premium_verified.json", "w", encoding="utf-8") as f:
    json.dump(details, f, indent=2)

print("Saved ieee_premium_verified.json successfully!")
for it in details:
    print(f"- [{it['doi']}] {it['title']} ({it['journal']})")
