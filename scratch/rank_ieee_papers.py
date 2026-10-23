import json

with open("scratch/ieee_deep_search.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

keywords = ['detection', 'chatgpt', 'ai-generated', 'machine-generated', 'adversarial', 'deepfake', 'stylometric', 'watermark', 'perplexity', 'transformer', 'ensemble']

scored = []
for p in papers:
    title = p['title'].lower()
    score = sum(1 for kw in keywords if kw in title)
    journal_lower = p['journal'].lower()
    if 'transactions' in journal_lower:
        score += 4
    elif 'access' in journal_lower:
        score += 3
    elif 'conference' in journal_lower or 'symposium' in journal_lower:
        score += 1
    scored.append((score, p))

scored.sort(key=lambda x: x[0], reverse=True)

with open("scratch/top_ieee_ranked.json", "w", encoding="utf-8") as f:
    json.dump([p for score, p in scored], f, indent=2)

print(f"Top 20 IEEE papers ranked:")
for i, (score, p) in enumerate(scored[:20], 1):
    authors = ", ".join(p['authors'][:3])
    print(f"[{i}] (Score {score}) {p['title']}")
    print(f"    Venue: {p['journal']} ({p['year']}) | DOI: {p['doi']}")
    print(f"    Authors: {authors}")
