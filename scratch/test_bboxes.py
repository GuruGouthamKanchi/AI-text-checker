import requests
from src.report_generator import align_sentences_to_bboxes

pdf_path = 'd:/AI_Text_Checker/LLM_Architectures_and_Training_Survey.pdf'

url = 'http://localhost:8000/analyze'
files = {'file': open(pdf_path, 'rb')}
r = requests.post(url, files=files)
data = r.json()
proc_sents = data['sentences']

flagged = [s for s in proc_sents if s['confidence_tier'] in ['high', 'medium']]
print(f"Flagged count: {len(flagged)}")

sentences_list = [s.get("text", "") for s in flagged]
aligned = align_sentences_to_bboxes(pdf_path, sentences_list)
print(f"Aligned count: {len(aligned)}")

for a in aligned:
    print(f"Sentence: {repr(a['sentence'][:60])}")
    print(f"  Pages: {list(a['pages_bboxes'].keys())}")
    for p, bboxes in a['pages_bboxes'].items():
        print(f"    Page {p}: {len(bboxes)} bboxes")
