import fitz
from src.document_parser import extract_text_from_pdf
from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, run_predictions, smooth_predictions
from src.report_generator import align_sentences_to_bboxes, render_highlighted_pdf_page_images, clean_text_for_alignment

pdf_path = 'd:/AI_Text_Checker/LLM_Architectures_and_Training_Survey.pdf'
doc_text, pdf_pages = extract_text_from_pdf(pdf_path)
raw_paragraphs = split_into_smart_paragraphs(doc_text)
sentences = []
for p in raw_paragraphs:
    sentences.extend(segment_sentences(p))

preds, _ = run_predictions(sentences=sentences, model_dir='models/modernbert-academic', roberta_dir='models/roberta-sentence-academic-v2')
preds = smooth_predictions(preds)

proc_sents = []
for i, p in enumerate(preds):
    tier = 'high' if p['score'] >= 0.75 else ('medium' if p['score'] >= 0.60 else 'unflagged')
    proc_sents.append({'sentence': p['sentence'], 'confidence_tier': tier, 'score': p['score']})

flagged = [s for s in proc_sents if s['confidence_tier'] in ['high', 'medium']]
print(f'Total sentences: {len(proc_sents)}')
print(f'Flagged count: {len(flagged)}')
for f in flagged:
    print(f"TIER: {f['confidence_tier']}, SCORE: {f['score']:.4f}")
    print(f"SENTENCE: {repr(f['sentence'])}\n")

aligned = align_sentences_to_bboxes(pdf_path, [f['sentence'] for f in flagged])
print(f'Aligned results count: {len(aligned)}')
for a in aligned:
    print(f"Sentence: {repr(a['sentence'][:60])}")
    print(f"  Pages: {list(a['pages_bboxes'].keys())}")
    for p, bboxes in a['pages_bboxes'].items():
        print(f"    Page {p}: {len(bboxes)} bboxes")

hl_images = render_highlighted_pdf_page_images(pdf_path, proc_sents)
print(f'Rendered highlighted images: {len(hl_images)} pages')
