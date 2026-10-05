import fitz
import re
from src.report_generator import clean_text_for_alignment

def align_sentences_to_bboxes_v2(pdf_path: str, sentences: list[str]) -> list[dict]:
    doc = fitz.open(pdf_path)
    words_list = []
    full_text_buffer = []
    char_to_word_idx = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        words = page.get_text("words")
        words = sorted(words, key=lambda w: (w[1], w[0]))
        
        for w in words:
            word_text = w[4]
            words_list.append({
                "text": word_text,
                "bbox": (w[0], w[1], w[2], w[3]),
                "page": page_idx
            })
            
            full_text_buffer.append(word_text)
            full_text_buffer.append(" ")
            word_len = len(word_text)
            for _ in range(word_len + 1):
                char_to_word_idx.append(len(words_list) - 1)
                
    full_doc_text = "".join(full_text_buffer)
    clean_doc = clean_text_for_alignment(full_doc_text)
    
    clean_to_orig_idx = []
    orig_pos = 0
    for char in full_doc_text:
        char_clean = clean_text_for_alignment(char)
        if char_clean:
            clean_to_orig_idx.append(orig_pos)
        orig_pos += 1
        
    aligned_sentences = []
    search_start_pos = 0
    
    for sent in sentences:
        # Split sentence into sub-parts by newlines or major punctuation to handle multi-line/title blocks
        sub_parts = [p.strip() for p in re.split(r'[\n\r]+', sent) if p.strip()]
        if not sub_parts:
            sub_parts = [sent]

        bboxes_by_page = {}

        for part in sub_parts:
            part_clean = clean_text_for_alignment(part)
            if len(part_clean) < 3: # Skip trivial tokens like "1"
                continue
                
            match_idx = clean_doc.find(part_clean, search_start_pos)
            if match_idx == -1:
                match_idx = clean_doc.find(part_clean, 0)
                
            if match_idx != -1:
                try:
                    orig_start = clean_to_orig_idx[match_idx]
                    orig_end = clean_to_orig_idx[match_idx + len(part_clean) - 1]
                    
                    start_word_idx = char_to_word_idx[orig_start]
                    end_word_idx = char_to_word_idx[orig_end]
                    
                    for w_idx in range(start_word_idx, end_word_idx + 1):
                        word_info = words_list[w_idx]
                        p = word_info["page"]
                        if p not in bboxes_by_page:
                            bboxes_by_page[p] = []
                        bboxes_by_page[p].append(word_info["bbox"])
                        
                    search_start_pos = match_idx + len(part_clean)
                except Exception as e:
                    pass

        if bboxes_by_page:
            aligned_sentences.append({
                "sentence": sent,
                "pages_bboxes": bboxes_by_page
            })
                
    doc.close()
    return aligned_sentences

pdf_path = 'd:/AI_Text_Checker/LLM_Architectures_and_Training_Survey.pdf'
from src.document_parser import extract_text_from_pdf
from src.app import split_into_smart_paragraphs
from src.highlighter import segment_sentences, run_predictions, smooth_predictions

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
aligned = align_sentences_to_bboxes_v2(pdf_path, [f['sentence'] for f in flagged])

print(f"Aligned results count: {len(aligned)}")
for a in aligned:
    print(f"Sentence: {repr(a['sentence'][:60])}")
    print(f"  Pages: {list(a['pages_bboxes'].keys())}")
    for p, bboxes in a['pages_bboxes'].items():
        print(f"    Page {p}: {len(bboxes)} bboxes")
