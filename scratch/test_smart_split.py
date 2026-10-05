import fitz
import re
import sys
import os

sys.path.append(os.path.abspath("."))
from src.highlighter import segment_sentences

pdf_path = "d:\\AI_Text_Checker\\LLM_Architectures_and_Training_Survey.pdf"
doc = fitz.open(pdf_path)
raw_text = "\n\n".join([page.get_text("text") for page in doc])

def enhanced_split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    
    # 1. Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Insert explicit paragraph breaks before key structural markers embedded in lines
    text = re.sub(r'(?<=\S)\s*(\b(?:Abstract|Index Terms|1\s+Introduction|Introduction|Background|Related Work|Methodology|Experiments|Results|Discussion|Conclusion|References|Bibliography)\b[\s—\-\.\:]+)', r'\n\n\1', text, flags=re.IGNORECASE)
    
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    paragraphs = []
    
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) <= 1:
            paragraphs.append(block)
            continue
            
        current_chunk = []
        for line in lines:
            # Check for section headings, title numbers, affiliations, dates, emails, keywords, and citations
            is_heading = bool(re.match(
                r'^(?:[0-9]+\.|\d+\s+[A-Z]|\b(?:abstract|introduction|background|related work|methodology|experiments|results|discussion|conclusion|references|works cited|bibliography|department|school|university|e-mail|email|keywords|author|authors|independent research|july \d{4}|june \d{4}|august \d{4})\b)', 
                line, 
                re.IGNORECASE
            ))
            is_bracket_ref = bool(re.match(r'^\[\d+\]', line))
            is_author_ref = bool(re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.?(?:\s+&\s+[A-Z][a-z]+,?\s+[A-Z]\.?)?\s+\(\d{4}\)', line))
            is_bullet = bool(re.match(r'^(?:[\*\-\•]|\d+\.)\s+', line))
            
            # Treat short standalone header/metadata lines at the beginning of a block as separate
            is_short_header = (len(line) < 70 and not line.endswith("."))
            
            if (is_heading or is_bracket_ref or is_author_ref or is_bullet or is_short_header) and current_chunk:
                paragraphs.append(" ".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            paragraphs.append(" ".join(current_chunk))
            
    return [p for p in paragraphs if p.strip()]

paragraphs = enhanced_split_paragraphs(raw_text)
print(f"[Smart Split] Total Paragraphs extracted: {len(paragraphs)}")
print("\n[First 5 Paragraphs]:")
for i, p in enumerate(paragraphs[:5]):
    print(f"--- Paragraph {i+1} ---")
    print(p)
