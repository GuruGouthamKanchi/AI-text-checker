import os
import re
import hashlib
import random

def detect_citation_style(full_text: str) -> dict:
    """
    Detects which citation style a given research paper uses (IEEE, APA, MLA, Chicago, or Harvard),
    returning detected style, confidence, and in-text marker count.
    """
    # 1. Locate bibliography/references section
    ref_headers = list(re.finditer(r'\b(references|bibliography|works\s+cited|notes)\b', full_text.lower()))
    ref_section = ""
    ref_header_matched = "Unknown"
    
    if ref_headers:
        last_match = ref_headers[-1]
        ref_section = full_text[last_match.start():]
        ref_header_matched = last_match.group(1).lower()

    # Split text into body and reference section to avoid detecting references as in-text markers
    body_text = full_text[:len(full_text) - len(ref_section)] if ref_section else full_text

    # 2. Count in-text citation pattern occurrences in body
    # IEEE: [1], [1, 2], [1-3], [1]–[3]
    ieee_pattern = r'\[\d+(?:\s*(?:,\s*\d+|-\s*\d+|–\s*\d+))*\s*\]'
    ieee_matches = list(re.finditer(ieee_pattern, body_text))
    ieee_count = len(ieee_matches)

    # APA / Harvard: (Smith, 2020), (Jones & Smith, 2019), (Davis et al., 2018)
    apa_harvard_pattern = r'\(\s*[A-Z][A-Za-z]+(?:\s+et\s+al\.)?(?:\s*(?:and|&)\s*[A-Z][A-Za-z]+)?\s*,\s*\d{4}\s*\)'
    apa_harvard_matches = list(re.finditer(apa_harvard_pattern, body_text))
    apa_harvard_count = len(apa_harvard_matches)

    # MLA: (Smith 45), (Jones and Smith 112) - Author Page
    mla_pattern = r'\(\s*[A-Z][A-Za-z]+(?:\s*(?:et\s+al\.|and|&)\s*[A-Z][A-Za-z]+)*\s+\d{1,3}\s*\)'
    mla_matches = list(re.finditer(mla_pattern, body_text))
    mla_count = len(mla_matches)

    # Chicago (footnote style superscript markers, e.g. text^1 text^2 or footnote text matching number)
    chicago_score = 0.0
    if ref_header_matched in ["bibliography", "notes"] and ieee_count <= 1 and apa_harvard_count <= 1 and mla_count <= 1:
        chicago_score = 0.4

    # 3. Reference list start format clues
    has_numbered_refs = False
    if ref_section:
        lines = [line.strip() for line in ref_section.split('\n') if line.strip()]
        numbered_matches = 0
        for line in lines[:8]:
            if re.match(r'^\[\d+\]', line) or re.match(r'^\d+\.', line):
                numbered_matches += 1
        if numbered_matches >= 2:
            has_numbered_refs = True

    # 4. Classify Style based on metrics
    scores = {
        "IEEE": 0.0,
        "APA": 0.0,
        "MLA": 0.0,
        "Harvard": 0.0,
        "Chicago": chicago_score
    }

    # IEEE score calculation (adjusted for short test snippets)
    if ieee_count > 1:
        scores["IEEE"] = 0.8
        if has_numbered_refs:
            scores["IEEE"] = 0.95
    elif has_numbered_refs:
        scores["IEEE"] = 0.9

    # APA / Harvard score calculation
    if apa_harvard_count > 1:
        if ref_section:
            apa_year_in_parens = len(re.findall(r'\b[A-Z][A-Za-z]+,\s*[A-Z]\.\s*\(\d{4}\)', ref_section))
            harvard_year_no_parens = len(re.findall(r'\b[A-Z][A-Za-z]+,\s*[A-Z]\.\s*\d{4}\b', ref_section))
            if apa_year_in_parens >= harvard_year_no_parens:
                scores["APA"] = 0.9
                scores["Harvard"] = 0.7
            else:
                scores["Harvard"] = 0.9
                scores["APA"] = 0.7
        else:
            scores["APA"] = 0.8
            scores["Harvard"] = 0.7

    # MLA score calculation
    if mla_count > 1:
        scores["MLA"] = 0.8
        if ref_header_matched == "works cited":
            scores["MLA"] = 0.95

    # Determine winning style
    best_style = "Unknown"
    best_score = 0.0
    for style, score in scores.items():
        if score > best_score:
            best_score = score
            best_style = style

    if best_score < 0.5:
        best_style = "Unknown"
        best_score = 0.0

    marker_count = 0
    if best_style == "IEEE":
        marker_count = ieee_count
    elif best_style in ["APA", "Harvard"]:
        marker_count = apa_harvard_count
    elif best_style == "MLA":
        marker_count = mla_count
        
    return {
        "style": best_style,
        "confidence": round(best_score, 2),
        "in_text_marker_count": marker_count
    }

def extract_ieee_citations(text: str) -> list:
    """
    Extracts IEEE in-text citation markers like [1], [1, 2], [1-3], [1]–[3].
    """
    citations = []
    matched_spans = []

    # 1. Match bracketed ranges like [3]-[5] or [3]–[5]
    range_pattern = r'\[(\d+)\]\s*(?:-|–)\s*\[(\d+)\]'
    for match in re.finditer(range_pattern, text):
        start_num = int(match.group(1))
        end_num = int(match.group(2))
        start_pos = match.start()
        marker_text = match.group(0)
        matched_spans.append(match.span())
        
        for num in range(start_num, end_num + 1):
            citations.append({
                "marker_text": marker_text,
                "position_in_document": start_pos,
                "referenced_value": str(num)
            })

    # 2. Match individual bracketed blocks like [1, 2, 3] or [1-3] or [1]
    block_pattern = r'\[(\d+(?:\s*(?:,\s*\d+|-\s*\d+|–\s*\d+))*\s*)\]'
    for match in re.finditer(block_pattern, text):
        # Avoid double-matching elements parsed in the range pattern step
        overlap = False
        for span in matched_spans:
            if match.start() >= span[0] and match.end() <= span[1]:
                overlap = True
                break
        if overlap:
            continue
            
        marker_text = match.group(0)
        inner_content = match.group(1)
        start_pos = match.start()
        
        parts = re.split(r'\s*,\s*', inner_content)
        for part in parts:
            range_match = re.split(r'\s*(?:-|–)\s*', part)
            if len(range_match) == 2:
                try:
                    start_num = int(range_match[0])
                    end_num = int(range_match[1])
                    for num in range(start_num, end_num + 1):
                        citations.append({
                            "marker_text": marker_text,
                            "position_in_document": start_pos,
                            "referenced_value": str(num)
                        })
                except ValueError:
                    pass
            else:
                citations.append({
                    "marker_text": marker_text,
                    "position_in_document": start_pos,
                    "referenced_value": part.strip()
                })
    return citations

def extract_apa_harvard_citations(text: str) -> list:
    """
    Extracts APA/Harvard in-text citations like (Author, Year).
    """
    citations = []
    pattern = r'\(\s*([A-Z][A-Za-z]+)(?:\s+et\s+al\.)?(?:\s*(?:and|&)\s*[A-Z][A-Za-z]+)?\s*,\s*(\d{4})\s*\)'
    for match in re.finditer(pattern, text):
        marker_text = match.group(0)
        author = match.group(1)
        year = match.group(2)
        start_pos = match.start()
        
        citations.append({
            "marker_text": marker_text,
            "position_in_document": start_pos,
            "referenced_value": {
                "author": author,
                "year": year
            }
        })
    return citations

def extract_mla_citations(text: str) -> list:
    """
    Extracts MLA in-text citations like (Author Page).
    """
    citations = []
    pattern = r'\(\s*([A-Z][A-Za-z]+)(?:\s*(?:et\s+al\.|and|&)\s*[A-Z][A-Za-z]+)?\s+(\d{1,3})\s*\)'
    for match in re.finditer(pattern, text):
        marker_text = match.group(0)
        author = match.group(1)
        page = match.group(2)
        start_pos = match.start()
        
        citations.append({
            "marker_text": marker_text,
            "position_in_document": start_pos,
            "referenced_value": {
                "author": author,
                "page": page
            }
        })
    return citations

def parse_reference_list(text: str, style: str) -> list[dict]:
    """
    Locates the bibliography section and parses individual reference entries based on style rules.
    """
    ref_headers = list(re.finditer(r'\b(references|bibliography|works\s+cited|notes)\b', text.lower()))
    if not ref_headers:
        return []
        
    start_pos = ref_headers[-1].start()
    ref_text = text[start_pos:]
    
    header_name = ref_headers[-1].group(1).strip()
    header_len = len(header_name)
    content_text = ref_text[header_len:].strip()
    
    parsed_entries = []
    
    if style == "IEEE":
        parts = re.split(r'(\[\d+\])', content_text)
        for i in range(1, len(parts), 2):
            num_marker = parts[i]
            entry_content = parts[i+1] if i+1 < len(parts) else ""
            entry_content_clean = re.sub(r'\s+', ' ', entry_content).strip()
            num_val = num_marker.strip("[]")
            
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', entry_content_clean)
            year = year_match.group(1) if year_match else None
            
            parsed_entries.append({
                "entry_number_or_index": num_val,
                "raw_text": f"{num_marker} {entry_content_clean}",
                "author_surname": None,
                "year": year
            })
    else:
        # APA / MLA / Harvard / Chicago: split on lines
        lines = [line.strip() for line in content_text.split('\n') if line.strip()]
        filtered_lines = []
        for line in lines:
            if re.match(r'^\d+$', line) or len(line) < 15:
                continue
            filtered_lines.append(line)
            
        for idx, line in enumerate(filtered_lines):
            author_match = re.match(r'^([A-Z][a-zA-Z]+)', line)
            author = author_match.group(1) if author_match else None
            
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
            year = year_match.group(1) if year_match else None
            
            parsed_entries.append({
                "entry_number_or_index": str(idx + 1),
                "raw_text": line,
                "author_surname": author,
                "year": year
            })
            
    return parsed_entries

def match_citations_to_references(in_text_citations: list, reference_list: list[dict], style: str) -> dict:
    """
    Links each in-text citation occurrence to its corresponding reference list entry.
    """
    matched_citations = []
    unmatched_in_text = []
    referenced_keys = set()
    
    ref_map = {}
    if style == "IEEE":
        for ref in reference_list:
            ref_map[ref["entry_number_or_index"]] = ref
    else:
        for ref in reference_list:
            if ref["author_surname"]:
                ref_map[ref["author_surname"].lower()] = ref

    for cit in in_text_citations:
        val = cit["referenced_value"]
        matched_ref = None
        
        if style == "IEEE":
            matched_ref = ref_map.get(val)
        else:
            # Match by author surname
            author = val.get("author", "").lower() if isinstance(val, dict) else str(val).lower()
            matched_ref = ref_map.get(author)
            
        if matched_ref:
            referenced_keys.add(matched_ref["entry_number_or_index"])
            matched_citations.append({
                "marker_text": cit["marker_text"],
                "referenced_value": str(val),
                "reference_entry": matched_ref["raw_text"],
                "status": "matched"
            })
        else:
            unmatched_in_text.append({
                "marker_text": cit["marker_text"],
                "referenced_value": str(val)
            })

    # Find unused reference list entries
    unreferenced_entries = []
    for ref in reference_list:
        if ref["entry_number_or_index"] not in referenced_keys:
            unreferenced_entries.append({
                "entry_number_or_index": ref["entry_number_or_index"],
                "raw_text": ref["raw_text"]
            })

    return {
        "matched_citations": matched_citations,
        "unmatched_citations": unmatched_in_text,
        "unreferenced_entries": unreferenced_entries
    }

def audit_citations_integrity(text: str) -> dict:
    """
    Orchestrates the entire style detection, reference extraction, and citation matching pipeline.
    """
    style_info = detect_citation_style(text)
    style = style_info["style"]
    
    # Locate reference bibliography list
    ref_list = parse_reference_list(text, style)
    
    # Segment body text separate from bibliography list to prevent self-matching markers
    ref_headers = list(re.finditer(r'\b(references|bibliography|works\s+cited|notes)\b', text.lower()))
    body_text = text[:ref_headers[-1].start()] if ref_headers else text

    # Extract in-text citation markers only from body_text
    if style == "IEEE":
        in_text = extract_ieee_citations(body_text)
    elif style in ["APA", "Harvard"]:
        in_text = extract_apa_harvard_citations(body_text)
    elif style == "MLA":
        in_text = extract_mla_citations(body_text)
    else:
        in_text = []
        
    # Match in-text citations against the reference bibliography
    match_results = match_citations_to_references(in_text, ref_list, style)
    
    # Run DOI-based metadata verification on the parsed reference list
    from src.doi_verifier import verify_all_references
    ref_list = verify_all_references(ref_list)
    
    # Calculate health score based on DOI verification status
    dois_count = 0
    suspicious_count = 0
    for ref in ref_list:
        status = ref["status"]
        if status in ["verified", "partial_match", "mismatch", "doi_not_found", "lookup_failed"]:
            dois_count += 1
            if status in ["mismatch", "doi_not_found"]:
                suspicious_count += 1
                
    if dois_count > 0:
        health_score = max(0, 100 - int((suspicious_count / dois_count) * 100))
    else:
        health_score = 100
        
    return {
        "style": style,
        "confidence": style_info["confidence"],
        "in_text_marker_count": style_info["in_text_marker_count"],
        "health_score": health_score,
        "references": ref_list,
        "matched_citations": match_results["matched_citations"],
        "unmatched_citations": match_results["unmatched_citations"],
        "unreferenced_entries": match_results["unreferenced_entries"]
    }
