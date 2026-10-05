import re

"""
Academic Paper Section Analyzer & AI Forensic Segmenter.
Intelligently segments research papers into standard structural sections (IMRaD & beyond)
and computes granular per-section AI risk probabilities and word densities.
"""

# Canonical Section Categories with Keyword Patterns
SECTION_PATTERNS = [
    {
        "category": "ABSTRACT",
        "label": "Abstract & Keywords",
        "pattern": r'^(?:abstract|executive summary|keywords|index terms)\b'
    },
    {
        "category": "INTRO",
        "label": "1. Introduction & Background",
        "pattern": r'^(?:1\.?|I\.?|section 1)?\s*(?:introduction|background|motivation|problem statement)\b'
    },
    {
        "category": "LIT_REVIEW",
        "label": "2. Literature Review & Related Work",
        "pattern": r'^(?:2\.?|II\.?|section 2)?\s*(?:related work|literature review|literature survey|prior work|state of the art)\b'
    },
    {
        "category": "METHODOLOGY",
        "label": "3. Proposed Methodology & Design",
        "pattern": r'^(?:3\.?|III\.?|section 3)?\s*(?:methodology|proposed method|system model|architecture|materials and methods|theoretical framework|approach|study area)\b'
    },
    {
        "category": "RESULTS",
        "label": "4. Experimental Results & Evaluation",
        "pattern": r'^(?:4\.?|IV\.?|section 4)?\s*(?:experimental setup|experiments|results|performance evaluation|comparative study|implementation|water balance)\b'
    },
    {
        "category": "DISCUSSION",
        "label": "5. Discussion & Limitations",
        "pattern": r'^\s*(?:5\.?|V\.?|section 5)?\s*(?:discussion|limitations|threats to validity|ablation study)\b'
    },
    {
        "category": "CONCLUSION",
        "label": "6. Conclusion & Future Work",
        "pattern": r'^\s*(?:6\.?|VI\.?|section 6)?\s*(?:conclusion|concluding remarks|future work|summary)\b'
    },
    {
        "category": "REFERENCES",
        "label": "7. References & Bibliography",
        "pattern": r'^\s*(?:7\.?|VII\.?|section 7)?\s*(?:references|bibliography|works cited|appendix)\b'
    }
]

def classify_text_segment(text: str) -> dict | None:
    """
    Checks if the beginning of a line or sentence matches an academic section heading.
    """
    clean = text.strip()
    if not clean:
        return None

    # Inspect first 80 characters of the text segment
    prefix = clean[:80].strip()

    for item in SECTION_PATTERNS:
        if re.search(item["pattern"], prefix, re.IGNORECASE):
            # Extract heading title up to 60 chars or first colon/newline
            title_match = re.split(r'[\n\:\.]', prefix)[0].strip()
            return {
                "category": item["category"],
                "label": item["label"],
                "heading": title_match if len(title_match) > 2 else item["label"]
            }

    # Match numbered section patterns (e.g. "1 Introduction", "2.1 Study Area", "3 Results")
    num_match = re.match(r'^(?:[0-9]+\.?[0-9]*|[IVXLCDM]+\.)\s+([A-Z][a-zA-Z\s]{2,40})', prefix)
    if num_match:
        heading_title = prefix[:45].strip()
        return {
            "category": "GENERAL_SECTION",
            "label": heading_title,
            "heading": heading_title
        }

    return None

def analyze_paper_sections(doc_text: str, sentences: list[dict]) -> dict:
    """
    Splits paper sentences into structural sections based on headings and paragraph indices,
    computing per-section weighted AI risk percentages.
    """
    if not sentences:
        return {"sections": [], "section_breakdown": []}

    sections_list = []
    current_section = {
        "section_id": 1,
        "title": "Title & Manuscript Metadata",
        "category": "HEADER",
        "sentences": [],
        "word_count": 0,
        "weighted_ai": 0.0
    }

    for s in sentences:
        text = s.get("text", "").strip()
        prob = s.get("ai_probability", 0.0)
        words = len([w for w in text.split() if w.strip()])

        heading_info = classify_text_segment(text)
        
        # Start a new section if heading is detected and current section has content
        if heading_info and len(current_section["sentences"]) > 0 and heading_info["category"] != current_section["category"]:
            if current_section["word_count"] > 0:
                current_section["ai_percentage"] = round((current_section["weighted_ai"] / current_section["word_count"]) * 100, 1)
            else:
                current_section["ai_percentage"] = round(prob * 100, 1)

            pct = current_section["ai_percentage"]
            current_section["risk_level"] = "HIGH" if pct >= 65 else "MEDIUM" if pct >= 45 else "LOW"
            sections_list.append(current_section)

            current_section = {
                "section_id": len(sections_list) + 1,
                "title": heading_info["heading"],
                "category": heading_info["category"],
                "sentences": [],
                "word_count": 0,
                "weighted_ai": 0.0
            }

        current_section["sentences"].append(s)
        current_section["word_count"] += words
        current_section["weighted_ai"] += words * prob

    # Finalize last section
    if current_section["sentences"]:
        if current_section["word_count"] > 0:
            current_section["ai_percentage"] = round((current_section["weighted_ai"] / current_section["word_count"]) * 100, 1)
        else:
            current_section["ai_percentage"] = 0.0

        pct = current_section["ai_percentage"]
        current_section["risk_level"] = "HIGH" if pct >= 65 else "MEDIUM" if pct >= 45 else "LOW"
        sections_list.append(current_section)

    # Calculate breakdown overview
    breakdown = [
        {
            "section_id": sec["section_id"],
            "title": sec["title"],
            "category": sec["category"],
            "word_count": sec["word_count"],
            "sentence_count": len(sec["sentences"]),
            "ai_percentage": sec["ai_percentage"],
            "risk_level": sec["risk_level"]
        }
        for sec in sections_list
    ]

    highest_risk = max(breakdown, key=lambda x: x["ai_percentage"]) if breakdown else None
    lowest_risk = min(breakdown, key=lambda x: x["ai_percentage"]) if breakdown else None

    return {
        "sections": sections_list,
        "section_breakdown": breakdown,
        "highest_risk_section": highest_risk,
        "lowest_risk_section": lowest_risk
    }

