import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor

def extract_doi(reference_text: str) -> str | None:
    """
    Extracts DOI from reference text if present.
    Matches bare DOI, prefixed, or URL style DOIs, and strips trailing punctuation.
    """
    # Pattern to match a general DOI
    pattern = r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    match = re.search(pattern, reference_text)
    if match:
        doi = match.group(0)
        # Strip trailing punctuation commonly appended to DOIs in bibliographies
        while doi and doi[-1] in ".,;)]}":
            doi = doi[:-1]
        return doi
    return None

def resolve_doi(doi: str) -> dict | None:
    """
    Queries CrossRef's public metadata API for a given DOI.
    Includes timeout, retry-once, and polite User-Agent headers.
    """
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "VeriPaperAI/1.0 (mailto:integrity@veripaper.ai)"
    }
    
    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})
                
                # Title
                title_list = message.get("title", [])
                title = title_list[0] if title_list else ""
                
                # Authors
                authors = []
                for author_obj in message.get("author", []):
                    family = author_obj.get("family")
                    if family:
                        authors.append(family)
                        
                # Journal / Container
                container_list = message.get("container-title", [])
                container = container_list[0] if container_list else ""
                
                # Year
                year = None
                published = message.get("published-print") or message.get("published-online") or message.get("created")
                if published:
                    date_parts = published.get("date-parts", [])
                    if date_parts and date_parts[0]:
                        year = str(date_parts[0][0])
                        
                return {
                    "title": title,
                    "authors": authors,
                    "journal": container,
                    "year": year,
                    "publisher": message.get("publisher", ""),
                    "doi": doi,
                    "status": "resolved"
                }
            elif response.status_code == 404:
                return {"status": "not_found", "doi": doi}
            else:
                return {"status": "lookup_failed", "doi": doi}
        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(0.5)
                continue
            return {"status": "lookup_failed", "doi": doi}
        except Exception:
            return {"status": "lookup_failed", "doi": doi}
            
    return {"status": "lookup_failed", "doi": doi}

def compare_reference_to_metadata(reference_text: str, resolved_metadata: dict) -> dict:
    """
    Fuzzy-compares the reference text against resolved CrossRef metadata.
    """
    if not resolved_metadata or resolved_metadata.get("status") == "not_found":
        return {
            "status": "doi_not_found",
            "details": "DOI could not be verified — may be fabricated or contain a typo.",
            "metadata": None
        }
    if resolved_metadata.get("status") == "lookup_failed":
        return {
            "status": "lookup_failed",
            "details": "CrossRef registry verification couldn't be completed (network issue).",
            "metadata": None
        }
        
    ref_clean = reference_text.lower()
    title_clean = resolved_metadata["title"].lower()
    
    # Substring check first
    if title_clean and title_clean in ref_clean:
        similarity = 1.0
    else:
        # Token overlap ratio check
        title_words = re.findall(r'\b\w+\b', title_clean)
        if title_words:
            matched_words = [w for w in title_words if w in ref_clean]
            similarity = len(matched_words) / len(title_words)
        else:
            similarity = 0.0
            
    # Year check
    year_match = True
    resolved_year = resolved_metadata.get("year")
    if resolved_year:
        if resolved_year not in reference_text:
            year_match = False
            similarity *= 0.9  # apply a penalty for mismatching years
            
    if similarity >= 0.75:
        status = "verified"
        details = f"Verified: Match found for '{resolved_metadata['title']}'."
    elif similarity >= 0.40:
        status = "partial_match"
        details = f"Review Suggested: Stated reference partially matches metadata for '{resolved_metadata['title']}'."
    else:
        status = "mismatch"
        details = f"Warning: Metadata mismatch. Stated citation title does not match registered title '{resolved_metadata['title']}'."
        
    if not year_match and status == "verified":
        status = "partial_match"
        details = f"Review Suggested: Year discrepancy. Stated reference year does not match registered year {resolved_year}."
        
    return {
        "status": status,
        "details": details,
        "metadata": resolved_metadata
    }

def verify_all_references(reference_list: list[dict]) -> list[dict]:
    """
    Batch-verifies references containing DOIs concurrently.
    """
    # Extract DOIs
    for ref in reference_list:
        ref["doi"] = extract_doi(ref["raw_text"])
        
    # Gather unique DOIs to resolve
    dois_to_resolve = list(set(ref["doi"] for ref in reference_list if ref["doi"]))
    
    # Resolve concurrently (max 5 workers)
    resolved_map = {}
    if dois_to_resolve:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(resolve_doi, dois_to_resolve))
            for doi, res in zip(dois_to_resolve, results):
                resolved_map[doi] = res
                
    # Compare each reference
    updated_references = []
    for ref in reference_list:
        doi = ref["doi"]
        if doi:
            resolved = resolved_map.get(doi)
            comparison = compare_reference_to_metadata(ref["raw_text"], resolved)
            ref["status"] = comparison["status"]
            ref["details"] = comparison["details"]
            ref["metadata"] = comparison["metadata"]
        else:
            ref["status"] = "no_doi_present"
            ref["details"] = "No DOI found in this reference entry."
            ref["metadata"] = None
            
        updated_references.append(ref)
        
    return updated_references
