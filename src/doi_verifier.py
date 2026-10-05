import os
import re
import time
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor

def extract_doi(reference_text: str) -> str | None:
    """
    Extracts DOI from reference text if present.
    Matches bare DOI, prefixed, or URL style DOIs, and strips trailing punctuation.
    """
    pattern = r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    match = re.search(pattern, reference_text)
    if match:
        doi = match.group(0)
        while doi and doi[-1] in ".,;)]}":
            doi = doi[:-1]
        return doi
    return None

def resolve_doi(doi: str) -> dict | None:
    """
    Queries CrossRef's public metadata API for a given DOI.
    Extracts title, authors, journal, year, publisher, and citation count.
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
                        
                citation_count = message.get("is-referenced-by-count", 0)
                
                return {
                    "title": title,
                    "authors": authors,
                    "journal": container,
                    "year": year,
                    "publisher": message.get("publisher", ""),
                    "doi": doi,
                    "citation_count": citation_count,
                    "provider": "CrossRef",
                    "status": "resolved"
                }
            elif response.status_code == 404:
                return {"status": "not_found", "doi": doi}
            elif response.status_code == 429:
                # Rate limited by CrossRef public API — wait retry-after seconds or default to 1s
                retry_after = float(response.headers.get("Retry-After", 1.0))
                if attempt == 0:
                    time.sleep(min(retry_after, 2.0))
                    continue
                return {"status": "lookup_failed", "doi": doi}
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

def search_crossref_by_title(title: str) -> dict | None:
    """
    Searches CrossRef API by paper title when no DOI is present.
    """
    if not title or len(title) < 10:
        return None
    
    encoded_title = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.title={encoded_title}&rows=1"
    headers = {"User-Agent": "VeriPaperAI/1.0 (mailto:integrity@veripaper.ai)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5.0)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                title_list = item.get("title", [])
                matched_title = title_list[0] if title_list else ""
                
                # Check similarity score
                words_original = set(re.findall(r'\b\w{4,}\b', title.lower()))
                words_matched = set(re.findall(r'\b\w{4,}\b', matched_title.lower()))
                overlap = len(words_original.intersection(words_matched)) / max(len(words_original), 1)
                
                if overlap >= 0.5:
                    authors = [a.get("family") for a in item.get("author", []) if a.get("family")]
                    container = item.get("container-title", [""])[0] if item.get("container-title") else ""
                    published = item.get("published-print") or item.get("published-online") or item.get("created")
                    year = str(published.get("date-parts", [[None]])[0][0]) if published and published.get("date-parts") else None
                    
                    return {
                        "title": matched_title,
                        "authors": authors,
                        "journal": container,
                        "year": year,
                        "publisher": item.get("publisher", ""),
                        "doi": item.get("DOI", ""),
                        "citation_count": item.get("is-referenced-by-count", 0),
                        "provider": "CrossRef Title Search",
                        "status": "resolved"
                    }
    except Exception:
        pass
    return None

def compare_reference_to_metadata(reference_text: str, resolved_metadata: dict) -> dict:
    """
    Fuzzy-compares the reference text against resolved CrossRef / OpenAlex metadata.
    """
    if not resolved_metadata or resolved_metadata.get("status") == "not_found":
        return {
            "status": "doi_not_found",
            "details": "Citation could not be verified in CrossRef or OpenAlex registries — suspected fabricated reference.",
            "metadata": None
        }
    if resolved_metadata.get("status") == "lookup_failed":
        return {
            "status": "lookup_failed",
            "details": "Registry verification service timed out.",
            "metadata": None
        }
        
    ref_clean = reference_text.lower()
    title_clean = resolved_metadata.get("title", "").lower()
    
    if title_clean and title_clean in ref_clean:
        similarity = 1.0
    else:
        title_words = re.findall(r'\b\w+\b', title_clean)
        if title_words:
            matched_words = [w for w in title_words if w in ref_clean]
            similarity = len(matched_words) / len(title_words)
        else:
            similarity = 0.0
            
    year_match = True
    resolved_year = resolved_metadata.get("year")
    if resolved_year and resolved_year not in reference_text:
        year_match = False
        similarity *= 0.9
            
    if similarity >= 0.70:
        status = "verified"
        details = f"Verified: Registered paper found in {resolved_metadata.get('provider', 'CrossRef')} ({resolved_metadata.get('citation_count', 0)} citations)."
    elif similarity >= 0.35:
        status = "partial_match"
        details = f"Review Suggested: Stated reference partially matches '{resolved_metadata.get('title')}'."
    else:
        status = "mismatch"
        details = f"Warning: Stated title does not match registered title '{resolved_metadata.get('title')}'."
        
    if not year_match and status == "verified":
        status = "partial_match"
        details = f"Review Suggested: Year discrepancy with registered year {resolved_year}."
        
    return {
        "status": status,
        "details": details,
        "metadata": resolved_metadata
    }

def verify_all_references(reference_list: list[dict]) -> list[dict]:
    """
    Batch-verifies references containing DOIs or titles concurrently.
    """
    for ref in reference_list:
        ref["doi"] = extract_doi(ref["raw_text"])
        
    dois_to_resolve = list(set(ref["doi"] for ref in reference_list if ref["doi"]))
    
    resolved_map = {}
    if dois_to_resolve:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(resolve_doi, dois_to_resolve))
            for doi, res in zip(dois_to_resolve, results):
                resolved_map[doi] = res
                
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
            title_match = re.search(r'["“]([^"”]+)["”]', ref["raw_text"])
            title = title_match.group(1) if title_match else ref["raw_text"]
            resolved = search_crossref_by_title(title)
            if resolved:
                comparison = compare_reference_to_metadata(ref["raw_text"], resolved)
                ref["status"] = comparison["status"]
                ref["details"] = comparison["details"]
                ref["metadata"] = comparison["metadata"]
                ref["doi"] = resolved.get("doi")
            else:
                ref["status"] = "no_doi_present"
                ref["details"] = "No DOI or verified paper match found in CrossRef registry."
                ref["metadata"] = None
            
        updated_references.append(ref)
        
    return updated_references
