"""
Semantics Guardrail & Citation Preservation Engine for AccaHumanize-CL.

Provides AST-level token locking for inline citations, DOIs, URLs, and LaTeX equations,
and evaluates semantic similarity between original and rewritten text.
"""

import re
import math
from typing import Tuple, Dict, List

# Try importing sentence_transformers for deep semantic similarity calculation
try:
    from sentence_transformers import SentenceTransformer, util
    _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _SEMANTIC_MODEL = None


class SemanticsGuardrail:
    def __init__(self):
        # Regex patterns for elements that must be locked
        self.patterns = {
            "LATEX_BLOCK": r'\$\$.*?\$\$',
            "LATEX_INLINE": r'\$.*?\$',
            "DOI": r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',
            "URL": r'https?://[^\s()<>]+',
            "IEEE_CITATION": r'\[\d+(?:\s*(?:,\s*\d+|-\s*\d+|–\s*\d+))*\s*\]',
            "APA_CITATION": r'\(\s*[A-Z][A-Za-z]+(?:\s+et\s+al\.)?(?:\s*(?:and|&)\s*[A-Z][A-Za-z]+)?\s*,\s*\d{4}\s*\)',
        }

    def lock_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replaces citations, DOIs, URLs, and LaTeX formulas with placeholder tokens.
        Returns locked_text and a map of placeholder -> original token.
        """
        lock_dict = {}
        locked_text = text
        placeholder_idx = 0

        # Process patterns in priority order
        for p_name, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, locked_text))
            # Reverse order replacement to preserve character indices
            for m in reversed(matches):
                orig_val = m.group(0)
                placeholder = f"__LOCK_{p_name}_{placeholder_idx}__"
                placeholder_idx += 1
                lock_dict[placeholder] = orig_val
                locked_text = locked_text[:m.start()] + placeholder + locked_text[m.end():]

        return locked_text, lock_dict

    def restore_tokens(self, text: str, lock_dict: Dict[str, str]) -> str:
        """
        Restores placeholder tokens back to their original citations/math values.
        """
        restored_text = text
        for placeholder, original in lock_dict.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def verify_academic_integrity(self, original_text: str, candidate_text: str, lock_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Verifies that all locked citations, DOIs, and math formulas exist in candidate text.
        Returns (is_valid, list_of_missing_tokens).
        """
        missing = []
        for placeholder, original_token in lock_dict.items():
            if placeholder in original_text or original_token in original_text:
                if placeholder not in candidate_text and original_token not in candidate_text:
                    missing.append(original_token)
        return len(missing) == 0, missing

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates cosine similarity between text1 and text2.
        Uses SentenceTransformer if available, otherwise falls back to Word Jaccard / Cosine.
        """
        if not text1.strip() or not text2.strip():
            return 1.0

        if _SEMANTIC_MODEL is not None:
            try:
                emb1 = _SEMANTIC_MODEL.encode(text1, convert_to_tensor=True)
                emb2 = _SEMANTIC_MODEL.encode(text2, convert_to_tensor=True)
                similarity = float(util.cos_sim(emb1, emb2)[0][0])
                return max(0.0, min(1.0, similarity))
            except Exception:
                pass

        # Fallback: Character n-gram & Word overlap similarity
        return self._fallback_text_similarity(text1, text2)

    def _fallback_text_similarity(self, text1: str, text2: str) -> float:
        """Lightweight semantic similarity fallback."""
        w1 = set(re.findall(r'\w+', text1.lower()))
        w2 = set(re.findall(r'\w+', text2.lower()))

        if not w1 or not w2:
            return 1.0

        intersection = len(w1.intersection(w2))
        union = len(w1.union(w2))
        jaccard = intersection / max(1, union)

        # Combine jaccard with length ratio
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        return round(0.7 * jaccard + 0.3 * len_ratio, 4)
