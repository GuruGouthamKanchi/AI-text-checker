import unittest
from src.citation_parser import (
    detect_citation_style,
    extract_ieee_citations,
    extract_apa_harvard_citations,
    extract_mla_citations,
    parse_reference_list,
    match_citations_to_references,
    audit_citations_integrity
)

class TestCitationParser(unittest.TestCase):
    def test_ieee_style_detection(self):
        text = """
        This is a paper about antennas [1]. Some other work has explored array configurations [2], [3].
        A related design combined a cross-shaped DGS with semi-circular patch slots to reach high gain [4]–[6].
        
        REFERENCES
        [1] Yuvaraj K, Sanam Narayana Reddy. "Progressive Design of An E-Slotted Microstrip Patch Antenna."
        [2] Khalid S, et al. "4-port MIMO antenna with defected ground structure for 5G."
        [3] Khabba A, et al. "Pretty-small four-port dual-wideband 28/38 GHz MIMO antenna."
        [4] Jones B. "Antenna engineering and design handbook."
        [5] Smith A. "MIMO systems in 5G communication."
        [6] Davis C. "Substrate choice in millimetre-wave antennas."
        """
        result = detect_citation_style(text)
        self.assertEqual(result["style"], "IEEE")
        self.assertGreaterEqual(result["confidence"], 0.8)
        self.assertEqual(result["in_text_marker_count"], 5)  # [1], [2], [3], [4], [6]

    def test_ieee_extraction_and_expansion(self):
        text = "Design isolation can be improved [3], [5]–[7], [12]."
        citations = extract_ieee_citations(text)
        referenced_vals = [c["referenced_value"] for c in citations]
        # Should parse [3], expand [5]-[7] to 5, 6, 7, and parse [12]
        self.assertIn("3", referenced_vals)
        self.assertIn("5", referenced_vals)
        self.assertIn("6", referenced_vals)
        self.assertIn("7", referenced_vals)
        self.assertIn("12", referenced_vals)
        self.assertEqual(len(referenced_vals), 5)

    def test_apa_style_detection(self):
        text = """
        Previous research indicates that antenna array spacing is crucial (Vaswani et al., 2017).
        However, other models suggest different configurations (Smith & Jones, 2019).
        
        References
        Smith, A. & Jones, B. (2019). "MIMO design guidelines."
        Vaswani, A. (2017). "Attention is all you need."
        """
        result = detect_citation_style(text)
        self.assertIn(result["style"], ["APA", "Harvard"])
        self.assertGreaterEqual(result["confidence"], 0.7)

    def test_mla_style_detection(self):
        text = """
        This structure is often referred to in literature (Balanis 45).
        Another author has presented a similar method (Pozar 112).
        
        Works Cited
        Balanis, Constantine. "Antenna Theory Analysis and Design."
        Pozar, David. "Microstrip Antennas."
        """
        result = detect_citation_style(text)
        self.assertEqual(result["style"], "MLA")
        self.assertGreaterEqual(result["confidence"], 0.7)

    def test_unknown_style_detection(self):
        text = "This is a generic blog post about coding without any academic citations in it."
        result = detect_citation_style(text)
        self.assertEqual(result["style"], "Unknown")
        self.assertEqual(result["confidence"], 0.0)

    def test_matching_integrity(self):
        text = """
        Let's check matching [1] and [3]. We also have [4].
        
        REFERENCES
        [1] Author One. "Paper One."
        [2] Author Two. "Paper Two."
        [3] Author Three. "Paper Three."
        """
        audit = audit_citations_integrity(text)
        self.assertEqual(audit["style"], "IEEE")
        
        # Matches in-text citations [1] -> ref 1, [3] -> ref 3
        matched = [c["referenced_value"] for c in audit["matched_citations"]]
        self.assertIn("1", matched)
        self.assertIn("3", matched)
        
        # Unmatched in-text: [4]
        unmatched = [c["referenced_value"] for c in audit["unmatched_citations"]]
        self.assertIn("4", unmatched)
        
        # Unreferenced bib entries: [2] is in references but never cited in-text
        unreferenced = [r["entry_number_or_index"] for r in audit["unreferenced_entries"]]
        self.assertIn("2", unreferenced)

if __name__ == "__main__":
    unittest.main()
