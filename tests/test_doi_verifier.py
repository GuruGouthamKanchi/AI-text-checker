import unittest
from unittest.mock import patch, MagicMock
from src.doi_verifier import (
    extract_doi,
    resolve_doi,
    compare_reference_to_metadata,
    verify_all_references
)

class TestDoiVerifier(unittest.TestCase):
    def test_extract_doi(self):
        # Bare DOI
        self.assertEqual(extract_doi("Some text 10.1007/s10470-022-02045-8 more text"), "10.1007/s10470-022-02045-8")
        # DOI URL
        self.assertEqual(extract_doi("https://doi.org/10.1684/agr.2011.0499"), "10.1684/agr.2011.0499")
        # Prefixed DOI
        self.assertEqual(extract_doi("doi:10.1038/s41598-024-79859-1"), "10.1038/s41598-024-79859-1")
        # Trailing punctuation
        self.assertEqual(extract_doi("10.1007/s10470-022-02045-8."), "10.1007/s10470-022-02045-8")
        self.assertEqual(extract_doi("10.1007/s10470-022-02045-8)"), "10.1007/s10470-022-02045-8")
        # No DOI
        self.assertIsNone(extract_doi("A normal reference with no doi indicators at all."))

    @patch("requests.get")
    def test_resolve_doi_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "title": ["An Awesome Antenna Design"],
                "author": [{"family": "Smith"}, {"family": "Jones"}],
                "container-title": ["Journal of Antennas"],
                "published-print": {"date-parts": [[2022, 5, 12]]},
                "publisher": "IEEE"
            }
        }
        mock_get.return_value = mock_response
        
        result = resolve_doi("10.1000/xyz123")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["title"], "An Awesome Antenna Design")
        self.assertEqual(result["authors"], ["Smith", "Jones"])
        self.assertEqual(result["year"], "2022")

    @patch("requests.get")
    def test_resolve_doi_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = resolve_doi("10.1000/notfound")
        self.assertEqual(result["status"], "not_found")

    def test_compare_reference_to_metadata_verified(self):
        ref_text = "[14] Smith, A. and Jones, B. \"An Awesome Antenna Design\", Journal of Antennas, 2022."
        metadata = {
            "title": "An Awesome Antenna Design",
            "year": "2022",
            "status": "resolved"
        }
        res = compare_reference_to_metadata(ref_text, metadata)
        self.assertEqual(res["status"], "verified")
        self.assertIn("Verified", res["details"])

    def test_compare_reference_to_metadata_partial_match(self):
        # Stated year 2021, registered year 2022 -> partial_match
        ref_text = "[14] Smith, A. and Jones, B. \"An Awesome Antenna Design\", 2021."
        metadata = {
            "title": "An Awesome Antenna Design",
            "year": "2022",
            "status": "resolved"
        }
        res = compare_reference_to_metadata(ref_text, metadata)
        self.assertEqual(res["status"], "partial_match")
        self.assertIn("discrepancy", res["details"])

    def test_compare_reference_to_metadata_mismatch(self):
        ref_text = "[14] Smith, A. and Jones, B. \"Completely Mismatching Title Text\", 2022."
        metadata = {
            "title": "An Awesome Antenna Design",
            "year": "2022",
            "status": "resolved"
        }
        res = compare_reference_to_metadata(ref_text, metadata)
        self.assertEqual(res["status"], "mismatch")
        self.assertIn("mismatch", res["details"].lower())

    @patch("src.doi_verifier.resolve_doi")
    def test_verify_all_references(self, mock_resolve):
        # Mock resolve_doi return values
        mock_resolve.side_effect = lambda doi: {
            "title": "Real Registered Title",
            "year": "2022",
            "status": "resolved"
        }
        
        references = [
            {"raw_text": "[1] Smith A. Real Registered Title for 10.1000/doi1, 2022."},
            {"raw_text": "[2] Jones B. Mismatched Reference Text for 10.1000/doi2, 2022."},
            {"raw_text": "[3] Davis C. Reference with no doi at all."}
        ]
        
        results = verify_all_references(references)
        
        # Verify result lengths
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["status"], "verified")
        self.assertEqual(results[1]["status"], "mismatch")
        self.assertEqual(results[2]["status"], "no_doi_present")

if __name__ == "__main__":
    unittest.main()
