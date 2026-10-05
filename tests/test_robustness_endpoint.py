import unittest
from fastapi.testclient import TestClient
from src.app import app

class TestRobustnessEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_robustness_evaluate_endpoint(self):
        payload = {
            "text": "Furthermore, it is crucial to delve into the multifaceted structure of transformer models. Moreover, meticulous parameter tuning underscores the overall performance improvements."
        }
        response = self.client.post("/robustness/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify required keys
        self.assertIn("original_ai_percentage", data)
        self.assertIn("rewritten_ai_percentage", data)
        self.assertIn("robustness_confidence_delta", data)
        self.assertIn("resilience_verdict", data)
        self.assertIn("explanation", data)

        # Humanizer feature: verify rewritten_text is returned
        self.assertIn("rewritten_text", data)
        self.assertIsInstance(data["rewritten_text"], str)
        self.assertGreater(len(data["rewritten_text"]), 0)

    def test_robustness_humanize_endpoint(self):
        payload = {
            "text": "Furthermore, it is crucial to delve into the multifaceted structure of transformer models."
        }
        response = self.client.post("/robustness/humanize", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rewritten_text", data)
        self.assertNotIn("delve", data["rewritten_text"].lower())

    def test_robustness_evaluate_empty_payload(self):
        response = self.client.post("/robustness/evaluate", json={"text": "   "})
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
