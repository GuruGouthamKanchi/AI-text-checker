import os
import tempfile
import unittest
from robustness_eval.config import RobustnessConfig
from robustness_eval.rewriters import (
    combined_rewrite,
    perplexity_rewriter,
    burstiness_rewriter,
    vocabulary_rewriter,
)
from robustness_eval.datasets.adversarial_pairs import AdversarialPair, AdversarialDataset
from robustness_eval.datasets.build_dataset import build_adversarial_dataset
from robustness_eval.eval.harness import AdversarialHarness
from robustness_eval.eval.report import RobustnessReport
from robustness_eval.finetune_appendix import load_fine_tuning_data

class TestRobustnessEvalModule(unittest.TestCase):
    def setUp(self):
        self.sample_text = (
            "Furthermore, it is crucial to delve into the multifaceted structure of transformer models. "
            "Moreover, meticulous parameter tuning underscores the overall performance improvements."
        )

    def test_perplexity_rewriter(self):
        rewritten = perplexity_rewriter(self.sample_text)
        self.assertIsInstance(rewritten, str)
        self.assertGreater(len(rewritten), 0)

    def test_burstiness_rewriter(self):
        rewritten = burstiness_rewriter(self.sample_text)
        self.assertIsInstance(rewritten, str)
        self.assertGreater(len(rewritten), 0)

    def test_vocabulary_rewriter(self):
        rewritten = vocabulary_rewriter(self.sample_text)
        self.assertIsInstance(rewritten, str)
        self.assertNotIn("delve", rewritten.lower())

    def test_combined_rewrite(self):
        rewritten = combined_rewrite(self.sample_text)
        self.assertIsInstance(rewritten, str)
        self.assertGreater(len(rewritten), 0)
        self.assertNotIn("delve", rewritten.lower())

    def test_adversarial_dataset_jsonl(self):
        pair = AdversarialPair(
            original_text=self.sample_text,
            rewritten_text="Test rewrite",
            source_label="ai",
            rewrite_method="combined_v1"
        )
        dataset = AdversarialDataset(pairs=[pair])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            dataset.save_to_jsonl(tmp_path)
            loaded = AdversarialDataset.load_from_jsonl(tmp_path)
            self.assertEqual(len(loaded.pairs), 1)
            self.assertEqual(loaded.pairs[0].original_text, self.sample_text)
            self.assertEqual(loaded.pairs[0].source_label, "ai")
            self.assertEqual(loaded.pairs[0].rewrite_method, "combined_v1")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_build_adversarial_dataset(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ds = build_adversarial_dataset(input_path="nonexistent.parquet", output_path=tmp_path, max_samples=3)
            self.assertGreaterEqual(len(ds.pairs), 1)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_fine_tuning_data(self):
        data = load_fine_tuning_data("nonexistent.jsonl")
        self.assertGreater(len(data), 0)
        self.assertIn("text", data[0])
        self.assertIn("label", data[0])

    def test_harness_with_mock_inference(self):
        # Test harness using mock inference function (black-box)
        def mock_detector(text: str) -> float:
            if "delve" in text.lower():
                return 90.0
            return 40.0

        harness = AdversarialHarness(inference_fn=mock_detector)
        report = harness.run_suite([self.sample_text], transformations=["combined"])
        self.assertEqual(report.total_evaluations, 1)
        self.assertGreater(report.mean_original_score, 0.0)

if __name__ == "__main__":
    unittest.main()
