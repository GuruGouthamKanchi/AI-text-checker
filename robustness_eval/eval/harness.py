import os
import argparse
from typing import List, Callable, Optional, Dict, Any
from robustness_eval.config import DEFAULT_CONFIG, RobustnessConfig
from robustness_eval.datasets.adversarial_pairs import AdversarialPair, AdversarialDataset
from robustness_eval.rewriters import combined_rewrite
from robustness_eval.eval.report import RobustnessReport

class AdversarialHarness:
    """
    Evaluation harness that loads adversarial text datasets, runs the existing unmodified
    detector inference function against original_text and rewritten_text, and outputs metrics.
    """

    def __init__(
        self,
        inference_fn: Optional[Callable[[str], float]] = None,
        config: Optional[RobustnessConfig] = None
    ):
        self.config = config or DEFAULT_CONFIG
        
        if inference_fn is not None:
            self.inference_fn = inference_fn
        else:
            self.inference_fn = self._default_black_box_inference

    def _default_black_box_inference(self, text: str) -> float:
        """
        Black-box wrapper around existing detector's run_analysis_pipeline.
        Imports existing detector; existing detector has zero knowledge of this module.
        """
        from src.app import run_analysis_pipeline
        result = run_analysis_pipeline(doc_text=text, filename="eval_sample.txt", page_count=1)
        return float(result.get("overall_ai_percentage", 0.0))

    def evaluate_pair(self, pair: AdversarialPair) -> AdversarialPair:
        """
        Evaluates a single pair: queries black-box detector for original and rewritten text,
        records scores, score drop, and whether verdict flipped from AI -> Human.
        """
        orig_score = self.inference_fn(pair.original_text)
        rewritten_score = self.inference_fn(pair.rewritten_text)
        score_drop = max(0.0, orig_score - rewritten_score)
        threshold = self.config.medium_threshold * 100.0  # 60.0%
        evaded = (orig_score >= threshold) and (rewritten_score < threshold)

        pair.original_ai_score = orig_score
        pair.rewritten_ai_score = rewritten_score
        pair.score_drop = score_drop
        pair.evaded = evaded
        return pair

    def evaluate_dataset(
        self,
        dataset: AdversarialDataset,
        output_dir: str = "robustness_eval/reports"
    ) -> RobustnessReport:
        """
        Runs evaluation across all pairs in the dataset and generates reports.
        """
        print(f"[harness] Evaluating {len(dataset.pairs)} adversarial pairs against black-box detector...")
        for i, pair in enumerate(dataset.pairs):
            self.evaluate_pair(pair)
            if (i + 1) % 5 == 0 or (i + 1) == len(dataset.pairs):
                print(f"[harness] Processed {i + 1}/{len(dataset.pairs)} pairs...")

        report = RobustnessReport.from_dataset(dataset, config=self.config)
        saved_paths = report.save_reports(output_dir=output_dir)
        print(f"[harness] Analysis complete!")
        print(f"[harness] JSON Report saved to: {saved_paths['json']}")
        print(f"[harness] Markdown Report saved to: {saved_paths['markdown']}")
        return report

    def run_suite(
        self,
        samples: List[str],
        transformations: Optional[List[str]] = None
    ) -> RobustnessReport:
        """
        Convenience method to evaluate a list of sample strings directly.
        """
        dataset = AdversarialDataset()
        for sample in samples:
            pair = AdversarialPair(
                original_text=sample,
                rewritten_text=combined_rewrite(sample),
                source_label="ai",
                rewrite_method="combined_v1"
            )
            dataset.add_pair(pair)
        return self.evaluate_dataset(dataset)

def evaluate_robustness(
    dataset_path: str = "robustness_eval/datasets/adversarial_v1.jsonl",
    output_dir: str = "robustness_eval/reports",
    inference_fn: Optional[Callable[[str], float]] = None
) -> RobustnessReport:
    if not os.path.exists(dataset_path):
        from robustness_eval.datasets.build_dataset import build_adversarial_dataset
        build_adversarial_dataset(output_path=dataset_path, max_samples=10)
    dataset = AdversarialDataset.load_from_jsonl(dataset_path)
    harness = AdversarialHarness(inference_fn=inference_fn)
    return harness.evaluate_dataset(dataset, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="Standalone Adversarial Evaluation Harness for AI Text Detector.")
    parser.add_argument("--dataset", type=str, default="robustness_eval/datasets/adversarial_v1.jsonl", help="Path to adversarial .jsonl dataset.")
    parser.add_argument("--out-dir", type=str, default="robustness_eval/reports", help="Output directory for reports.")
    args = parser.parse_args()

    evaluate_robustness(dataset_path=args.dataset, output_dir=args.out_dir)

if __name__ == "__main__":
    main()
