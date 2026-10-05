import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from robustness_eval.config import DEFAULT_CONFIG, RobustnessConfig
from robustness_eval.datasets.adversarial_pairs import AdversarialDataset, AdversarialPair

@dataclass
class RobustnessReport:
    """
    Aggregate metrics report evaluating overall accuracy on original text vs rewritten text,
    accuracy drop per rewrite method, and sentence-level patterns that fooled the model.
    """
    total_evaluations: int = 0
    accuracy_original: float = 0.0          # % of original AI text flagged as AI (score >= 60%)
    accuracy_rewritten: float = 0.0         # % of rewritten AI text still flagged as AI (score >= 60%)
    accuracy_drop: float = 0.0              # accuracy_original - accuracy_rewritten
    mean_original_score: float = 0.0
    mean_rewritten_score: float = 0.0
    mean_score_drop: float = 0.0
    verdict_flip_rate: float = 0.0          # % of samples where verdict flipped from AI -> Human
    adversarial_resilience_index: float = 0.0  # ARI score in [0.0, 1.0]
    confidence_under_attack_signal: str = "High Resilience"
    method_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fooling_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        md = []
        md.append("# Adversarial Robustness Evaluation Report")
        md.append("")
        md.append("## Overview Metrics")
        md.append(f"- **Total Evaluations**: {self.total_evaluations}")
        md.append(f"- **Original Text Accuracy**: {self.accuracy_original:.1f}%")
        md.append(f"- **Rewritten Text Accuracy**: {self.accuracy_rewritten:.1f}%")
        md.append(f"- **Accuracy Drop**: **{self.accuracy_drop:.1f}%**")
        md.append(f"- **Verdict Flip Rate (AI → Human)**: {self.verdict_flip_rate:.1f}%")
        md.append(f"- **Mean Score Drop**: {self.mean_score_drop:.1f}%")
        md.append(f"- **Adversarial Resilience Index (ARI)**: {self.adversarial_resilience_index:.2f} / 1.00")
        md.append(f"- **Confidence Under Attack Signal**: **{self.confidence_under_attack_signal}**")
        md.append("")
        md.append("## Rewrite Method Breakdown")
        md.append("| Rewrite Method | Samples | Orig Accuracy | Rewritten Accuracy | Accuracy Drop | Mean Score Drop | Flip Rate |")
        md.append("|---|---|---|---|---|---|---|")
        for method, stats in self.method_breakdown.items():
            md.append(
                f"| {method} | {int(stats.get('sample_count', 0))} | "
                f"{stats.get('orig_acc', 0.0):.1f}% | {stats.get('rewritten_acc', 0.0):.1f}% | "
                f"{stats.get('acc_drop', 0.0):.1f}% | {stats.get('score_drop', 0.0):.1f}% | "
                f"{stats.get('flip_rate', 0.0):.1f}% |"
            )
        md.append("")
        md.append("## Sentence-Level Fooling Patterns")
        if self.fooling_patterns:
            for pattern in self.fooling_patterns:
                md.append(f"- **{pattern.get('pattern_name', 'Pattern')}** (Occurrences: {pattern.get('count', 0)}, Impact: -{pattern.get('avg_score_drop', 0.0):.1f}% AI score)")
                md.append(f"  *Description*: {pattern.get('description', '')}")
        else:
            md.append("No critical fooling patterns detected.")

        return "\n".join(md)

    def save_reports(self, output_dir: str = "robustness_eval/reports", base_filename: str = "robustness_report_v1") -> Dict[str, str]:
        """Saves JSON and Markdown summary reports to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        md_path = os.path.join(output_dir, f"{base_filename}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        return {"json": json_path, "markdown": md_path}

    @classmethod
    def from_dataset(
        cls,
        dataset: AdversarialDataset,
        config: Optional[RobustnessConfig] = None
    ) -> 'RobustnessReport':
        pairs = dataset.pairs
        if not pairs:
            return cls()

        cfg = config or DEFAULT_CONFIG
        threshold = cfg.medium_threshold * 100.0  # 60.0%
        total = len(pairs)

        orig_correct = sum(1 for p in pairs if p.original_ai_score >= threshold)
        rewritten_correct = sum(1 for p in pairs if p.rewritten_ai_score >= threshold)
        verdict_flips = sum(1 for p in pairs if p.original_ai_score >= threshold and p.rewritten_ai_score < threshold)

        acc_orig = (orig_correct / total) * 100.0
        acc_rewritten = (rewritten_correct / total) * 100.0
        acc_drop = acc_orig - acc_rewritten
        flip_rate = (verdict_flips / total) * 100.0

        orig_scores = [p.original_ai_score for p in pairs]
        rewritten_scores = [p.rewritten_ai_score for p in pairs]
        drops = [p.score_drop for p in pairs]

        mean_orig = sum(orig_scores) / total
        mean_rewritten = sum(rewritten_scores) / total
        mean_drop = sum(drops) / total

        # Resilience Index calculation: 1.0 is immune, 0.0 is completely vulnerable
        ari = max(0.0, min(1.0, 1.0 - (acc_drop / 100.0)))

        if ari >= 0.85:
            signal = "High Resilience (Robust Defense)"
        elif ari >= 0.65:
            signal = "Moderate Resilience (Minor Score Decay)"
        elif ari >= 0.40:
            signal = "Low Resilience (Moderate Evasion Risk)"
        else:
            signal = "Critical Vulnerability (High Evasion Risk)"

        # Group by rewrite_method
        method_groups = {}
        for p in pairs:
            m = p.rewrite_method or "combined_v1"
            method_groups.setdefault(m, []).append(p)

        breakdown = {}
        for method, m_pairs in method_groups.items():
            m_count = len(m_pairs)
            m_orig_corr = sum(1 for p in m_pairs if p.original_ai_score >= threshold)
            m_rew_corr = sum(1 for p in m_pairs if p.rewritten_ai_score >= threshold)
            m_flips = sum(1 for p in m_pairs if p.original_ai_score >= threshold and p.rewritten_ai_score < threshold)
            m_orig_acc = (m_orig_corr / m_count) * 100.0
            m_rew_acc = (m_rew_corr / m_count) * 100.0
            m_drop = m_orig_acc - m_rew_acc
            m_score_drop = sum(p.score_drop for p in m_pairs) / m_count
            m_flip_rate = (m_flips / m_count) * 100.0

            breakdown[method] = {
                "sample_count": m_count,
                "orig_acc": round(m_orig_acc, 2),
                "rewritten_acc": round(m_rew_acc, 2),
                "acc_drop": round(m_drop, 2),
                "score_drop": round(m_score_drop, 2),
                "flip_rate": round(m_flip_rate, 2),
            }

        # Analyze fooling patterns
        fooling_patterns = []
        clause_flips = sum(1 for p in pairs if "as empirical evidence suggests" in p.rewritten_text.lower() and p.evaded)
        split_flips = sum(1 for p in pairs if "Indeed." in p.rewritten_text and p.evaded)
        vocab_flips = sum(1 for p in pairs if ("delve" in p.original_text.lower() and "delve" not in p.rewritten_text.lower()) and p.evaded)

        if clause_flips > 0:
            fooling_patterns.append({
                "pattern_name": "Syntactic Clause Insertion (Perplexity)",
                "count": clause_flips,
                "avg_score_drop": round(mean_drop, 1),
                "description": "Inserting rare academic sub-clauses disrupts detector N-gram language model expectations."
            })
        if split_flips > 0:
            fooling_patterns.append({
                "pattern_name": "Sentence Boundary Restructuring (Burstiness)",
                "count": split_flips,
                "avg_score_drop": round(mean_drop, 1),
                "description": "Splitting uniform sentences artificially increases sentence length variance (burstiness)."
            })
        if vocab_flips > 0:
            fooling_patterns.append({
                "pattern_name": "LLM Cliché Term Substitution (Vocabulary)",
                "count": vocab_flips,
                "avg_score_drop": round(mean_drop, 1),
                "description": "Replacing high-signal LLM indicator terms ('delve', 'crucial', 'leverage') with human synonyms."
            })

        return cls(
            total_evaluations=total,
            accuracy_original=round(acc_orig, 2),
            accuracy_rewritten=round(acc_rewritten, 2),
            accuracy_drop=round(acc_drop, 2),
            mean_original_score=round(mean_orig, 2),
            mean_rewritten_score=round(mean_rewritten, 2),
            mean_score_drop=round(mean_drop, 2),
            verdict_flip_rate=round(flip_rate, 2),
            adversarial_resilience_index=round(ari, 2),
            confidence_under_attack_signal=signal,
            method_breakdown=breakdown,
            fooling_patterns=fooling_patterns
        )

def generate_report(dataset: AdversarialDataset, output_dir: str = "robustness_eval/reports") -> RobustnessReport:
    report = RobustnessReport.from_dataset(dataset)
    report.save_reports(output_dir=output_dir)
    return report
