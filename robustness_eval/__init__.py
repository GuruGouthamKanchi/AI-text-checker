"""
Robustness Evaluation & AI Text Humanizer Suite.
Provides adversarial evaluation metrics and state-of-the-art AI text humanization.
"""

from robustness_eval.config import RobustnessConfig, DEFAULT_CONFIG
from robustness_eval.datasets.adversarial_pairs import AdversarialPair, AdversarialDataset
from robustness_eval.rewriters import (
    combined_rewrite,
    perplexity_rewriter,
    rewrite_perplexity,
    burstiness_rewriter,
    rewrite_burstiness,
    vocabulary_rewriter,
    rewrite_vocabulary,
    PerplexityRewriter,
    BurstinessRewriter,
    VocabularyRewriter,
)
from robustness_eval.eval import (
    AdversarialHarness,
    evaluate_robustness,
    RobustnessReport,
    generate_report,
)
from robustness_eval.finetune_appendix import finetune_robust_model, load_fine_tuning_data
from robustness_eval.humanizer import (
    AIHumanizerEngine,
    get_humanizer_engine,
    humanize_text,
    train_t5_humanizer,
    load_training_pairs,
)
from robustness_eval.router import router as robustness_router

__all__ = [
    "RobustnessConfig",
    "DEFAULT_CONFIG",
    "AdversarialPair",
    "AdversarialDataset",
    "combined_rewrite",
    "perplexity_rewriter",
    "rewrite_perplexity",
    "burstiness_rewriter",
    "rewrite_burstiness",
    "vocabulary_rewriter",
    "rewrite_vocabulary",
    "PerplexityRewriter",
    "BurstinessRewriter",
    "VocabularyRewriter",
    "AdversarialHarness",
    "evaluate_robustness",
    "RobustnessReport",
    "generate_report",
    "finetune_robust_model",
    "load_fine_tuning_data",
    "AIHumanizerEngine",
    "get_humanizer_engine",
    "humanize_text",
    "train_t5_humanizer",
    "load_training_pairs",
    "robustness_router",
]
