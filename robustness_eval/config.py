from dataclasses import dataclass
import os

@dataclass
class RobustnessConfig:
    """Isolated configuration settings for adversarial robustness evaluation."""
    model_dir: str = "models/modernbert-academic"
    high_threshold: float = 0.80
    medium_threshold: float = 0.60
    perplexity_target_boost: float = 1.25
    burstiness_std_target: float = 8.5
    synonym_replacement_rate: float = 0.15
    random_seed: int = 42

DEFAULT_CONFIG = RobustnessConfig()
