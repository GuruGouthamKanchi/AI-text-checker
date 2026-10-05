"""
Datasets subpackage for managing adversarial text pairs.
"""
from .adversarial_pairs import AdversarialPair, AdversarialDataset
from .build_dataset import build_adversarial_dataset, load_ai_training_samples

__all__ = [
    "AdversarialPair",
    "AdversarialDataset",
    "build_adversarial_dataset",
    "load_ai_training_samples",
]
