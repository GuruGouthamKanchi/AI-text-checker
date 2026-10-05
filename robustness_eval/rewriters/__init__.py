from typing import Optional
from robustness_eval.rewriters.perplexity_rewriter import perplexity_rewriter, rewrite_perplexity, PerplexityRewriter
from robustness_eval.rewriters.burstiness_rewriter import burstiness_rewriter, rewrite_burstiness, BurstinessRewriter
from robustness_eval.rewriters.vocabulary_rewriter import vocabulary_rewriter, rewrite_vocabulary, VocabularyRewriter

def combined_rewrite(text: str, seed: Optional[int] = 42, tone: str = "academic") -> str:
    """
    Chains all three rewriters sequentially:
    1. Burstiness Rewriter (Restructures sentence boundaries for target tone).
    2. Vocabulary Rewriter (Substitutes overused LLM cliché vocabulary).
    3. Perplexity Rewriter (Injects natural transitions and perplexity variations).
    
    This is the primary function called by the adversarial evaluation harness and text humanizer.
    """
    # Step 1: Restructure boundaries and active voice first
    step1 = burstiness_rewriter(text, seed=seed, tone=tone)
    # Step 2: Vocabulary substitutions
    step2 = vocabulary_rewriter(step1, seed=seed)
    # Step 3: Perplexity transitions
    step3 = perplexity_rewriter(step2, seed=seed)
    return step3

__all__ = [
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
]
