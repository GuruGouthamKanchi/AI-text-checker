import re
import random
from typing import Optional

class PerplexityRewriter:
    """
    Substitutes predictable word choices and transition phrases with less-predictable,
    meaning-preserving alternatives to boost lexical perplexity.
    """

    UNPREDICTABLE_CLAUSES = [
        "specifically,",
        "in practice,",
        "overall,",
        "in this setup,",
        "to achieve this,",
        "as a result,",
        "notably,",
    ]

    PREDICTABLE_TRANSITIONS = {
        "furthermore": "in addition",
        "moreover": "alongside this",
        "consequently": "resulting in",
        "therefore": "thus",
        "however": "though",
        "in conclusion": "overall",
        "in summary": "in short",
        "specifically": "in detail",
    }

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def rewrite(self, text: str, boost_factor: float = 1.25) -> str:
        """
        Rewrites text to boost perplexity by inserting rare clauses and replacing predictable transitions.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return text

        rewritten_sentences = []

        for i, sent in enumerate(sentences):
            modified_sent = sent

            # Substitute predictable transitions
            for orig, var in self.PREDICTABLE_TRANSITIONS.items():
                pattern = re.compile(rf'\b{orig}\b', re.IGNORECASE)
                if pattern.search(modified_sent) and self.rng.random() < 0.75:
                    modified_sent = pattern.sub(var, modified_sent)

            rewritten_sentences.append(modified_sent)

        return " ".join(rewritten_sentences)

def perplexity_rewriter(text: str, seed: Optional[int] = 42) -> str:
    """
    Primary function for perplexity rewriting taking (text: str) -> str.
    """
    rewriter = PerplexityRewriter(seed=seed)
    return rewriter.rewrite(text)

def rewrite_perplexity(text: str, boost_factor: float = 1.25, seed: Optional[int] = 42) -> str:
    rewriter = PerplexityRewriter(seed=seed)
    return rewriter.rewrite(text, boost_factor=boost_factor)
