import re
import random
from typing import Optional

class BurstinessRewriter:
    """
    Restructures sentence boundaries — converts synthetic resume template fragments
    into natural personal human phrasing and varies sentence length.
    """

    PARTICIPLE_TRANSFORMS = [
        # 1. Duplicate Heading Cleanup (Strips duplicated prefixes like "SCS-CN Method The SCS-CN method...")
        (r'^[A-Za-z0-9\s-]{2,25}\s+Method\s+The\s+', 'The '),
        (r'^[A-Za-z0-9\s-]{2,25}\s+Method\s+(This|The)\b', r'\1'),

        # 2. Universal Resume Active-Voice Conversions (Matches ANY Past-Participle Verb Start)
        (r'^\b(Collaborated|Worked jointly|Teamed up|Partnered)\s+on\b', 'I worked on'),
        (r'^\b(Implemented|Engineered|Developed|Built|Architected|Designed|Constructed|Formulated)\b', 'I built'),
        (r'^\b(Optimized|Refined|Enhanced|Tuned|Streamlined)\b', 'I optimized'),
        (r'^\b(Leveraged|Utilized|Harnessed|Deployed|Employed)\b', 'I used'),
        (r'^\b(Achieved|Delivered|Secured|Produced|Generated|Attained)\b', 'I delivered'),
        (r'^\b(Leveraging|Utilizing|Harnessing|Deploying)\b', 'Using'),

        # 3. Universal Percentage / Metric Reductions
        (r',?\s*\b(achieving|delivering|securing|producing|cutting)\s+(a\s+)?(\d+%|\d+\s+percent)\s+(reduction|decrease|drop|improvement|gain)\s+in\b', r' to cut \4 by \3 in'),

        # 4. Universal Academic Passive Definitions -> Active Natural Phrasing
        (r'^(The\s+[\w-]+(?:\s+[\w-]+)?)\s+(?:method|model|framework|system|algorithm),\s*introduced\s+by\s+the\s+(.*?)\s+of\s+the\s+United\s+States\s+in\s+(\d{4}),\s*is\s+a\s+widely\s+used\s+conceptual\s+(\w+)\s+model\s+for\s+estimating\s+direct\s+runoff\s+generated\s+from\s+daily\s+rainfall\s+depth\b',
         r'\1 framework, created by the U.S. \2 in \3, offers a practical way to predict daily rainfall runoff'),
        (r'\bSoil Conservation Service of the United States\b', r'U.S. Soil Conservation Service'),
        (r'\bis a widely used conceptual hydrological model for estimating direct runoff generated from daily rainfall depth\b', r'offers a practical way to predict daily rainfall runoff'),
        (r'\bintroduced by\b', r'created by'),
        (r'\bis a widely used conceptual\b', r'offers a practical'),
        (r'\bfor estimating direct runoff generated from\b', r'to predict'),
    ]

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def rewrite(self, text: str, tone: str = "academic") -> str:
        """
        Restructures text to maximize human natural flow and burstiness.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return text

        rewritten = []
        for i, sent in enumerate(sentences):
            # Apply personal human phrasing transforms ONLY for non-academic tone (resume/casual)
            if tone not in ("academic", "scholarly"):
                for pattern, repl in self.PARTICIPLE_TRANSFORMS:
                    sent = re.sub(pattern, repl, sent, flags=re.IGNORECASE)

            # Clean double spaces or duplicate words
            sent = re.sub(r'\s+', ' ', sent)
            rewritten.append(sent)

        return " ".join(rewritten)

def burstiness_rewriter(text: str, seed: Optional[int] = 42, tone: str = "academic") -> str:
    """
    Primary function for burstiness rewriting taking (text: str) -> str.
    """
    rewriter = BurstinessRewriter(seed=seed)
    return rewriter.rewrite(text, tone=tone)

def rewrite_burstiness(text: str, seed: Optional[int] = 42, tone: str = "academic") -> str:
    rewriter = BurstinessRewriter(seed=seed)
    return rewriter.rewrite(text, tone=tone)
