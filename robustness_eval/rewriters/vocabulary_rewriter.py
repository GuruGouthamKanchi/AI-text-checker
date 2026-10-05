import re
import random
from typing import Optional

class VocabularyRewriter:
    """
    Detects and replaces overused LLM academic terms and phrases with natural human alternatives,
    preserving exact sentence semantics and LaTeX lock tokens.
    """

    PHRASE_MAP = [
        (r'\bdelve\s+into\b', 'explore'),
        (r'\bdelves\s+into\b', 'explores'),
        (r'\bdelving\s+into\b', 'exploring'),
        (r'\bit\s+is\s+important\s+to\s+note\s+that\b', 'notably,'),
        (r'\bit\s+is\s+worth\s+emphasizing\s+that\b', 'notably,'),
        (r'\bit\s+is\s+crucial\s+to\b', 'we must'),
        (r'\bserve(?:s)?\s+as\s+a\s+testament\s+to\b', 'demonstrates'),
        (r'\btransformative\s+potential\s+of\b', 'impact of'),
        (r'\bin\s+the\s+rapidly\s+evolving\s+landscape\s+of\b', 'in modern'),
        (r'\bplay(?:s)?\s+a\s+pivotal\s+role\s+in\b', 'is key to'),
        (r'\bturned\s+their\s+attention\s+toward\b', 'focused on'),
        (r'\bpave(?:s)?\s+the\s+way\s+for\b', 'enables'),
        (r'\bvital\s+importance\s+of\b', 'importance of'),
        (r'\bcontinuous\s+architectural\s+refinement\b', 'architectural optimization'),
        (r'\btraditional\s+gradient\s+descent\s+methodologies\s+often\s+suffer\s+from\b', 'standard gradient descent methods often hit'),
        (r'\bheterogeneous\s+empirical\s+datasets\b', 'diverse benchmark datasets'),
        (r'\badopting\s+adaptive\s+optimization\s+techniques\s+becomes\s+paramount\b', 'adaptive optimization is essential'),
        (r'\bexhibit\s+remarkable\s+capabilities\b', 'show strong performance'),
        (r'\bacross\s+diverse\s+operational\s+domains\b', 'across multiple tasks'),
        (r'\bcomprehensive\s+framework\b', 'thorough framework'),
        (r'\bsynergistic\s+optimization\s+pipeline\b', 'unified optimization pipeline'),
        (r'\bmultifaceted\s+challenges\b', 'complex challenges'),
        (r'\bartificial\s+intelligence\s+alignment\b', 'AI alignment'),
        (r'\bfoster\s+responsible\s+development\b', 'support safe development'),
    ]

    VOCAB_MAP = {
        "delve": ["examine", "investigate", "explore"],
        "delves": ["examines", "investigates", "explores"],
        "delving": ["examining", "investigating", "exploring"],
        "leverage": ["use", "apply", "harness"],
        "leverages": ["uses", "applies", "harnesses"],
        "leveraging": ["using", "applying", "harnessing"],
        "comprehensive": ["thorough", "detailed", "extensive"],
        "crucial": ["vital", "essential", "key"],
        "testament": ["proof", "evidence"],
        "tapestry": ["mix", "structure"],
        "multifaceted": ["complex", "diverse"],
        "underscore": ["highlight", "emphasize"],
        "underscores": ["highlights", "emphasizes"],
        "paramount": ["vital", "critical"],
        "meticulous": ["thorough", "detailed"],
        "meticulously": ["thoroughly", "carefully"],
        "foster": ["support", "promote"],
        "fosters": ["supports", "promotes"],
        "pivotal": ["key", "central"],
        "seamlessly": ["smoothly", "naturally"],
    }

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def rewrite(self, text: str, substitution_rate: float = 0.15) -> str:
        """
        Substitutes overused LLM terms and phrases with varied natural human alternatives.
        """
        result = text

        # Step 1: Multi-word phrase substitutions
        for pattern, replacement in self.PHRASE_MAP:
            def repl_func(m):
                matched = m.group(0)
                if matched[0].isupper():
                    return replacement.capitalize()
                return replacement
            result = re.sub(pattern, repl_func, result, flags=re.IGNORECASE)

        # Step 2: Single word vocabulary substitutions
        words = result.split()
        rewritten_words = []

        for word in words:
            # Skip LaTeX lock placeholders
            if "__TEX_LOCK_" in word or "__LOCK_" in word:
                rewritten_words.append(word)
                continue

            match = re.match(r'^([^\w]*)([\w-]+)([^\w]*)$', word)
            if not match:
                rewritten_words.append(word)
                continue

            prefix, clean_word, suffix = match.groups()
            clean_lower = clean_word.lower()

            if clean_lower in self.VOCAB_MAP:
                candidates = self.VOCAB_MAP[clean_lower]
                synonym = self.rng.choice(candidates)
                if clean_word[0].isupper():
                    synonym = synonym.capitalize()
                rewritten_words.append(f"{prefix}{synonym}{suffix}")
            else:
                rewritten_words.append(word)

        result = " ".join(rewritten_words)

        # Clean grammar artifacts & double transition/preposition errors
        result = re.sub(r'\b(explore|examine|investigate)\s+into\b', r'\1', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(Additionally|Furthermore|Moreover|In addition|Alongside this),\s*(notably|importantly),\b', r'\2,', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()

        # Enforce initial capitalization for sentence starters
        result = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)

        return result

def vocabulary_rewriter(text: str, seed: Optional[int] = 42) -> str:
    rewriter = VocabularyRewriter(seed=seed)
    return rewriter.rewrite(text)

def rewrite_vocabulary(text: str, substitution_rate: float = 0.15, seed: Optional[int] = 42) -> str:
    rewriter = VocabularyRewriter(seed=seed)
    return rewriter.rewrite(text, substitution_rate=substitution_rate)

