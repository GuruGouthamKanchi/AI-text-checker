import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class AdversarialPair:
    """
    Storage schema for a single (original, rewritten) adversarial text pair.
    """
    original_text: str = ""
    rewritten_text: str = ""
    source_label: str = "ai"               # "ai" or "human"
    rewrite_method: str = "combined_v1"     # "combined_v1", "perplexity", etc.
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pair_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_ai_score: float = 0.0
    rewritten_ai_score: float = 0.0
    score_drop: float = 0.0
    evaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        original_text: str = "",
        rewritten_text: str = "",
        source_label: str = "ai",
        rewrite_method: str = "combined_v1",
        created_at: Optional[str] = None,
        pair_id: Optional[str] = None,
        original_ai_score: float = 0.0,
        rewritten_ai_score: float = 0.0,
        score_drop: float = 0.0,
        evaded: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        transformation_type: Optional[str] = None
    ):
        self.original_text = original_text
        self.rewritten_text = rewritten_text
        self.source_label = source_label
        self.rewrite_method = transformation_type if transformation_type is not None else rewrite_method
        self.created_at = created_at if created_at is not None else datetime.now(timezone.utc).isoformat()
        self.pair_id = pair_id if pair_id is not None else str(uuid.uuid4())
        self.original_ai_score = original_ai_score
        self.rewritten_ai_score = rewritten_ai_score
        self.score_drop = score_drop
        self.evaded = evaded
        self.metadata = metadata if metadata is not None else {}

    @property
    def transformation_type(self) -> str:
        return self.rewrite_method

    @transformation_type.setter
    def transformation_type(self, val: str) -> None:
        self.rewrite_method = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "source_label": self.source_label,
            "rewrite_method": self.rewrite_method,
            "created_at": self.created_at,
            "pair_id": self.pair_id,
            "original_ai_score": self.original_ai_score,
            "rewritten_ai_score": self.rewritten_ai_score,
            "score_drop": self.score_drop,
            "evaded": self.evaded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdversarialPair':
        return cls(**data)

@dataclass
class AdversarialDataset:
    """
    Collection of AdversarialPair items with JSON and JSONL serialization utilities.
    """
    pairs: List[AdversarialPair] = field(default_factory=list)

    def add_pair(self, pair: AdversarialPair) -> None:
        self.pairs.append(pair)

    def save_to_jsonl(self, filepath: str) -> None:
        """Saves pairs as JSON Lines (.jsonl)."""
        with open(filepath, "w", encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_from_jsonl(cls, filepath: str) -> 'AdversarialDataset':
        """Loads pairs from JSON Lines (.jsonl)."""
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pairs.append(AdversarialPair.from_dict(data))
        return cls(pairs=pairs)

    def save_to_json(self, filepath: str) -> None:
        data = [p.to_dict() for p in self.pairs]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'AdversarialDataset':
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [AdversarialPair.from_dict(d) for d in data]
        return cls(pairs=pairs)
