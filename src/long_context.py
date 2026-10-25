"""
src/long_context.py
Document-Level Native Long-Context Scoring Engine for ModernBERT (up to 8,192 tokens).

Strictly complies with Project Rules 1, 2, 5, 7, 10:
- Feeds full document text (up to 8,192 tokens natively) to models/modernbert-academic.
- Applies chunking ONLY when input exceeds 8,192 tokens.
- Fails loudly if model weights are missing when STRICT_PRODUCTION=1 or REQUIRE_MODELS=1 (Rule 7).
- Supports Windows 10 RTX 2050 (4GB) with automatic memory management and CPU fallback (Rule 10).
"""

import os
import sys
import re
import math
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np

# Default Model Directories
MODERNBERT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "modernbert-academic")
ROBERTA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "roberta-sentence-academic-v2")

class LongContextModelManager:
    """Singleton manager for loading and caching long-context and baseline models."""
    _instance = None
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mb_model = None
        self.mb_tokenizer = None
        self.rob_model = None
        self.rob_tokenizer = None
        self.loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LongContextModelManager()
        return cls._instance

    def load_models(self, mb_dir: str = MODERNBERT_DIR, rob_dir: str = ROBERTA_DIR):
        """Loads ModernBERT and RoBERTa models from disk."""
        if self.loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # 1. Load ModernBERT
        mb_exists = os.path.exists(mb_dir) and os.path.exists(os.path.join(mb_dir, "config.json"))
        if not mb_exists:
            if os.getenv("STRICT_PRODUCTION", "0") == "1" or os.getenv("REQUIRE_MODELS", "0") == "1":
                raise FileNotFoundError(f"[Fatal] ModernBERT model not found at '{mb_dir}' (Project Rule 7).")
            print(f"[Warning] ModernBERT weights not found at '{mb_dir}'.")
        else:
            try:
                self.mb_tokenizer = AutoTokenizer.from_pretrained(mb_dir, local_files_only=True)
                self.mb_model = AutoModelForSequenceClassification.from_pretrained(mb_dir, local_files_only=True).to(self.device)
                self.mb_model.eval()
                print(f"[LongContext] Loaded ModernBERT from '{mb_dir}' on {self.device}.")
            except Exception as e:
                print(f"[LongContext] Error loading ModernBERT: {e}")

        # 2. Load RoBERTa
        rob_exists = os.path.exists(rob_dir) and os.path.exists(os.path.join(rob_dir, "config.json"))
        if not rob_exists:
            if os.getenv("STRICT_PRODUCTION", "0") == "1" or os.getenv("REQUIRE_MODELS", "0") == "1":
                raise FileNotFoundError(f"[Fatal] RoBERTa model not found at '{rob_dir}' (Project Rule 7).")
            print(f"[Warning] RoBERTa weights not found at '{rob_dir}'.")
        else:
            try:
                self.rob_tokenizer = AutoTokenizer.from_pretrained(rob_dir, local_files_only=True)
                self.rob_model = AutoModelForSequenceClassification.from_pretrained(rob_dir, local_files_only=True).to(self.device)
                self.rob_model.eval()
                print(f"[LongContext] Loaded RoBERTa from '{rob_dir}' on {self.device}.")
            except Exception as e:
                print(f"[LongContext] Error loading RoBERTa: {e}")

        self.loaded = True


def count_document_tokens(text: str, tokenizer=None) -> int:
    """Counts the exact token count using tokenizer if available, else standard ratio."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, truncation=False))
        except Exception:
            pass
    return int(len(text.split()) * 1.33)


def score_document_full_modernbert_8k(
    text: str, 
    max_context: int = 8192, 
    chunk_overlap: int = 512,
    manager: Optional[LongContextModelManager] = None
) -> Dict[str, Any]:
    """
    Feeds full document text (up to 8,192 tokens natively) to ModernBERT.
    Applies sliding window chunking ONLY when input exceeds 8,192 tokens.
    
    Returns:
        dict with keys: 'score', 'chunk_scores', 'num_tokens', 'num_chunks', 'is_chunked'
    """
    mgr = manager or LongContextModelManager.get_instance()
    if not mgr.loaded:
        mgr.load_models()

    if mgr.mb_model is None or mgr.mb_tokenizer is None:
        raise RuntimeError("ModernBERT model/tokenizer is not available for full-context scoring.")

    total_tokens = count_document_tokens(text, mgr.mb_tokenizer)

    # 1. Native Full-Context Path (<= 8,192 tokens)
    if total_tokens <= max_context:
        with torch.inference_mode():
            inputs = mgr.mb_tokenizer(text, truncation=True, max_length=max_context, return_tensors="pt")
            device = mgr.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                logits = mgr.mb_model(**inputs).logits
            except torch.cuda.OutOfMemoryError:
                # Rule 10: Windows 10 RTX 2050 (4GB) safety fallback to CPU if 8k attention spikes VRAM
                print("[LongContext] CUDA OOM during 8k attention forward pass; falling back to CPU.")
                torch.cuda.empty_cache()
                inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                mgr.mb_model.to("cpu")
                logits = mgr.mb_model(**inputs_cpu).logits
                mgr.mb_model.to(device)
            
            prob_ai = float(torch.softmax(logits, dim=-1).cpu().numpy()[0][1])

        return {
            "score": prob_ai,
            "chunk_scores": [prob_ai],
            "num_tokens": total_tokens,
            "num_chunks": 1,
            "is_chunked": False
        }

    # 2. Chunked Path (> 8,192 tokens)
    # Tokenize full input IDs without truncation to split strictly by token boundaries
    all_input_ids = mgr.mb_tokenizer.encode(text, truncation=False)
    stride = max_context - chunk_overlap
    chunk_scores = []

    with torch.inference_mode():
        for start_idx in range(0, len(all_input_ids), stride):
            chunk_ids = all_input_ids[start_idx : start_idx + max_context]
            if len(chunk_ids) < 32 and chunk_scores:
                # Skip tiny residual fragments
                break
            
            input_tensor = torch.tensor([chunk_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_tensor)
            inputs = {"input_ids": input_tensor.to(mgr.device), "attention_mask": attention_mask.to(mgr.device)}

            try:
                logits = mgr.mb_model(**inputs).logits
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                mgr.mb_model.to("cpu")
                logits = mgr.mb_model(**inputs_cpu).logits
                mgr.mb_model.to(mgr.device)

            prob = float(torch.softmax(logits, dim=-1).cpu().numpy()[0][1])
            chunk_scores.append(prob)

    # Document-level aggregation across 8k chunks: top-1 max pool and mean pool
    doc_score = float(np.mean(chunk_scores))

    return {
        "score": doc_score,
        "chunk_scores": chunk_scores,
        "max_chunk_score": float(np.max(chunk_scores)),
        "mean_chunk_score": doc_score,
        "num_tokens": len(all_input_ids),
        "num_chunks": len(chunk_scores),
        "is_chunked": True
    }


def score_document_sliding_roberta_512(
    text: str, 
    max_length: int = 512, 
    stride: int = 256,
    manager: Optional[LongContextModelManager] = None
) -> Dict[str, Any]:
    """
    Evaluates document using RoBERTa with a standard 512-token sliding window across the entire document.
    """
    mgr = manager or LongContextModelManager.get_instance()
    if not mgr.loaded:
        mgr.load_models()

    if mgr.rob_model is None or mgr.rob_tokenizer is None:
        raise RuntimeError("RoBERTa model/tokenizer is not available for sliding-window scoring.")

    all_input_ids = mgr.rob_tokenizer.encode(text, truncation=False)
    total_tokens = len(all_input_ids)

    if total_tokens <= max_length:
        with torch.inference_mode():
            inputs = mgr.rob_tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(mgr.device) for k, v in inputs.items()}
            logits = mgr.rob_model(**inputs).logits
            prob = float(torch.softmax(logits, dim=-1).cpu().numpy()[0][1])
        return {
            "score": prob,
            "chunk_scores": [prob],
            "num_tokens": total_tokens,
            "num_chunks": 1
        }

    chunk_scores = []
    with torch.inference_mode():
        for start_idx in range(0, total_tokens, stride):
            chunk_ids = all_input_ids[start_idx : start_idx + max_length]
            if len(chunk_ids) < 32 and chunk_scores:
                break
            
            input_tensor = torch.tensor([chunk_ids], dtype=torch.long).to(mgr.device)
            attention_mask = torch.ones_like(input_tensor).to(mgr.device)
            inputs = {"input_ids": input_tensor, "attention_mask": attention_mask}
            
            logits = mgr.rob_model(**inputs).logits
            prob = float(torch.softmax(logits, dim=-1).cpu().numpy()[0][1])
            chunk_scores.append(prob)

    # Standard sliding window aggregation: mean and top-k pool
    mean_score = float(np.mean(chunk_scores))
    top_k_score = float(np.mean(sorted(chunk_scores, reverse=True)[:min(3, len(chunk_scores))]))

    return {
        "score": mean_score,
        "top_k_score": top_k_score,
        "chunk_scores": chunk_scores,
        "num_tokens": total_tokens,
        "num_chunks": len(chunk_scores)
    }


def score_document_3sent_pipeline(
    text: str,
    manager: Optional[LongContextModelManager] = None
) -> Dict[str, Any]:
    """
    Evaluates document using the existing production 3-sentence sliding window pipeline from src/highlighter.py.
    """
    mgr = manager or LongContextModelManager.get_instance()
    if not mgr.loaded:
        mgr.load_models()

    from src.highlighter import split_into_sentences, run_predictions

    sentences = split_into_sentences(text)
    if not sentences:
        return {"score": 0.02, "num_sentences": 0, "sentence_scores": []}

    preds, is_sim = run_predictions(
        sentences,
        preloaded_model=mgr.mb_model,
        preloaded_tokenizer=mgr.mb_tokenizer,
        preloaded_roberta_model=mgr.rob_model,
        preloaded_roberta_tokenizer=mgr.rob_tokenizer
    )

    scores = [p["score"] for p in preds]
    mean_score = float(np.mean(scores))
    top_3_score = float(np.mean(sorted(scores, reverse=True)[:min(3, len(scores))]))

    return {
        "score": mean_score,
        "top_3_score": top_3_score,
        "num_sentences": len(sentences),
        "sentence_scores": scores,
        "is_simulated": is_sim
    }


def score_document_combination(
    score_full_mb_8k: float,
    score_sliding_rob_512: float,
    weight_mb: float = 0.50,
    weight_rob: float = 0.50
) -> float:
    """Combines full-context ModernBERT score with sliding-window RoBERTa score."""
    return float(weight_mb * score_full_mb_8k + weight_rob * score_sliding_rob_512)
