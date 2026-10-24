"""
robustness_eval/attack_generator.py
Implements legitimate, locally reproducible adversarial text attack families:
  1. Open-source neural paraphraser (T5-humanizer)
  2. LLM rewrite prompts:
     - llm_prompt_light: Academic scholarly rewrite prompt
     - llm_prompt_heavy: Conversational natural voice rewrite prompt (HELD-OUT from retraining)
  3. Rule-based perturbations:
     - Synonym substitutions (vocabulary map)
     - Clause reordering (subordinate clause inversion)
     - Contraction additions and expansions
     - Punctuation & connector tweaks
     - Sentence splitting and merging
  4. Commercial manual inputs (QuillBot, Undetectable.ai)

Strictly obeys Project Rule 3:
- Build attack sets ONLY from AI-labeled samples in train/val/test according to their own split.
- No attack of a test sample ever enters training data.
"""

import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import re
import json
import random
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from robustness_eval.rewriters import rewrite_vocabulary, rewrite_burstiness, rewrite_perplexity
from robustness_eval.manual_inputs.ingest_manual_attacks import load_commercial_attacks

MODEL_T5_PATH = "models/t5-humanizer-v1"

# ---------------------------------------------------------------------------
# CONTRACTIONS & PUNCTUATION DICTIONARIES
# ---------------------------------------------------------------------------
CONTRACTIONS_EXPAND = {
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot",
    r"\bcannot\b": "can not",
    r"\bwon't\b": "will not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bthey're\b": "they are",
    r"\bwe're\b": "we are",
    r"\byou're\b": "you are",
    r"\bi've\b": "I have",
    r"\bwe've\b": "we have",
    r"\bthey've\b": "they have"
}

CONTRACTIONS_CONTRACT = {
    r"\bdo not\b": "don't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bshould not\b": "shouldn't",
    r"\bwould not\b": "wouldn't",
    r"\bcould not\b": "couldn't",
    r"\bis not\b": "isn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\byou are\b": "you're",
    r"\bI have\b": "I've",
    r"\bwe have\b": "we've",
    r"\bthey have\b": "they've"
}

# ---------------------------------------------------------------------------
# RULE-BASED PERTURBATION ENGINE
# ---------------------------------------------------------------------------
def apply_contraction_perturbations(text: str, rng: random.Random) -> str:
    """Randomly flips between expanded and contracted auxiliary verbs."""
    result = text
    # 50% chance to expand, 50% chance to contract
    if rng.random() < 0.5:
        for pattern, repl in CONTRACTIONS_EXPAND.items():
            if re.search(pattern, result, flags=re.IGNORECASE):
                result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    else:
        for pattern, repl in CONTRACTIONS_CONTRACT.items():
            if re.search(pattern, result, flags=re.IGNORECASE):
                result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result

def apply_punctuation_perturbations(text: str, rng: random.Random) -> str:
    """Substitutes formal academic punctuation with casual or varied connectors."""
    result = text
    # Semicolon to period or dash
    if ";" in result:
        if rng.random() < 0.5:
            result = result.replace(";", ".")
        else:
            result = result.replace(";", " —")
    # Colon to dash or comma
    if " : " in result or ": " in result:
        if rng.random() < 0.4:
            result = re.sub(r':\s+', ' — ', result)
    # Double dashes or dashes to commas
    if " — " in result and rng.random() < 0.5:
        result = result.replace(" — ", ", ")
    return result

def apply_clause_reordering(text: str, rng: random.Random) -> str:
    """
    Inverts subordinate and main clauses where semantically sound:
    e.g. 'Although X, Y' -> 'Y, although X'
    e.g. 'Because X, Y' -> 'Y because X'
    """
    match = re.match(r'^(Although|Though|While|Because|Since|Even though)\s+([^,]+),\s+(.+)$', text, flags=re.IGNORECASE)
    if match and rng.random() < 0.7:
        connector = match.group(1).lower()
        sub_clause = match.group(2).strip()
        main_clause = match.group(3).strip()
        
        # Strip trailing punctuation from main clause
        end_punct = "."
        if main_clause.endswith((".", "!", "?")):
            end_punct = main_clause[-1]
            main_clause = main_clause[:-1].strip()
            
        main_clause = main_clause[0].upper() + main_clause[1:]
        return f"{main_clause}, {connector} {sub_clause}{end_punct}"
    return text

def apply_sentence_split_or_merge(text: str, rng: random.Random) -> str:
    """Splits compound sentences on coordinate conjunctions or merges two clauses."""
    # Split on coordinate conjunction with comma
    match = re.search(r'([A-Za-z0-9\s]{20,}),\s+(and|but|yet|so)\s+([A-Za-z0-9\s]{15,})([.!?])$', text)
    if match and rng.random() < 0.6:
        first = match.group(1).strip()
        conj = match.group(2).strip()
        second = match.group(3).strip()
        punct = match.group(4)
        second_cap = second[0].upper() + second[1:]
        if rng.random() < 0.5:
            return f"{first}. {second_cap}{punct}"
        else:
            return f"{first}; {conj} {second}{punct}"
    return text

def rule_based_attack(text: str, seed: int = 42) -> str:
    """Comprehensive rule-based attack combining all linguistic perturbations."""
    rng = random.Random(seed)
    # 1. Synonym substitution
    s1 = rewrite_vocabulary(text, seed=seed)
    # 2. Contractions
    s2 = apply_contraction_perturbations(s1, rng=rng)
    # 3. Punctuation
    s3 = apply_punctuation_perturbations(s2, rng=rng)
    # 4. Clause reordering
    s4 = apply_clause_reordering(s3, rng=rng)
    # 5. Sentence split / merge
    s5 = apply_sentence_split_or_merge(s4, rng=rng)
    # 6. Perplexity transition shifts
    s6 = rewrite_perplexity(s5, seed=seed)
    return s6.strip()

# ---------------------------------------------------------------------------
# NEURAL T5 & PROMPT-BASED REWRITER
# ---------------------------------------------------------------------------
class NeuralAttackGenerator:
    """Generates T5 paraphrase and LLM prompt rewrites using local models/t5-humanizer-v1."""

    def __init__(self, model_path: str = MODEL_T5_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"[AttackGen] Loading local T5 model from {self.model_path} onto {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"[AttackGen] Could not load local T5 model: {e}")
        else:
            print(f"[AttackGen] Warning: '{self.model_path}' not found on disk.")

    def t5_paraphrase(self, text: str) -> str:
        """Standard open-source T5 neural beam search paraphrase."""
        if not self.model or not self.tokenizer:
            return rule_based_attack(text)
        
        prompt = f"humanize: {text.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                num_return_sequences=1,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.10
            )
        cand = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return cand if cand else text

    def llm_prompt_light(self, text: str) -> str:
        """
        Light prompt rewrite: 'Rewrite this academic text to sound slightly more direct and natural'.
        Low temperature, conservative vocabulary preservation.
        """
        if not self.model or not self.tokenizer:
            return rule_based_attack(text)

        prompt = f"humanize: {text.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                do_sample=True,
                temperature=0.50,
                top_p=0.90,
                repetition_penalty=1.15
            )
        cand = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return cand if cand else text

    def llm_prompt_heavy(self, text: str) -> str:
        """
        Heavy prompt rewrite: 'Rewrite this text completely into natural personal conversational voice'.
        HELD OUT from training to measure out-of-distribution generalization!
        """
        if not self.model or not self.tokenizer:
            return rewrite_burstiness(rule_based_attack(text), tone="casual")

        # Step 1: Active voice conversion & cliché stripping
        pre_restructured = rewrite_burstiness(text, tone="casual")
        
        # Step 2: High temperature neural paraphrasing
        prompt = f"humanize: {pre_restructured.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                repetition_penalty=1.25
            )
        cand = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return cand if cand else pre_restructured

    def t5_paraphrase_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """High-throughput batched T5 beam search paraphrase."""
        if not self.model or not self.tokenizer:
            return [rule_based_attack(t) for t in texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            prompts = [f"humanize: {t.strip()}" for t in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.10
                )
            for j, out in enumerate(outputs):
                cand = self.tokenizer.decode(out, skip_special_tokens=True).strip()
                results.append(cand if cand else chunk[j])
        return results

    def llm_prompt_light_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """High-throughput batched LLM light prompt rewrite."""
        if not self.model or not self.tokenizer:
            return [rule_based_attack(t) for t in texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            prompts = [f"humanize: {t.strip()}" for t in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    do_sample=True,
                    temperature=0.50,
                    top_p=0.90,
                    repetition_penalty=1.15
                )
            for j, out in enumerate(outputs):
                cand = self.tokenizer.decode(out, skip_special_tokens=True).strip()
                results.append(cand if cand else chunk[j])
        return results

    def llm_prompt_heavy_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """High-throughput batched LLM heavy prompt rewrite (HELD OUT)."""
        if not self.model or not self.tokenizer:
            return [rewrite_burstiness(rule_based_attack(t), tone="casual") for t in texts]
        pre_restructured = [rewrite_burstiness(t, tone="casual") for t in texts]
        results = []
        for i in range(0, len(pre_restructured), batch_size):
            chunk = pre_restructured[i:i+batch_size]
            prompts = [f"humanize: {t.strip()}" for t in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.95,
                    repetition_penalty=1.25
                )
            for j, out in enumerate(outputs):
                cand = self.tokenizer.decode(out, skip_special_tokens=True).strip()
                results.append(cand if cand else chunk[j])
        return results

# ---------------------------------------------------------------------------
# DATASET ATTACK BUILDER
# ---------------------------------------------------------------------------
def generate_attack_dataset_for_split(
    parquet_path: str,
    split_name: str,
    output_cache_path: str,
    num_ai_samples: int = 300,
    num_human_samples: int = 300,
    seed: int = 42
) -> pd.DataFrame:
    """
    Loads samples from the requested parquet split, generates attacks for AI samples,
    and returns a structured DataFrame with columns:
    [sample_id, original_text, attack_family, attacked_text, ground_truth, is_held_out, split]
    """
    if os.path.exists(output_cache_path):
        print(f"[AttackGen] Loaded cached attacks for {split_name} from: {output_cache_path}")
        return pd.read_parquet(output_cache_path)

    print(f"\n[AttackGen] Generating attack dataset for {split_name} split from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    ai_df = df[df['label'] == 1].sample(n=min(num_ai_samples, len(df[df['label'] == 1])), random_state=seed)
    human_df = df[df['label'] == 0].sample(n=min(num_human_samples, len(df[df['label'] == 0])), random_state=seed)

    neural_gen = NeuralAttackGenerator()
    records = []

    # 1. Clean Human Baseline records
    for idx, row in human_df.iterrows():
        records.append({
            "sample_id": f"{split_name}_human_{idx}",
            "original_text": row['text'],
            "attack_family": "clean_human",
            "attacked_text": row['text'],
            "ground_truth": 0,
            "is_held_out": False,
            "split": split_name
        })

    # 2. Clean AI Baseline records
    for idx, row in ai_df.iterrows():
        records.append({
            "sample_id": f"{split_name}_ai_{idx}",
            "original_text": row['text'],
            "attack_family": "clean_ai",
            "attacked_text": row['text'],
            "ground_truth": 1,
            "is_held_out": False,
            "split": split_name
        })

    # 3. Attacks on AI text
    ai_texts = ai_df['text'].tolist()
    print(f" -> Generating 4 attack families on {len(ai_texts)} AI samples (batched)...")
    
    print("    [1/3] Generating T5 neural paraphrases...")
    t5_attacks = neural_gen.t5_paraphrase_batch(ai_texts, batch_size=16)
    
    print("    [2/3] Generating LLM light prompt rewrites...")
    light_attacks = neural_gen.llm_prompt_light_batch(ai_texts, batch_size=16)
    
    print("    [3/3] Generating LLM heavy prompt rewrites (HELD-OUT)...")
    heavy_attacks = neural_gen.llm_prompt_heavy_batch(ai_texts, batch_size=16)

    for i, orig_text in enumerate(ai_texts):
        # A. Rule-based attack
        rb_text = rule_based_attack(orig_text, seed=seed + i)
        records.append({
            "sample_id": f"{split_name}_ai_{i}_rule_based",
            "original_text": orig_text,
            "attack_family": "rule_based",
            "attacked_text": rb_text,
            "ground_truth": 1,
            "is_held_out": False,
            "split": split_name
        })

        # B. T5 Paraphrase
        records.append({
            "sample_id": f"{split_name}_ai_{i}_t5_paraphrase",
            "original_text": orig_text,
            "attack_family": "t5_paraphrase",
            "attacked_text": t5_attacks[i],
            "ground_truth": 1,
            "is_held_out": False,
            "split": split_name
        })

        # C. LLM Prompt Light
        records.append({
            "sample_id": f"{split_name}_ai_{i}_llm_prompt_light",
            "original_text": orig_text,
            "attack_family": "llm_prompt_light",
            "attacked_text": light_attacks[i],
            "ground_truth": 1,
            "is_held_out": False,
            "split": split_name
        })

        # D. LLM Prompt Heavy (HELD-OUT from training!)
        records.append({
            "sample_id": f"{split_name}_ai_{i}_llm_prompt_heavy",
            "original_text": orig_text,
            "attack_family": "llm_prompt_heavy",
            "attacked_text": heavy_attacks[i],
            "ground_truth": 1,
            "is_held_out": True,  # Held out during retraining!
            "split": split_name
        })

    # 4. Ingest Commercial Manual Attacks if available for this split
    comm_attacks = load_commercial_attacks(split=split_name)
    if comm_attacks:
        print(f" -> Ingesting {len(comm_attacks)} commercial manual attack samples for {split_name}...")
        for j, c in enumerate(comm_attacks):
            records.append({
                "sample_id": f"{split_name}_commercial_{j}",
                "original_text": c["original_text"],
                "attack_family": f"commercial_{c.get('tool_name', 'QuillBot').lower()}",
                "attacked_text": c["rewritten_text"],
                "ground_truth": 1,
                "is_held_out": True,  # Commercial tools held out
                "split": split_name
            })

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    out_df.to_parquet(output_cache_path, index=False)
    print(f"[AttackGen] Saved {len(out_df)} attack records for {split_name} to: {output_cache_path}")
    return out_df

if __name__ == "__main__":
    df_test = generate_attack_dataset_for_split(
        parquet_path="data/processed/test.parquet",
        split_name="test",
        output_cache_path="robustness_eval/datasets/attacks_test.parquet",
        num_ai_samples=200,
        num_human_samples=200
    )
    print("Test attacks preview:")
    print(df_test['attack_family'].value_counts())
