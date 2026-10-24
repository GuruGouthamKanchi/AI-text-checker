"""
eval/run_ood_retraining_colab.py
Google Colab / Multi-GPU Training Script for Leave-One-Generator-Out (LOGO) and
Leave-One-Domain-Out (LODO) Out-of-Distribution Retraining.

Evaluates detector generalization to unseen AI generators (Claude, Gemini, GPT-5, Llama, ChatGPT)
and unseen domains (Academic, Essay, Q&A) across 3 random seeds (42, 43, 44).
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
GENERATORS = ["claude", "gemini", "gpt-5", "llama", "chatgpt"]
DOMAINS = ["academic", "essay", "qa"]

def match_generator(source_str: str) -> str:
    s = str(source_str).lower()
    if "claude" in s:
        return "claude"
    if "gemini" in s:
        return "gemini"
    if "gpt-5" in s or "gpt5" in s:
        return "gpt-5"
    if "llama" in s:
        return "llama"
    if "chatgpt" in s or "chat_gpt" in s:
        return "chatgpt"
    return "other"

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "label": self.labels[idx]
        }

def train_and_eval_ood(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_col: str,
    holdout_val: str,
    base_model_dir: str = "models/roberta-sentence-academic-v2",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Filter training data: exclude the held-out generator or domain
    train_split = train_df[train_df[holdout_col] != holdout_val].copy()
    test_target = test_df[test_df[holdout_col] == holdout_val].copy()
    test_human = test_df[test_df['label'] == 0].copy()

    print(f" -> Training: {len(train_split)} samples (excluding '{holdout_val}') | Target OOD Test: {len(test_target)} samples")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=2).to(device)

    train_ds = TextClassificationDataset(train_split['text'].tolist(), train_split['label'].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = ce_loss(logits, lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"    Epoch {ep+1}/{epochs} loss: {running_loss/len(train_loader):.4f} ({time.time()-t0:.1f}s)")

    # Evaluation on held-out target
    model.eval()
    def get_probs(df_eval):
        texts = df_eval['text'].tolist()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = texts[i:i+64]
                inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                l = model(**inp).logits
                p = torch.softmax(l, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    target_probs = get_probs(test_target)
    human_probs = get_probs(test_human)

    threshold = 0.58
    target_recall = float(np.mean(target_probs >= threshold))
    human_fpr = float(np.mean(human_probs >= threshold))

    return {
        "holdout_type": holdout_col,
        "holdout_target": holdout_val,
        "seed": seed,
        "ood_recall": target_recall,
        "human_fpr": human_fpr,
        "target_samples": len(test_target)
    }

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out OOD Retraining Suite")
    parser.add_argument("--mode", choices=["generator", "domain", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print("=" * 80)
    print(" OUT-OF-DISTRIBUTION (OOD) RETRAINING PIPELINE")
    print("=" * 80)

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    train_df["gen_group"] = train_df["source"].apply(match_generator)
    test_df["gen_group"] = test_df["source"].apply(match_generator)

    results = []

    if args.mode in ["generator", "both"]:
        print("\n--- Running Leave-One-Generator-Out (LOGO) ---")
        for gen in GENERATORS:
            print(f"\n[LOGO] Holding out Generator: {gen.upper()}")
            for s in args.seeds:
                res = train_and_eval_ood(train_df, test_df, "gen_group", gen, seed=s, epochs=args.epochs)
                results.append(res)

    if args.mode in ["domain", "both"]:
        print("\n--- Running Leave-One-Domain-Out (LODO) ---")
        for dom in DOMAINS:
            print(f"\n[LODO] Holding out Domain: {dom.upper()}")
            for s in args.seeds:
                res = train_and_eval_ood(train_df, test_df, "domain", dom, seed=s, epochs=args.epochs)
                results.append(res)

    out_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_df.to_csv("results/ood_retrain_results.csv", index=False)
    print(f"\n[Complete] Saved all OOD retraining metrics to results/ood_retrain_results.csv")

if __name__ == "__main__":
    main()
