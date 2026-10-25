"""
eval/run_ood_leave_one_out_colab.py
Google Colab / Multi-GPU Unconstrained Training Script for Leave-One-Generator-Out (LOGO)
and Leave-One-Domain-Out (LODO) Out-of-Distribution Retraining.

Designed for Google Colab (T4, V100, A100 GPUs) to retrain roberta-base across all groups
and random seeds (42, 43, 44) on the full unconstrained dataset.
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import core evaluation functions from run_ood_leave_one_out
from eval.run_ood_leave_one_out import (
    GENERATORS,
    DOMAINS,
    SEEDS,
    map_generator,
    load_datasets,
    prepare_training_sample,
    train_single_run,
    generate_reports,
    CHECKPOINT_PATH
)


def main():
    parser = argparse.ArgumentParser(description="Google Colab Leave-One-Out OOD Retraining Suite")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="-1 for unconstrained full dataset")
    parser.add_argument("--groups", nargs="+", default=["all"], help="Groups to hold out or 'all'")
    args = parser.parse_args()

    print("=" * 80)
    print(" GOOGLE COLAB / HIGH-COMPUTE LEAVE-ONE-OUT RETRAINING SUITE")
    print(f" Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Max Training Samples: {'Unconstrained (Full Data)' if args.max_train_samples < 0 else args.max_train_samples}")
    print(f" Seeds: {args.seeds} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    train_df, val_df, test_df = load_datasets()

    checkpoint_data = {"in_dist": [], "runs": []}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"[Checkpoint] Resuming from {len(checkpoint_data.get('runs', []))} completed runs")
        except Exception as e:
            print(f"[Checkpoint] Warning: {e}")

    # In-distribution baseline runs
    in_dist_completed = {r["seed"] for r in checkpoint_data["in_dist"]}
    for s in args.seeds:
        if s not in in_dist_completed:
            sample_df = prepare_training_sample(train_df, "in_distribution", "none", args.max_train_samples, s)
            res = train_single_run(
                sample_df, val_df, test_df, "in_distribution", "none",
                seed=s, epochs=args.epochs, batch_size=args.batch_size
            )
            checkpoint_data["in_dist"].append(res)
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

    # Leave-one-out targets
    all_targets = [("generator", g) for g in GENERATORS] + [("domain", d) for d in DOMAINS]
    completed_pairs = {(r["holdout_type"], r["holdout_val"], r["seed"]) for r in checkpoint_data["runs"]}

    for h_type, h_val in all_targets:
        for s in args.seeds:
            if (h_type, h_val, s) in completed_pairs:
                print(f"[Skip] {h_type.upper()}:{h_val} Seed={s} already completed.")
                continue

            sample_df = prepare_training_sample(train_df, h_type, h_val, args.max_train_samples, s)
            res = train_single_run(
                sample_df, val_df, test_df, h_type, h_val,
                seed=s, epochs=args.epochs, batch_size=args.batch_size
            )
            checkpoint_data["runs"].append(res)

            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            generate_reports(checkpoint_data["runs"], checkpoint_data["in_dist"])

    generate_reports(checkpoint_data["runs"], checkpoint_data["in_dist"])
    print("\n[Done] Colab Leave-One-Out Training Finished Successfully!")


if __name__ == "__main__":
    main()
