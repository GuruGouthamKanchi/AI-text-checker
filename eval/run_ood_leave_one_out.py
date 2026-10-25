"""
eval/run_ood_leave_one_out.py
Leave-One-Generator-Out (LOGO) and Leave-One-Domain-Out (LODO) Out-of-Distribution Suite.

Evaluates generalization to unseen AI generators (Claude, Gemini, GPT-5, GPT-OSS-120B, ChatGPT, Llama)
and unseen domains (academic, essay, qa) by retraining sentence models from roberta-base across 3 seeds.

Project Rules Compliance:
- Rule 1: No invented metrics. Every number saved to results/ood_leave_one_out.csv and results/OOD_REPORT.md.
- Rule 3: No target numbers. Report empirical findings as measured.
- Rule 4: Thresholds tuned exclusively on validation split of training groups. Never on test.parquet.
- Rule 8: Sample size n reported beside every metric.
- Rule 10: Hardware: Windows 10, RTX 2050 (4GB), fp16 autocast, PyTorch 2.6.
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set workspace paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(WORKSPACE_ROOT, "data", "processed", "train.parquet")
VAL_PATH = os.path.join(WORKSPACE_ROOT, "data", "processed", "val.parquet")
TEST_PATH = os.path.join(WORKSPACE_ROOT, "data", "processed", "test.parquet")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
CHECKPOINT_PATH = os.path.join(WORKSPACE_ROOT, "eval", "ood_checkpoint.json")
CSV_PATH = os.path.join(RESULTS_DIR, "ood_leave_one_out.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "OOD_REPORT.md")

GENERATORS = ["Claude", "Gemini", "GPT-5", "GPT-OSS-120B", "ChatGPT", "Llama"]
DOMAINS = ["academic", "essay", "qa"]
SEEDS = [42, 43, 44]

# Generalization Stated Tolerances
TOLERANCE_RECALL_DELTA = -10.0   # Percentage points max drop allowed
TOLERANCE_FPR_DELTA = +5.0       # Percentage points max inflation allowed
TOLERANCE_MIN_RECALL = 85.0      # Absolute minimum recall for generalization
TOLERANCE_MAX_FPR = 10.0         # Absolute maximum FPR for operational safety


def map_generator(source: str, label: int) -> str:
    """Map raw dataset source string to canonical generator group."""
    if label == 0:
        return "human"
    s = str(source).lower()
    if "claude" in s:
        return "Claude"
    if "gemini" in s:
        return "Gemini"
    if "gpt-5" in s or "gpt5" in s:
        return "GPT-5"
    if "gpt-oss-120b" in s or "gpt-oss-20b" in s:
        return "GPT-OSS-120B"
    if "chatgpt" in s or "chat_gpt" in s:
        return "ChatGPT"
    if "llama" in s:
        return "Llama"
    return "Other_AI"


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed parquets and attach canonical group annotations."""
    print(f"[Dataset] Loading parquets from data/processed/ ...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    for df in [train_df, val_df, test_df]:
        df["gen"] = [map_generator(s, l) for s, l in zip(df["source"], df["label"])]
        df["domain"] = df["domain"].astype(str)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def prepare_training_sample(
    df: pd.DataFrame,
    holdout_type: str,
    holdout_val: str,
    max_samples: int,
    seed: int
) -> pd.DataFrame:
    """Filter held-out group and draw balanced training sample."""
    if holdout_type == "generator":
        filtered = df[df["gen"] != holdout_val].copy()
    elif holdout_type == "domain":
        filtered = df[df["domain"] != holdout_val].copy()
    else:  # in_distribution
        filtered = df.copy()

    if max_samples > 0 and len(filtered) > max_samples:
        n_per_class = max_samples // 2
        ai_df = filtered[filtered["label"] == 1]
        hu_df = filtered[filtered["label"] == 0]

        n_ai = min(len(ai_df), n_per_class)
        n_hu = min(len(hu_df), n_per_class)

        ai_sample = ai_df.sample(n=n_ai, random_state=seed)
        hu_sample = hu_df.sample(n=n_hu, random_state=seed)
        sample_df = pd.concat([ai_sample, hu_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        sample_df = filtered.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sample_df


def train_single_run(
    train_sample: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_type: str,
    holdout_val: str,
    seed: int,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_length: int = 64,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Train roberta-base on remaining training groups, select threshold on remaining val, and evaluate on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n[{holdout_type.upper()}: {holdout_val}] Seed={seed} | Training on {len(train_sample):,} samples...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device)

    # Tokenize training data
    train_enc = tokenizer(
        train_sample["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    train_labels = torch.tensor(train_sample["label"].tolist(), dtype=torch.long)
    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for b_ids, b_mask, b_lbl in train_loader:
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_lbl = b_lbl.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device):
                out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()

        print(f"  Epoch {ep+1}/{epochs} loss: {ep_loss/len(train_loader):.4f}", flush=True)

    train_time = time.time() - t0
    print(f"  Training finished in {train_time:.1f}s.", flush=True)

    # Model inference helper
    model.eval()

    def get_probabilities(texts: List[str]) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch_texts = texts[i:i+64]
                inp = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)
                with torch.amp.autocast(device):
                    logits = model(**inp).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    # 1. Choose threshold strictly on validation split of training groups (Rule 4)
    if holdout_type == "generator":
        val_eval_df = val_df[val_df["gen"] != holdout_val]
    elif holdout_type == "domain":
        val_eval_df = val_df[val_df["domain"] != holdout_val]
    else:
        val_eval_df = val_df

    # Subsample validation to 2,000 for fast leak-free threshold tuning
    if len(val_eval_df) > 2000:
        val_eval_df = val_eval_df.sample(2000, random_state=seed)

    val_probs = get_probabilities(val_eval_df["text"].tolist())
    val_labels = val_eval_df["label"].to_numpy()

    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.10, 0.90, 81):
        pred = (val_probs >= t).astype(int)
        score = f1_score(val_labels, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    print(f"  Val-chosen threshold on training groups: {best_thresh:.2f} (Val Macro-F1 = {best_f1:.4f})", flush=True)

    # 2. Evaluate on Test Split
    # If in_distribution, evaluate on all groups
    eval_results = {}
    if holdout_type == "in_distribution":
        # General in-distribution metrics across all generators and domains
        for gen in GENERATORS:
            gen_test = test_df[test_df["gen"] == gen]
            p_ai = get_probabilities(gen_test["text"].tolist())
            rec = float(np.mean(p_ai >= best_thresh))
            eval_results[f"in_dist_recall_{gen}"] = rec
            eval_results[f"in_dist_n_ai_{gen}"] = len(gen_test)

        for dom in DOMAINS:
            dom_ai = test_df[(test_df["domain"] == dom) & (test_df["label"] == 1)]
            dom_hu = test_df[(test_df["domain"] == dom) & (test_df["label"] == 0)]
            p_ai = get_probabilities(dom_ai["text"].tolist())
            p_hu = get_probabilities(dom_hu["text"].tolist())
            eval_results[f"in_dist_recall_{dom}"] = float(np.mean(p_ai >= best_thresh))
            eval_results[f"in_dist_fpr_{dom}"] = float(np.mean(p_hu >= best_thresh))
            eval_results[f"in_dist_n_ai_{dom}"] = len(dom_ai)
            eval_results[f"in_dist_n_hu_{dom}"] = len(dom_hu)

        # Overall clean human FPR
        all_human = test_df[test_df["label"] == 0]
        hu_probs = get_probabilities(all_human["text"].tolist())
        all_fpr = float(np.mean(hu_probs >= best_thresh))
        eval_results["in_dist_human_fpr"] = all_fpr
        eval_results["in_dist_n_human"] = len(all_human)
        print(f"  In-Dist Base Eval: Human FPR = {all_fpr*100:.2f}% (n={len(all_human)})", flush=True)

    elif holdout_type == "generator":
        # Held-out AI generator
        test_ai = test_df[test_df["gen"] == holdout_val]
        p_ai = get_probabilities(test_ai["text"].tolist())
        ai_rec = float(np.mean(p_ai >= best_thresh))

        # Overall clean human FPR
        test_hu = test_df[test_df["label"] == 0]
        p_hu = get_probabilities(test_hu["text"].tolist())
        hu_fpr = float(np.mean(p_hu >= best_thresh))

        eval_results["heldout_recall"] = ai_rec
        eval_results["heldout_fpr"] = hu_fpr
        eval_results["n_ai"] = len(test_ai)
        eval_results["n_human"] = len(test_hu)
        print(f"  Test Eval: Held-out {holdout_val} Rec = {ai_rec*100:.2f}% (n={len(test_ai)}) | Human FPR = {hu_fpr*100:.2f}% (n={len(test_hu)})", flush=True)

    elif holdout_type == "domain":
        # Held-out domain (both human and AI)
        test_ai = test_df[(test_df["domain"] == holdout_val) & (test_df["label"] == 1)]
        test_hu = test_df[(test_df["domain"] == holdout_val) & (test_df["label"] == 0)]

        p_ai = get_probabilities(test_ai["text"].tolist())
        p_hu = get_probabilities(test_hu["text"].tolist())

        ai_rec = float(np.mean(p_ai >= best_thresh))
        hu_fpr = float(np.mean(p_hu >= best_thresh))

        eval_results["heldout_recall"] = ai_rec
        eval_results["heldout_fpr"] = hu_fpr
        eval_results["n_ai"] = len(test_ai)
        eval_results["n_human"] = len(test_hu)
        print(f"  Test Eval: Held-out Domain '{holdout_val}' Rec = {ai_rec*100:.2f}% (n={len(test_ai)}) | Domain FPR = {hu_fpr*100:.2f}% (n={len(test_hu)})", flush=True)

    # Free CUDA memory
    del model, optimizer, scaler, train_enc, train_labels, train_ds, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "holdout_type": holdout_type,
        "holdout_val": holdout_val,
        "seed": seed,
        "val_chosen_threshold": best_thresh,
        "val_macro_f1": best_f1,
        "train_time_sec": train_time,
        **eval_results
    }


def compute_confidence_intervals(values: List[float]) -> Tuple[float, float, float, float]:
    """Calculate mean, standard deviation, and 95% confidence interval bounds."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci_low = float(mean - 1.96 * se)
    ci_high = float(mean + 1.96 * se)
    return mean, std, ci_low, ci_high


def generate_reports(all_runs: List[Dict[str, Any]], in_dist_runs: List[Dict[str, Any]]):
    """Aggregate multi-seed runs, compute CIs, apply tolerances, and write CSV & Markdown reports."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Process in-distribution reference numbers
    in_dist_ref = {}
    for gen in GENERATORS:
        recs = [r[f"in_dist_recall_{gen}"] for r in in_dist_runs]
        in_dist_ref[f"recall_{gen}"] = float(np.mean(recs))

    for dom in DOMAINS:
        recs = [r[f"in_dist_recall_{dom}"] for r in in_dist_runs]
        fprs = [r[f"in_dist_fpr_{dom}"] for r in in_dist_runs]
        in_dist_ref[f"recall_{dom}"] = float(np.mean(recs))
        in_dist_ref[f"fpr_{dom}"] = float(np.mean(fprs))

    overall_fprs = [r["in_dist_human_fpr"] for r in in_dist_runs]
    in_dist_ref["human_fpr_overall"] = float(np.mean(overall_fprs))

    # 2. Group held-out runs by (holdout_type, holdout_val)
    groups = {}
    for r in all_runs:
        k = (r["holdout_type"], r["holdout_val"])
        groups.setdefault(k, []).append(r)

    summary_rows = []

    for (h_type, h_val), runs in groups.items():
        rec_list = [r["heldout_recall"] * 100.0 for r in runs]
        fpr_list = [r["heldout_fpr"] * 100.0 for r in runs]
        thresh_list = [r["val_chosen_threshold"] for r in runs]

        r_mean, r_std, r_low, r_high = compute_confidence_intervals(rec_list)
        f_mean, f_std, f_low, f_high = compute_confidence_intervals(fpr_list)
        t_mean = float(np.mean(thresh_list))

        n_ai = runs[0]["n_ai"]
        n_hu = runs[0]["n_human"]

        # Baseline in-distribution numbers
        if h_type == "generator":
            base_rec = in_dist_ref[f"recall_{h_val}"] * 100.0
            base_fpr = in_dist_ref["human_fpr_overall"] * 100.0
        else:  # domain
            base_rec = in_dist_ref[f"recall_{h_val}"] * 100.0
            base_fpr = in_dist_ref[f"fpr_{h_val}"] * 100.0

        delta_rec = r_mean - base_rec
        delta_fpr = f_mean - base_fpr

        # Tolerance check
        pass_recall = (delta_rec >= TOLERANCE_RECALL_DELTA) or (r_mean >= TOLERANCE_MIN_RECALL)
        pass_fpr = (delta_fpr <= TOLERANCE_FPR_DELTA) or (f_mean <= TOLERANCE_MAX_FPR)

        if pass_recall and pass_fpr:
            decision = "GENERALIZES (CLAIM ALLOWED)"
            decision_code = "ALLOWED"
        else:
            decision = "DOES NOT GENERALIZE (EXPLICIT LIMITATION)"
            decision_code = "LIMITATION"

        summary_rows.append({
            "group_type": h_type,
            "held_out_target": h_val,
            "n_ai": n_ai,
            "n_human": n_hu,
            "val_threshold_mean": round(t_mean, 2),
            "heldout_recall_mean": round(r_mean, 2),
            "heldout_recall_ci95": f"[{r_low:.2f}%, {r_high:.2f}%]",
            "heldout_fpr_mean": round(f_mean, 2),
            "heldout_fpr_ci95": f"[{f_low:.2f}%, {f_high:.2f}%]",
            "in_dist_recall": round(base_rec, 2),
            "in_dist_fpr": round(base_fpr, 2),
            "delta_recall": round(delta_rec, 2),
            "delta_fpr": round(delta_fpr, 2),
            "generalization_verdict": decision_code,
            "verdict_details": decision
        })

    # Save to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CSV_PATH, index=False)
    print(f"\n[Report] Saved summary table to {CSV_PATH}")

    # Write Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        "# Out-of-Distribution Generalization Audit Report",
        f"\n**Date:** {now_str}  ",
        "**Hardware Environment:** Windows 10, RTX 2050 (4GB), Python 3.12 (fp16 autocast)  ",
        f"**Evaluation Corpus:** `data/processed/test.parquet` (12,004 sentences)  ",
        f"**Training Splits:** Retrained from `roberta-base` across 3 seeds ({SEEDS})  ",
        f"**Threshold Selection:** Strictly tuned on `val.parquet` of remaining training groups (Rule 4)  ",
        "\n---",
        "\n## 1. Generalization Decision Standards & Tolerances",
        "\nIn accordance with Task Specification 4, claims of *'generalization to unseen generators/domains'* are strictly audited against empirical tolerances relative to in-distribution baselines:",
        f"- **Recall Retention Tolerance:** $\\Delta \\text{{Recall}} \\ge {TOLERANCE_RECALL_DELTA:.1f}\\%$ points relative to in-distribution (or absolute Recall $\\ge {TOLERANCE_MIN_RECALL:.1f}\\%$).",
        f"- **FPR Constraint Tolerance:** $\\Delta \\text{{FPR}} \\le +{TOLERANCE_FPR_DELTA:.1f}\\%$ points relative to in-distribution (or absolute FPR $\\le {TOLERANCE_MAX_FPR:.1f}\\%$).",
        "- **Operational Decision:** Any group failing either criterion is labeled an **EXPLICIT LIMITATION** in repository documentation and manuscripts.",
        "\n---",
        "\n## 2. Leave-One-Out Generalization Results Table",
        "\n| Target Group | Group Type | $n_{\\text{AI}}$ | $n_{\\text{Human}}$ | Held-out Rec (95% CI) | In-Dist Rec | $\\Delta$ Rec | Held-out FPR (95% CI) | In-Dist FPR | $\\Delta$ FPR | Verdict |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    for row in summary_rows:
        rec_str = f"**{row['heldout_recall_mean']:.2f}%** ({row['heldout_recall_ci95']})"
        fpr_str = f"**{row['heldout_fpr_mean']:.2f}%** ({row['heldout_fpr_ci95']})"
        d_rec = f"{row['delta_recall']:+.2f}%"
        d_fpr = f"{row['delta_fpr']:+.2f}%"
        v_str = f"`{row['generalization_verdict']}`"
        md_lines.append(
            f"| **{row['held_out_target']}** | {row['group_type']} | {row['n_ai']:,} | {row['n_human']:,} | {rec_str} | {row['in_dist_recall']:.2f}% | {d_rec} | {fpr_str} | {row['in_dist_fpr']:.2f}% | {d_fpr} | {v_str} |"
        )

    md_lines.extend([
        "\n---",
        "\n## 3. Data-Driven Generalization Findings",
        "\n### A. Generator Generalization Analysis (LOGO):"
    ])

    # Dynamic analysis based on data
    logo_rows = [r for r in summary_rows if r["group_type"] == "generator"]
    for r in logo_rows:
        g = r["held_out_target"]
        rec = r["heldout_recall_mean"]
        fpr = r["heldout_fpr_mean"]
        n_ai = r["n_ai"]
        n_hu = r["n_human"]
        if r["generalization_verdict"] == "ALLOWED":
            md_lines.append(
                f"- **{g} (Generalizes):** When completely excluded from training, the detector retains {rec:.2f}% recall ($n={n_ai:,}$) against unseen {g} text with clean human FPR at {fpr:.2f}% ($n={n_hu:,}$). The claim of generalizability to unseen {g} models is **EMPIRICALLY SUPPORTED**."
            )
        else:
            md_lines.append(
                f"- **{g} (Explicit Limitation):** Holding out {g} resulted in recall dropping to {rec:.2f}% ($n={n_ai:,}$) or human FPR inflating to {fpr:.2f}% ($n={n_hu:,}$), violating tolerance thresholds. Generalization to unseen {g} cannot be claimed and is an **EXPLICIT LIMITATION**."
            )

    md_lines.extend([
        "\n### B. Domain Generalization Analysis (LODO):"
    ])

    lodo_rows = [r for r in summary_rows if r["group_type"] == "domain"]
    for r in lodo_rows:
        d = r["held_out_target"]
        rec = r["heldout_recall_mean"]
        fpr = r["heldout_fpr_mean"]
        n_ai = r["n_ai"]
        n_hu = r["n_human"]
        if r["generalization_verdict"] == "ALLOWED":
            md_lines.append(
                f"- **{d} (Generalizes):** Zero-shot transfer to unseen domain '{d}' achieves {rec:.2f}% AI recall ($n={n_ai:,}$) and {fpr:.2f}% FPR ($n={n_hu:,}$), within operational safety limits."
            )
        else:
            md_lines.append(
                f"- **{d} (Explicit Limitation):** Transfer to '{d}' fails tolerance boundaries with {fpr:.2f}% human FPR ($n={n_hu:,}$) and {rec:.2f}% AI recall ($n={n_ai:,}$). The repository must document that accuracy on '{d}' requires domain-specific supervision."
            )

    md_lines.extend([
        "\n---",
        "\n## 4. Google Colab Replication Instructions",
        "\nFor researchers replicating full unconstrained training on multi-GPU or cloud instances (without sample budget constraints):",
        "```bash",
        "# 1. Clone repository and install dependencies",
        "git clone https://github.com/GuruGouthamKanchi/AI-text-checker.git",
        "cd AI-text-checker",
        "pip install -q transformers datasets accelerate scikit-learn pandas pyarrow",
        "",
        "# 2. Run unconstrained leave-one-out retraining suite across all seeds",
        "python eval/run_ood_leave_one_out_colab.py --seeds 42 43 44 --epochs 2 --batch_size 32",
        "```"
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[Report] Successfully generated comprehensive report at {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out OOD Generalization Suite")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds for retraining")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (optimized for RTX 2050)")
    parser.add_argument("--max_train_samples", type=int, default=4000, help="Max balanced training samples per run")
    parser.add_argument("--groups", nargs="+", default=["all"], help="Specific groups to run or 'all'")
    args = parser.parse_args()

    train_df, val_df, test_df = load_datasets()

    # Load existing checkpoint if available
    checkpoint_data = {"in_dist": [], "runs": []}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"[Checkpoint] Loaded {len(checkpoint_data.get('runs', []))} completed runs from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[Checkpoint] Warning: Could not read checkpoint: {e}")

    # Determine execution plan
    all_targets = []
    if "all" in args.groups:
        all_targets = [("generator", g) for g in GENERATORS] + [("domain", d) for d in DOMAINS]
    else:
        for item in args.groups:
            if item in GENERATORS:
                all_targets.append(("generator", item))
            elif item in DOMAINS:
                all_targets.append(("domain", item))
            elif item == "in_distribution":
                pass
            else:
                print(f"[Warning] Unknown target group '{item}' - skipping.")

    # 1. Run in-distribution baselines first if missing
    in_dist_completed_seeds = {r["seed"] for r in checkpoint_data["in_dist"]}
    for s in args.seeds:
        if s not in in_dist_completed_seeds:
            sample_df = prepare_training_sample(train_df, "in_distribution", "none", args.max_train_samples, s)
            res = train_single_run(
                sample_df, val_df, test_df, "in_distribution", "none",
                seed=s, epochs=args.epochs, batch_size=args.batch_size
            )
            checkpoint_data["in_dist"].append(res)
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

    # 2. Run leave-one-out targets
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

            # Persist checkpoint immediately
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Update reports incrementally
            generate_reports(checkpoint_data["runs"], checkpoint_data["in_dist"])

    # Final report generation
    generate_reports(checkpoint_data["runs"], checkpoint_data["in_dist"])
    print("\n[Done] All Leave-One-Out Retraining and OOD Evaluations Finished Successfully!")


if __name__ == "__main__":
    main()
