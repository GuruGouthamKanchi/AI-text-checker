"""
eval/eval_gated_fusion.py
Rigorous Empirical Evaluation of Learned Gating / Adaptive Fusion vs. Baselines.

Compares 6 systems on raw, unsmoothed data/processed/test.parquet:
1. RoBERTa alone
2. ModernBERT alone
3. Mean of the two
4. Max of the two
5. Top-k pooling
6. Learned fusion (src/gated_fusion.py)

Methodology:
- Zero test tuning: all decision thresholds and calibrators fitted strictly on val.parquet (Rule 4).
- Paired topic-level clustered bootstrap (1,000 resamples) with Holm-Bonferroni correction.
- Reports model weights, FPR at val-chosen threshold, TPR@1%FPR, and post-calibration ECE.
- Writes results/gated_fusion_comparison.csv and results/GATED_FUSION_REPORT.md.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKSPACE_ROOT)

from src.gated_fusion import GatedFusion, extract_fusion_features

VAL_CACHE_PATH = os.path.join(WORKSPACE_ROOT, "eval", "cache_val_preds.parquet")
TEST_CACHE_PATH = os.path.join(WORKSPACE_ROOT, "eval", "cache_test_preds.parquet")
FUSION_WEIGHTS_PATH = os.path.join(WORKSPACE_ROOT, "models", "fusion.json")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
CSV_OUTPUT = os.path.join(RESULTS_DIR, "gated_fusion_comparison.csv")
REPORT_OUTPUT = os.path.join(RESULTS_DIR, "GATED_FUSION_REPORT.md")

os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        b_min = bin_edges[i]
        b_max = bin_edges[i+1]
        mask = (y_prob >= b_min) & (y_prob <= b_max) if i == n_bins - 1 else (y_prob >= b_min) & (y_prob < b_max)
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> float:
    """Computes True Positive Rate (Recall) at or below target False Positive Rate."""
    neg_mask = (y_true == 0)
    pos_mask = (y_true == 1)
    if neg_mask.sum() == 0 or pos_mask.sum() == 0:
        return 0.0
    neg_scores = np.sort(y_prob[neg_mask])[::-1]
    n_neg = len(neg_scores)
    idx = int(np.floor(target_fpr * n_neg))
    idx = min(max(0, idx), n_neg - 1)
    threshold = neg_scores[idx]
    return float((y_prob[pos_mask] >= threshold).mean())


def compute_window_topk(prob_rob: np.ndarray) -> np.ndarray:
    """Computes 3-sentence window top-2 mean pooling."""
    n = len(prob_rob)
    res = np.zeros(n, dtype=float)
    for i in range(n):
        l = prob_rob[i-1] if i > 0 else prob_rob[i]
        c = prob_rob[i]
        r = prob_rob[i+1] if i < n-1 else prob_rob[i]
        win = sorted([l, c, r], reverse=True)
        res[i] = np.mean(win[:2])
    return res


def paired_topic_bootstrap(
    y_test: np.ndarray,
    topics_test: np.ndarray,
    prob_target: np.ndarray,
    prob_ref: np.ndarray,
    thresh_target: float,
    thresh_ref: float,
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Topic-clustered paired bootstrap test between target system and reference system.
    Returns delta metrics and empirical two-tailed p-values.
    """
    np.random.seed(seed)
    unique_topics = np.unique(topics_test)
    n_topics = len(unique_topics)
    
    # Map topic to indices
    topic_to_idx = {}
    for idx, t in enumerate(topics_test):
        if t not in topic_to_idx:
            topic_to_idx[t] = []
        topic_to_idx[t].append(idx)
    topic_to_idx = {k: np.array(v) for k, v in topic_to_idx.items()}
    
    pred_bin_target = (prob_target >= thresh_target).astype(int)
    pred_bin_ref = (prob_ref >= thresh_ref).astype(int)
    
    delta_acc_list = []
    delta_f1_list = []
    delta_auc_list = []
    
    for _ in range(n_resamples):
        sampled_topics = np.random.choice(unique_topics, size=n_topics, replace=True)
        sample_indices = np.concatenate([topic_to_idx[t] for t in sampled_topics])
        
        y_b = y_test[sample_indices]
        # Skip invalid samples
        if len(np.unique(y_b)) < 2:
            continue
            
        acc_tar = accuracy_score(y_b, pred_bin_target[sample_indices])
        acc_ref = accuracy_score(y_b, pred_bin_ref[sample_indices])
        delta_acc_list.append(acc_tar - acc_ref)
        
        f1_tar = f1_score(y_b, pred_bin_target[sample_indices], average="macro", zero_division=0)
        f1_ref = f1_score(y_b, pred_bin_ref[sample_indices], average="macro", zero_division=0)
        delta_f1_list.append(f1_tar - f1_ref)
        
        auc_tar = roc_auc_score(y_b, prob_target[sample_indices])
        auc_ref = roc_auc_score(y_b, prob_ref[sample_indices])
        delta_auc_list.append(auc_tar - auc_ref)
        
    delta_acc = np.mean(delta_acc_list)
    delta_f1 = np.mean(delta_f1_list)
    delta_auc = np.mean(delta_auc_list)
    
    p_acc = 2.0 * min(np.mean(np.array(delta_acc_list) <= 0), np.mean(np.array(delta_acc_list) >= 0))
    p_f1 = 2.0 * min(np.mean(np.array(delta_f1_list) <= 0), np.mean(np.array(delta_f1_list) >= 0))
    p_auc = 2.0 * min(np.mean(np.array(delta_auc_list) <= 0), np.mean(np.array(delta_auc_list) >= 0))
    
    return {
        "delta_acc": float(delta_acc),
        "p_acc": float(min(1.0, p_acc)),
        "delta_f1": float(delta_f1),
        "p_f1": float(min(1.0, p_f1)),
        "delta_auc": float(delta_auc),
        "p_auc": float(min(1.0, p_auc))
    }


def main():
    print("=" * 80)
    print(" LEARNED GATING / ADAPTIVE FUSION COMPREHENSIVE EMPIRICAL EVALUATION")
    print("=" * 80)

    # 1. Load Data
    val_df = pd.read_parquet(VAL_CACHE_PATH)
    test_df = pd.read_parquet(TEST_CACHE_PATH)
    
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)
    topics_test = test_df["topic_id"].values
    
    print(f"Validation Set: {len(val_df):,d} sentences ({y_val.sum():,d} AI, {(y_val==0).sum():,d} Human)")
    print(f"Test Set:       {len(test_df):,d} sentences ({y_test.sum():,d} AI, {(y_test==0).sum():,d} Human)")
    
    # 2. Extract Features & Load Gated Fusion Model
    X_val, feat_names = extract_fusion_features(
        val_df["prob_rob"].values, val_df["prob_mb"].values, val_df["text"].tolist(), val_df["topic_id"].tolist()
    )
    X_test, _ = extract_fusion_features(
        test_df["prob_rob"].values, test_df["prob_mb"].values, test_df["text"].tolist(), test_df["topic_id"].tolist()
    )
    
    fusion = GatedFusion(weights_path=FUSION_WEIGHTS_PATH)
    if not fusion.is_fitted:
        raise FileNotFoundError(f"[Fatal] Fusion model not found at {FUSION_WEIGHTS_PATH} (Rule 7).")
        
    prob_fusion_val = fusion.predict_proba(X_val)
    prob_fusion_test = fusion.predict_proba(X_test)
    
    # 3. Assemble the 6 Systems
    raw_rob_val = val_df["prob_rob"].values
    raw_rob_test = test_df["prob_rob"].values
    
    raw_mb_val = val_df["prob_mb"].values
    raw_mb_test = test_df["prob_mb"].values
    
    mean_val = 0.50 * (raw_rob_val + raw_mb_val)
    mean_test = 0.50 * (raw_rob_test + raw_mb_test)
    
    max_val = np.maximum(raw_rob_val, raw_mb_val)
    max_test = np.maximum(raw_rob_test, raw_mb_test)
    
    topk_val = compute_window_topk(raw_rob_val)
    topk_test = compute_window_topk(raw_rob_test)
    
    systems = [
        {
            "id": "SYS_ROB",
            "name": "RoBERTa Alone",
            "val_prob": raw_rob_val,
            "test_prob": raw_rob_test,
            "weights_desc": "w_rob = 1.0, w_mb = 0.0"
        },
        {
            "id": "SYS_MB",
            "name": "ModernBERT Alone",
            "val_prob": raw_mb_val,
            "test_prob": raw_mb_test,
            "weights_desc": "w_rob = 0.0, w_mb = 1.0"
        },
        {
            "id": "SYS_MEAN",
            "name": "Mean of the Two",
            "val_prob": mean_val,
            "test_prob": mean_test,
            "weights_desc": "0.50 * P_rob + 0.50 * P_mb"
        },
        {
            "id": "SYS_MAX",
            "name": "Max of the Two",
            "val_prob": max_val,
            "test_prob": max_test,
            "weights_desc": "max(P_rob, P_mb)"
        },
        {
            "id": "SYS_TOPK",
            "name": "Top-k Pooling (Window Top-2)",
            "val_prob": topk_val,
            "test_prob": topk_test,
            "weights_desc": "mean(top-2 in [P(i-1), P(i), P(i+1)])"
        },
        {
            "id": "SYS_FUSION",
            "name": "Learned Gated Fusion",
            "val_prob": prob_fusion_val,
            "test_prob": prob_fusion_test,
            "weights_desc": "sigmoid(w^T * X_scaled + b) (13 features)"
        }
    ]

    # 4. Fit Thresholds & Calibrators on Validation Set (Rule 4)
    print("\n[Val Tuning] Selecting optimal F1 threshold & fitting calibrators on val.parquet...")
    for s in systems:
        val_p = s["val_prob"]
        best_t = 0.50
        best_f1 = -1.0
        for t_cand in np.linspace(0.05, 0.95, 91):
            f1_c = f1_score(y_val, (val_p >= t_cand).astype(int), average="macro", zero_division=0)
            if f1_c > best_f1:
                best_f1 = float(f1_c)
                best_t = float(t_cand)
        s["val_thresh"] = best_t
        s["val_f1"] = best_f1
        
        # Fit Isotonic Calibrator strictly on val.parquet
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_p, y_val)
        s["calibrator"] = iso
        s["test_prob_cal"] = iso.predict(s["test_prob"])
        
        print(f" -> {s['name']:<30} | Val Thresh: {best_t:.2f} | Val F1: {best_f1*100:.2f}%")

    # 5. Evaluate All Systems on Test Set
    csv_rows = []
    print("\n[Test Evaluation] Computing performance metrics on test.parquet...")
    for s in systems:
        p_test = s["test_prob"]
        t_val = s["val_thresh"]
        p_cal = s["test_prob_cal"]
        
        y_pred = (p_test >= t_val).astype(int)
        
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        auc = float(roc_auc_score(y_test, p_test))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = float(fp / (fp + tn))
        recall = float(tp / (tp + fn))
        
        tpr1 = compute_tpr_at_fpr(y_test, p_test, target_fpr=0.01)
        ece_raw = compute_ece(y_test, p_test)
        ece_cal = compute_ece(y_test, p_cal)
        
        s["test_acc"] = acc
        s["test_f1"] = f1
        s["test_auc"] = auc
        s["test_fpr"] = fpr
        s["test_recall"] = recall
        s["tpr_at_1pct_fpr"] = tpr1
        s["ece_raw"] = ece_raw
        s["ece_cal"] = ece_cal
        
        csv_rows.append({
            "system_id": s["id"],
            "system_name": s["name"],
            "model_weights": s["weights_desc"],
            "val_chosen_threshold": f"{t_val:.2f}",
            "test_accuracy": f"{acc*100:.2f}%",
            "test_macro_f1": f"{f1*100:.2f}%",
            "test_roc_auc": f"{auc*100:.2f}%",
            "fpr_at_val_thresh": f"{fpr*100:.2f}%",
            "ai_recall": f"{recall*100:.2f}%",
            "tpr_at_1pct_fpr": f"{tpr1*100:.2f}%",
            "ece_raw": f"{ece_raw*100:.2f}%",
            "ece_calibrated": f"{ece_cal*100:.2f}%",
            "n_test": len(y_test)
        })

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(CSV_OUTPUT, index=False)
    print(f"[Success] Wrote comparison metrics to: {CSV_OUTPUT}")

    # 6. Paired Topic-Level Bootstrap with Holm-Bonferroni Correction
    print("\n[Bootstrap] Running paired topic-level bootstrap (1,000 resamples) vs Learned Fusion...")
    sys_fusion = systems[5]  # Learned Fusion
    
    comparisons = []
    for s_base in systems[:5]:
        res = paired_topic_bootstrap(
            y_test=y_test,
            topics_test=topics_test,
            prob_target=sys_fusion["test_prob"],
            prob_ref=s_base["test_prob"],
            thresh_target=sys_fusion["val_thresh"],
            thresh_ref=s_base["val_thresh"],
            n_resamples=1000,
            seed=42
        )
        comparisons.append({
            "baseline_id": s_base["id"],
            "baseline_name": s_base["name"],
            "delta_acc": res["delta_acc"],
            "p_acc": res["p_acc"],
            "delta_f1": res["delta_f1"],
            "p_f1": res["p_f1"],
            "delta_auc": res["delta_auc"],
            "p_auc": res["p_auc"]
        })
        
    # Apply Holm-Bonferroni Step-Down Correction on Accuracy p-values
    # Sort comparisons by raw p-value ascending
    m = len(comparisons)
    sorted_indices_acc = np.argsort([c["p_acc"] for c in comparisons])
    
    for rank, idx in enumerate(sorted_indices_acc):
        c = comparisons[idx]
        alpha_rank = 0.05 / (m - rank)
        is_sig = (c["p_acc"] <= alpha_rank)
        c["holm_bonferroni_acc_alpha"] = alpha_rank
        c["is_significant_acc"] = is_sig

    # Also for Macro-F1
    sorted_indices_f1 = np.argsort([c["p_f1"] for c in comparisons])
    for rank, idx in enumerate(sorted_indices_f1):
        c = comparisons[idx]
        alpha_rank = 0.05 / (m - rank)
        is_sig = (c["p_f1"] <= alpha_rank)
        c["holm_bonferroni_f1_alpha"] = alpha_rank
        c["is_significant_f1"] = is_sig

    # Print summary
    print("\n--- Paired Bootstrap Significance Results ---")
    for c in comparisons:
        sig_str = "YES" if c["is_significant_acc"] else "NO"
        print(f"Learned Fusion vs. {c['baseline_name']:<25}: Delta Acc = {c['delta_acc']*100:+.2f}%, p = {c['p_acc']:.4f} (Sig: {sig_str})")

    # 7. Write Markdown Report
    report_lines = []
    report_lines.append("# Learned Gating / Adaptive Pooling Empirical Decision Report\n")
    report_lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    report_lines.append(f"**Hardware Environment:** Windows 10, RTX 2050 (4GB), Python 3.12 (Rule 10)  ")
    report_lines.append(f"**Evaluation Corpus:** `data/processed/test.parquet` ({len(y_test):,d} sentences, strictly topic-held-out)  ")
    report_lines.append(f"**Validation Tuning:** `data/processed/val.parquet` ({len(y_val):,d} sentences, Rule 4)  ")
    report_lines.append(f"**Bootstrap Resamples:** 1,000 topic-clustered resamples with Holm-Bonferroni multi-test correction.\n")
    report_lines.append("---\n")

    report_lines.append("## 1. Executive Statistical Verdict\n")

    # Find simplest competitive baseline:
    # Top-k pooling vs Mean vs Learned Fusion
    topk_acc = sys_fusion["test_acc"]
    mean_comp = [c for c in comparisons if c["baseline_id"] == "SYS_MEAN"][0]
    topk_comp = [c for c in comparisons if c["baseline_id"] == "SYS_TOPK"][0]

    # Does learned fusion significantly beat Top-k Pooling or Mean?
    fusion_beats_topk = (topk_comp["delta_acc"] > 0 and topk_comp["is_significant_acc"])
    fusion_beats_mean = (mean_comp["delta_acc"] > 0 and mean_comp["is_significant_acc"])

    if fusion_beats_topk and fusion_beats_mean:
        verdict_text = (
            "**CLAIM SUPPORTED (KEEP GATING)**: Learned Gated Fusion achieved a statistically significant "
            "improvement over all competitive baselines after Holm-Bonferroni correction."
        )
        decision_code = "KEEP_GATING"
    else:
        # Determine best performing baseline
        best_baseline = max(systems[:5], key=lambda s: s["test_acc"])
        verdict_text = (
            f"**CLAIM UNSUPPORTED (DROP GATING)**: Learned Gated Fusion ({sys_fusion['test_acc']*100:.2f}% Acc) "
            f"did NOT significantly outperform the simplest competitive baseline (**{best_baseline['name']}**, "
            f"{best_baseline['test_acc']*100:.2f}% Acc, $\\Delta = {sys_fusion['test_acc'] - best_baseline['test_acc']:+.2f}\\%$, "
            f"adjusted $p$-value not significant). In accordance with Task Specification 3, the repository "
            f"must describe the simplest baseline (**{best_baseline['name']}**) as the primary method and "
            f"drop promotional claims of 'learned gating / adaptive pooling' from documentation."
        )
        decision_code = "DROP_GATING"

    report_lines.append(f"> **Architectural Decision Verdict:** {verdict_text}\n")

    report_lines.append("## 2. Comprehensive Test Benchmark Comparison (Table)\n")
    report_lines.append(df_csv.to_markdown(index=False))
    report_lines.append("\n---\n")

    report_lines.append("## 3. Paired Topic-Level Bootstrap Significance (1,000 Resamples)\n")
    report_lines.append("| Comparison (Learned Fusion vs.) | $\\Delta$ Accuracy | $p$-value (Acc) | $\\alpha$ Threshold | Significant? | $\\Delta$ Macro-F1 | $p$-value (F1) | $\\Delta$ ROC-AUC |")
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for c in comparisons:
        sig_str = "**YES**" if c["is_significant_acc"] else "NO"
        report_lines.append(
            f"| **{c['baseline_name']}** | {c['delta_acc']*100:+.2f}% | $p = {c['p_acc']:.4f}$ | $\\alpha = {c['holm_bonferroni_acc_alpha']:.4f}$ | {sig_str} | {c['delta_f1']*100:+.2f}% | $p = {c['p_f1']:.4f}$ | {c['delta_auc']*100:+.2f}% |"
        )
    report_lines.append("\n*Correction Method: Holm-Bonferroni step-down procedure across 5 paired comparisons.*  \n")
    report_lines.append("---\n")

    report_lines.append("## 4. Learned Model Weights & Parameter Analysis\n")
    report_lines.append("The learned fusion model was trained strictly on `data/processed/val.parquet` ($n=11,997$) and saved to `models/fusion.json`:\n")
    report_lines.append(f"- **Intercept ($b$):** `{fusion.intercept:.4f}`")
    report_lines.append(f"- **Validation-Tuned Decision Threshold:** `{fusion.threshold:.2f}`")
    report_lines.append("- **Feature Coefficients ($w$):**\n")
    report_lines.append("| Feature | Coefficient | Scale Mean | Scale Std | Interpretation |")
    report_lines.append("|---|:---:|:---:|:---:|---|")
    for idx, f_name in enumerate(fusion.feature_names):
        w_val = fusion.coefficients[idx]
        m_val = fusion.scaler_mean[idx]
        s_val = fusion.scaler_scale[idx]
        report_lines.append(f"| `{f_name}` | `{w_val:+.4f}` | `{m_val:.4f}` | `{s_val:.4f}` | Feature weight in normalized logistic fusion |")
    report_lines.append("\n---\n")

    report_lines.append("## 5. Calibration Analysis (ECE Before & After Isotonic Regression)\n")
    report_lines.append("| System | Raw ECE ($n=12,004$) | Calibrated ECE ($n=12,004$) | ECE Reduction |")
    report_lines.append("|---|:---:|:---:|:---:|")
    for s in systems:
        red = (1.0 - s["ece_cal"] / s["ece_raw"]) * 100 if s["ece_raw"] > 0 else 0.0
        report_lines.append(f"| **{s['name']}** | {s['ece_raw']*100:.2f}% | {s['ece_cal']*100:.2f}% | -{red:.1f}% |")

    report_lines.append("\n---\n")
    report_lines.append("## 6. Actionable Decisions for Repository & Ledger\n")
    if decision_code == "DROP_GATING":
        report_lines.append("1. **Drop 'Gating' Terminology:** Update `README.md`, `CLAUDE.md`, and `ieee_paper.tex` to describe the empirical pooling method (Top-k pooling / Equal Mean) rather than claiming a 'learned gating mechanism'.")
        report_lines.append("2. **Claims Ledger Update:** Mark gating claims (`CLM-006`, `CLM-023`, `CLM-030`, `CLM-058`) as `RESOLVED / QUALIFIED` to reflect that learned gating provides no statistically significant gain over simple local pooling.")
        report_lines.append("3. **Production Pipeline:** Retain Top-k pooling / Mean ensemble as primary inference engine in `src/highlighter.py` and document `src/gated_fusion.py` as an empirical research baseline with weights in `models/fusion.json`.")
    else:
        report_lines.append("1. **Retain Gating:** Adopt learned fusion in production with weights from `models/fusion.json`.")

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Success] Generated comprehensive report at: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
