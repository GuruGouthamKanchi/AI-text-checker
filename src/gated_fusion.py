"""
src/gated_fusion.py
Learned Gating / Adaptive Fusion Module.

Implements small learned fusion combining RoBERTa micro-syntax probability (P_rob),
ModernBERT macro-context probability (P_mb), and sentence-level contextual features
(sentence length, neighbor scores, score differences).

Fit ONLY on data/processed/val.parquet with cross-validation grouped by topic_id (Project Rule 4).
Weights, scaling parameters, and validation-tuned threshold are saved to models/fusion.json.
Production fails loudly if weights are missing (Project Rule 7).
"""

import os
import json
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models")
FUSION_WEIGHTS_PATH = os.path.join(MODELS_DIR, "fusion.json")


def extract_fusion_features(
    prob_rob: np.ndarray,
    prob_mb: np.ndarray,
    texts: List[str],
    topic_ids: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts 13 contextual and relational features from model probabilities and sentence text.
    Respects document/topic boundaries if topic_ids is provided.
    """
    n = len(prob_rob)
    prob_rob = np.asarray(prob_rob, dtype=float)
    prob_mb = np.asarray(prob_mb, dtype=float)
    
    # Sentence length features
    word_lens = np.array([len(str(t).split()) for t in texts], dtype=float)
    log_lens = np.log1p(word_lens)
    
    # Contextual neighbor scores
    prev_rob = np.zeros(n, dtype=float)
    next_rob = np.zeros(n, dtype=float)
    prev_mb = np.zeros(n, dtype=float)
    next_mb = np.zeros(n, dtype=float)
    
    for i in range(n):
        # Previous sentence neighbor (within same topic if topic_ids provided)
        if i > 0 and (topic_ids is None or topic_ids[i] == topic_ids[i-1]):
            prev_rob[i] = prob_rob[i-1]
            prev_mb[i] = prob_mb[i-1]
        else:
            prev_rob[i] = prob_rob[i]
            prev_mb[i] = prob_mb[i]
            
        # Next sentence neighbor (within same topic if topic_ids provided)
        if i < n - 1 and (topic_ids is None or topic_ids[i] == topic_ids[i+1]):
            next_rob[i] = prob_rob[i+1]
            next_mb[i] = prob_mb[i+1]
        else:
            next_rob[i] = prob_rob[i]
            next_mb[i] = prob_mb[i]
            
    diff_rob_mb = np.abs(prob_rob - prob_mb)
    mean_rob_mb = 0.5 * (prob_rob + prob_mb)
    max_rob_mb = np.maximum(prob_rob, prob_mb)
    neighbor_mean_rob = 0.5 * (prev_rob + next_rob)
    neighbor_mean_mb = 0.5 * (prev_mb + next_mb)
    
    X = np.column_stack([
        prob_rob,
        prob_mb,
        word_lens,
        log_lens,
        prev_rob,
        next_rob,
        prev_mb,
        next_mb,
        neighbor_mean_rob,
        neighbor_mean_mb,
        diff_rob_mb,
        mean_rob_mb,
        max_rob_mb
    ])
    
    feature_names = [
        "prob_rob",
        "prob_mb",
        "word_len",
        "log_len",
        "prev_prob_rob",
        "next_prob_rob",
        "prev_prob_mb",
        "next_prob_mb",
        "neighbor_mean_rob",
        "neighbor_mean_mb",
        "diff_rob_mb",
        "mean_rob_mb",
        "max_rob_mb"
    ]
    return X, feature_names


class GatedFusion:
    """
    Learned Gating / Adaptive Fusion Classifier.
    Computes P_fused = sigmoid(w^T * normalize(X) + b).
    Optionally applies post-hoc Isotonic Calibration fitted on val.parquet.
    """
    def __init__(self, weights_path: str = FUSION_WEIGHTS_PATH):
        self.weights_path = weights_path
        self.feature_names: List[str] = []
        self.coefficients: np.ndarray = np.array([])
        self.intercept: float = 0.0
        self.scaler_mean: np.ndarray = np.array([])
        self.scaler_scale: np.ndarray = np.array([])
        self.threshold: float = 0.50
        self.is_fitted: bool = False
        self.calibrator_x: Optional[List[float]] = None
        self.calibrator_y: Optional[List[float]] = None
        
        if os.path.exists(weights_path):
            self.load(weights_path)

    def load(self, path: str = FUSION_WEIGHTS_PATH):
        """Loads model weights, scaler, and threshold from JSON."""
        if not os.path.exists(path):
            if os.getenv("STRICT_PRODUCTION", "0") == "1" or os.getenv("REQUIRE_MODELS", "0") == "1":
                raise FileNotFoundError(f"[Fatal] Fusion model weights not found at '{path}' (Project Rule 7).")
            raise FileNotFoundError(f"Missing fusion model weights at '{path}'.")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.feature_names = data["feature_names"]
        self.coefficients = np.array(data["coefficients"], dtype=float)
        self.intercept = float(data["intercept"])
        self.scaler_mean = np.array(data["scaler_mean"], dtype=float)
        self.scaler_scale = np.array(data["scaler_scale"], dtype=float)
        self.threshold = float(data.get("val_threshold", 0.50))
        self.calibrator_x = data.get("calibrator_x")
        self.calibrator_y = data.get("calibrator_y")
        self.is_fitted = True

    def save(self, path: str = FUSION_WEIGHTS_PATH, metadata: Optional[Dict[str, Any]] = None):
        """Saves model weights, scaler, threshold, and metrics to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "model_type": "LogisticRegression_GatedFusion",
            "feature_names": self.feature_names,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "val_threshold": self.threshold,
            "calibrator_x": self.calibrator_x,
            "calibrator_y": self.calibrator_y,
            "metadata": metadata or {}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes raw fused probabilities: P_fused = sigmoid(w^T * normalize(X) + b)."""
        if not self.is_fitted:
            raise RuntimeError("GatedFusion model is not loaded or fitted.")
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        logits = np.dot(X_scaled, self.coefficients) + self.intercept
        # Numerically stable sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        return probs

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Classifies inputs using threshold tuned on val.parquet."""
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Applies Isotonic calibration fitted on val.parquet."""
        raw_probs = self.predict_proba(X)
        if self.calibrator_x is None or self.calibrator_y is None:
            return raw_probs
        # Piecewise linear interpolation for isotonic calibration
        cal_probs = np.interp(raw_probs, self.calibrator_x, self.calibrator_y)
        return np.clip(cal_probs, 0.0, 1.0)


def train_gated_fusion_on_val(
    val_parquet_path: str,
    val_cache_path: str,
    save_path: str = FUSION_WEIGHTS_PATH,
    n_splits: int = 5,
    random_seed: int = 42
) -> Tuple[GatedFusion, Dict[str, Any]]:
    """
    Fits learned fusion model strictly on val.parquet (Rule 4) with topic-grouped CV.
    Returns fitted GatedFusion instance and dictionary of validation metrics.
    """
    print(f"\n[GatedFusion] Loading validation split from: {val_cache_path}...")
    df_val = pd.read_parquet(val_cache_path)
    
    y_val = df_val["label"].values.astype(int)
    groups_val = df_val["topic_id"].values
    
    X_val, feat_names = extract_fusion_features(
        prob_rob=df_val["prob_rob"].values,
        prob_mb=df_val["prob_mb"].values,
        texts=df_val["text"].tolist(),
        topic_ids=groups_val.tolist()
    )
    
    print(f"[GatedFusion] Extracted {X_val.shape[1]} features across {len(X_val):,d} validation sentences.")
    print(f"[GatedFusion] Running {n_splits}-fold GroupKFold cross-validation grouped by topic_id...")
    
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(df_val), dtype=float)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_val, y_val, groups=groups_val), 1):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_val[train_idx])
        X_va = scaler_fold.transform(X_val[val_idx])
        
        clf_fold = LogisticRegression(C=1.0, random_state=random_seed, max_iter=1000)
        clf_fold.fit(X_tr, y_val[train_idx])
        
        probs_va = clf_fold.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = probs_va
        
        fold_acc = accuracy_score(y_val[val_idx], (probs_va >= 0.5).astype(int))
        fold_f1 = f1_score(y_val[val_idx], (probs_va >= 0.5).astype(int), average="macro")
        cv_scores.append({"fold": fold, "accuracy": fold_acc, "macro_f1": fold_f1})
        
    oof_acc = accuracy_score(y_val, (oof_probs >= 0.5).astype(int))
    oof_f1 = f1_score(y_val, (oof_probs >= 0.5).astype(int), average="macro")
    oof_auc = roc_auc_score(y_val, oof_probs)
    print(f" -> CV OOF Accuracy: {oof_acc*100:.2f}% | Macro-F1: {oof_f1*100:.2f}% | ROC-AUC: {oof_auc*100:.2f}%")
    
    # Fit final model on 100% of val.parquet
    print("[GatedFusion] Fitting final model on complete validation split...")
    scaler_final = StandardScaler()
    X_val_scaled = scaler_final.fit_transform(X_val)
    clf_final = LogisticRegression(C=1.0, random_state=random_seed, max_iter=1000)
    clf_final.fit(X_val_scaled, y_val)
    
    final_val_probs = clf_final.predict_proba(X_val_scaled)[:, 1]
    
    # Tune decision threshold on val.parquet strictly (Rule 4)
    best_t = 0.50
    best_f1_t = -1.0
    for t_cand in np.linspace(0.05, 0.95, 91):
        f1_c = f1_score(y_val, (final_val_probs >= t_cand).astype(int), average="macro", zero_division=0)
        if f1_c > best_f1_t:
            best_f1_t = float(f1_c)
            best_t = float(t_cand)
            
    print(f" -> Optimal Validation Threshold: {best_t:.2f} (Val Macro-F1: {best_f1_t*100:.2f}%)")
    
    # Fit Isotonic Calibrator on OOF validation probabilities
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_probs, y_val)
    
    # Extract calibrator piece-wise points
    cal_grid = np.linspace(0.0, 1.0, 101)
    cal_vals = iso.predict(cal_grid).tolist()
    
    # Construct GatedFusion instance
    fusion = GatedFusion(weights_path=save_path)
    fusion.feature_names = feat_names
    fusion.coefficients = clf_final.coef_[0]
    fusion.intercept = float(clf_final.intercept_[0])
    fusion.scaler_mean = scaler_final.mean_
    fusion.scaler_scale = scaler_final.scale_
    fusion.threshold = best_t
    fusion.calibrator_x = cal_grid.tolist()
    fusion.calibrator_y = cal_vals
    fusion.is_fitted = True
    
    # Formulate weights dictionary
    weights_dict = {feat_names[i]: float(fusion.coefficients[i]) for i in range(len(feat_names))}
    
    metadata = {
        "n_samples_val": len(df_val),
        "n_topics_val": int(len(np.unique(groups_val))),
        "cv_folds": n_splits,
        "oof_accuracy": oof_acc,
        "oof_macro_f1": oof_f1,
        "oof_roc_auc": oof_auc,
        "optimal_val_threshold": best_t,
        "feature_weights": weights_dict
    }
    
    fusion.save(save_path, metadata=metadata)
    print(f"[GatedFusion] Saved model weights and scaler to: {save_path}")
    
    return fusion, metadata


if __name__ == "__main__":
    VAL_PARQUET = os.path.join(WORKSPACE_ROOT, "data", "processed", "val.parquet")
    VAL_CACHE = os.path.join(WORKSPACE_ROOT, "eval", "cache_val_preds.parquet")
    fusion, meta = train_gated_fusion_on_val(VAL_PARQUET, VAL_CACHE)
