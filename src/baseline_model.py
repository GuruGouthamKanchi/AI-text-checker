import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(data_dir="data/processed"):
    print(f"\n[Baseline] Loading preprocessed datasets from: {data_dir}...")
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Preprocessed splits not found in {data_dir}. "
            "Please run 'python src/prepare_dataset.py' first."
        )
        
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"  - Train Set: {len(train_df):,} rows")
    print(f"  - Val Set:   {len(val_df):,} rows")
    print(f"  - Test Set:  {len(test_df):,} rows")
    
    return train_df, val_df, test_df


def compute_fpr(y_true, y_pred):
    """
    Computes the False Positive Rate (FPR) explicitly:
    FPR = FP / (FP + TN)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        denominator = fp + tn
        fpr = fp / denominator if denominator > 0 else 0.0
        return fpr, tn, fp, fn, tp
    else:
        # Fallback if binary matrix shape is missing (e.g. only one class predicted or present)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        denominator = fp + tn
        fpr = fp / denominator if denominator > 0 else np.nan
        return fpr, tn, fp, fn, tp


def evaluate_predictions(y_true, y_pred, dataset_name="Evaluation"):
    """
    Computes and logs accuracy, precision, recall, F1-score, FPR, and the confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr, tn, fp, fn, tp = compute_fpr(y_true, y_pred)
    
    print(f"\n{'-'*55}")
    print(f" METRICS SUMMARY: {dataset_name.upper()}")
    print(f"{'-'*55}")
    print(f"  Accuracy:                 {acc:7.4f}")
    print(f"  Precision (AI Class 1):   {prec:7.4f}")
    print(f"  Recall (AI Class 1):      {rec:7.4f}")
    print(f"  F1-Score (AI Class 1):    {f1:7.4f}")
    
    if np.isnan(fpr):
        print(f"  False Positive Rate (FPR): N/A (No human-written records in set)")
    else:
        print(f"  False Positive Rate (FPR): {fpr:7.4f} ({fp} FPs out of {fp + tn} Human docs)")
        
    print("\n  Confusion Matrix:")
    print("                         Predicted Human (0)   Predicted AI (1)")
    print(f"    True Human (0)         {tn:<21d} {fp:<21d}")
    print(f"    True AI (1)            {fn:<21d} {tp:<21d}")
    print(f"{'-'*55}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr
    }


def source_breakdown(df, y_pred, dataset_name="Validation"):
    """
    Generates a breakdown of model accuracy and FPR across the different dataset sources.
    This reveals if the model suffers from performance gaps between data distributions.
    """
    print(f"\n[Source Breakdown] Breakdown for {dataset_name} set:")
    df_eval = df.copy()
    df_eval['pred'] = y_pred
    
    unique_sources = sorted(df_eval['source'].unique())
    
    # Draw a clean ASCII table
    header = f"{'Source / Origin':<32} | {'Rows':<8} | {'Accuracy':<10} | {'FPR':<8} | {'Human (0)':<10} | {'AI (1)':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for src in unique_sources:
        src_df = df_eval[df_eval['source'] == src]
        y_true_src = src_df['label']
        y_pred_src = src_df['pred']
        
        acc = accuracy_score(y_true_src, y_pred_src)
        fpr, tn, fp, fn, tp = compute_fpr(y_true_src, y_pred_src)
        
        fpr_str = f"{fpr:7.4f}" if not np.isnan(fpr) else "N/A"
        human_count = int(np.sum(y_true_src == 0))
        ai_count = int(np.sum(y_true_src == 1))
        
        print(f"{src:<32} | {len(src_df):<8,d} | {acc:10.4f} | {fpr_str:<8} | {human_count:<10,d} | {ai_count:<8,d}")
        
    print("-" * len(header))


def main():
    # 1. Load Parquet files
    train_df, val_df, test_df = load_data()
    
    # 2. Initialize pipeline
    print("\n[Baseline] Building TF-IDF Vectorizer + Logistic Regression pipeline...")
    # TF-IDF max features 50k, ngram range (1,2) capturing unigrams and bigrams.
    # sublinear_tf=True applies sublinear scaling (1 + log(tf)) to reduce weight of high-frequency words.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            random_state=42
        ))
    ])
    
    # 3. Train the model
    print("[Baseline] Training Logistic Regression classifier on CPU...")
    pipeline.fit(train_df['text'], train_df['label'])
    print("[Baseline] Model fitting complete!")
    
    # 4. Generate Predictions
    print("\n[Baseline] Generating predictions...")
    val_preds = pipeline.predict(val_df['text'])
    test_preds = pipeline.predict(test_df['text'])
    
    # 5. Evaluate Performance
    val_metrics = evaluate_predictions(val_df['label'], val_preds, "Validation Set")
    test_metrics = evaluate_predictions(test_df['label'], test_preds, "Test Set")
    
    # 6. Source-Specific Breakdown
    source_breakdown(val_df, val_preds, "Validation")
    source_breakdown(test_df, test_preds, "Test")
    
    # 7. Save Model Pipeline
    models_dir = os.path.join("models", "baseline")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "baseline_pipeline.joblib")
    
    print(f"\n[Baseline] Saving trained pipeline to: {model_path}...")
    joblib.dump(pipeline, model_path)
    print("[Baseline] Save completed!")
    
    # 8. Print Clear Summary & Diagnostics
    print("\n" + "="*75)
    print(" BASELINE MODEL TRAINING DIAGNOSTIC SUMMARY")
    print("=" * 75)
    print(f"  Validation Accuracy:       {val_metrics['accuracy']:7.4f}")
    print(f"  Validation FPR:            {val_metrics['fpr']:7.4f}")
    print(f"  Test Accuracy:             {test_metrics['accuracy']:7.4f}")
    print(f"  Test FPR:                  {test_metrics['fpr']:7.4f}")
    print("-" * 75)
    
    val_acc = val_metrics['accuracy']
    test_acc = test_metrics['accuracy']
    val_fpr = val_metrics['fpr']
    test_fpr = test_metrics['fpr']
    
    # Automated logic to catch issues like target leakage or high false positives early
    if val_acc > 0.99 or test_acc > 0.99:
        verdict = (
            "WARNING: Near-perfect accuracy (>99%) detected! "
            "This suggests potential target leakage in Phase 1 preprocessing "
            "(e.g., prompt leakage or duplicate patterns) despite topic-based splitting. "
            "Examine data loading and normalization closely."
        )
    elif abs(val_acc - test_acc) > 0.05:
        verdict = (
            "WARNING: Significant accuracy gap (>5%) between validation and test sets. "
            "The model may be overfitted to validation set topic distributions."
        )
    elif val_fpr > 0.01:
        verdict = (
            "NOTICE: False Positive Rate (FPR) is above 1.0%. "
            "For a Turnitin-style AI text detector, false accusations are highly critical. "
            "The baseline is sane, but we must use Transformer fine-tuning to drive FPR down further."
        )
    else:
        verdict = "SUCCESS: Baseline performance looks sane and robust. Ready for Transformer fine-tuning!"
        
    print(f"  Verdict: {verdict}")
    print("=" * 75)

if __name__ == "__main__":
    main()
