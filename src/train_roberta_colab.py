# %% [markdown]
# # Phase 4: Transformer Model Fine-Tuning (RoBERTa-base Target)
# 
# This notebook is designed to run on Google Colab (T4 GPU) to fine-tune `roberta-base`.
# Key enhancements in this phase:
# 1. Transition to the larger **RoBERTa-base** model (125M params) for superior stylistic extraction.
# 2. Increase sequence length **MAX_LENGTH = 384** to capture full paragraphs and richer essay context.
# 3. Add **label smoothing (0.1)** to prevent overconfident false alarms.
# 4. Integrate **strict validation printing** to explicitly verify the active model type and prevent caching issues.
# 5. Extract **highest-confidence false positives** in the human `persuade_corpus` for qualitative error analysis.
# 6. Output a **three-way comparison table**: Baseline vs. DistilBERT vs. RoBERTa-base.
# 7. Implement a **sentence-level classifier function** `predict_sentence_scores` as a foundation for Turnitin-style highlighting.

# %% [markdown]
# ## CELL 1: Environment Setup & Configurations
# - Install packages, mount Google Drive, and set training parameters.

# %%
# 1. Install required packages in Google Colab (if running in Colab)
# Uncomment the line below when running in Google Colab:
# !pip install -q transformers[torch] datasets accelerate evaluate pandas pyarrow scikit-learn nltk

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import nltk
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download NLTK resources for sentence splitting
# We download both 'punkt' and 'punkt_tab' to support older and newer NLTK versions
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except (LookupError, AttributeError):
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[Warning] Failed to download NLTK resource {resource}: {e}")

# 2. Mount Google Drive
try:
    from google.colab import drive
    print("[Setup] Mounting Google Drive...")
    drive.mount('/content/drive')
except ImportError:
    print("[Setup] Google Drive mount skipped (running outside Google Colab).")

# 3. Global Hyperparameters and Paths Configuration
# Adjust paths to match your Google Drive structure.
DATA_DIR = "/content/drive/MyDrive/ai-text-detector/data/processed/"

# A fresh separate output directory for RoBERTa checkpoints and final weights
OUTPUT_DIR = "/content/drive/MyDrive/ai-text-detector/models/roberta-checkpoints/"
FINAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "roberta-base-final")

MODEL_NAME = "roberta-base"

# Increased sequence length to capture wider essay context (first ~270 words)
MAX_LENGTH = 384

# Batch size of 8 with gradient accumulation steps = 2 maintains an effective batch size of 16
# while remaining highly safe against Out-of-Memory (OOM) errors on T4 GPUs
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1.5e-5
NUM_EPOCHS = 3
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"[Setup] Configured parameters. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# %% [markdown]
# ## CELL 2: Load & Tokenize Data
# - Load train, val, and test splits.
# - Tokenize with RoBERTa's byte-level BPE tokenizer (with MAX_LENGTH = 384).

# %%
print(f"\n[Data] Loading Parquet splits from {DATA_DIR}...")
try:
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
except FileNotFoundError as e:
    print(f"\n[CRITICAL] Could not load datasets: {e}")
    print("Please verify your DATA_DIR path and ensure files were uploaded to your Drive.")
    # Creating small dummy mock sets for compilation checks if run locally
    mock_data = {
        "text": ["Human text example for debugging.", "AI generated sentence text model."],
        "label": [0, 1],
        "source": ["hc3_human", "hc3_chatgpt"],
        "topic_id": ["dummy1", "dummy2"]
    }
    train_df = pd.DataFrame(mock_data)
    val_df = pd.DataFrame(mock_data)
    test_df = pd.DataFrame(mock_data)

print(f"  - Train Set: {len(train_df):,} rows")
print(f"  - Val Set:   {len(val_df):,} rows")
print(f"  - Test Set:  {len(test_df):,} rows")

# Convert to HF Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load RoBERTa Tokenizer
print(f"\n[Tokenizer] Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Tokenize text with truncation and padding to MAX_LENGTH (384)
    # RoBERTa's AutoTokenizer handles special tokens and formatting internally
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length"
    )

print("[Tokenizer] Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'source', 'topic_id'])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'source', 'topic_id'])
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'source', 'topic_id'])

# Sanity Check Decoded Text
print("\n[Sanity Check] Decoding a processed training record:")
sample_ids = tokenized_train[0]['input_ids']
print(f"  - First 10 tokens: {sample_ids[:10]}")
print(f"  - Decoded: {tokenizer.decode(sample_ids[:50])}...")

# %% [markdown]
# ## CELL 3: Model, Custom Trainer & Verification Setup
# - Load sequence classification model.
# - Print explicit model type verification.
# - Compute dynamic class weights.
# - Build custom Trainer subclass with weighted CrossEntropyLoss.
# - Formulate evaluation metrics (calculating FPR).

# %%
import torch.nn as nn
from transformers import EvalPrediction

# 1. Initialize RoBERTa classification model
print(f"\n[Model] Ingesting model checkpoint {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# =========================================================================
# CRITICAL MODEL VERIFICATION POINT 1 (Pre-Training)
# =========================================================================
print("\n" + "="*80)
print(" VERIFICATION: ACTIVE MODEL LOAD CHECK (PRE-TRAINING)")
print("-" * 80)
print(f"  - Model Class Name:       {type(model).__name__}")
print(f"  - Tokenizer Class Name:   {type(tokenizer).__name__}")
print(f"  - Model Config Path/Name: {model.config.name_or_path}")
print("="*80)

# Check assertions to stop execution immediately if wrong model is loaded
assert "Roberta" in type(model).__name__, "CRITICAL ERROR: Loaded model is not RoBERTa!"
assert "Roberta" in type(tokenizer).__name__, "CRITICAL ERROR: Loaded tokenizer is not RoBERTa!"
assert "roberta" in model.config.name_or_path.lower(), "CRITICAL ERROR: Config is not roberta-base!"

# 2. Compute Class Weights for Imbalance Correction (64/36 ratio)
labels_train = train_df['label'].values
class_counts = np.bincount(labels_train)
total_samples = len(labels_train)
class_weights = total_samples / (2.0 * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"[Model] Class balance in train: Human={class_counts[0]}, AI={class_counts[1]}")
print(f"[Model] Calculated Weights: Human (0)={class_weights[0]:.4f}, AI (1)={class_weights[1]:.4f}")

# 3. Custom Trainer Subclass for Weighted CrossEntropyLoss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 4. Define compute_metrics
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, pos_label=1, zero_division=0)
    rec = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    # Calculate FPR (FP / (FP + TN))
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr
    }

# 5. Define Training Arguments
# label_smoothing_factor=0.1 reduces overconfident predictions, directly fighting false positives.
# per_device_train_batch_size=8 and gradient_accumulation_steps=2 makes training OOM-safe on T4.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    label_smoothing_factor=0.1,  # Smooth target logits (0.1)
    fp16=torch.cuda.is_available(), # Mixed precision on T4 GPU
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    seed=SEED,
    logging_steps=100,
    report_to="none"
)

# Initialize Custom Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    class_weights=weights_tensor
)
print("[Trainer] Trainer initialization complete.")

# %% [markdown]
# ## CELL 4: Training Execution
# - Run fine-tuning training loop.
# - Print epoch metrics.

# %%
print("\n[Training] Starting RoBERTa-base fine-tuning (Sequence length = 384)...")
train_result = trainer.train()
print("[Training] Training completed!")
print(train_result.metrics)

# %% [markdown]
# ## CELL 5: Expanded Evaluation & Comparison Table
# - Re-verify active model architecture.
# - Evaluate test set and compute source-specific breakdowns without warnings.
# - Extract top 25 high-confidence False Positives.
# - Print three-way comparison table.

# %%
# =========================================================================
# CRITICAL MODEL VERIFICATION POINT 2 (Post-Training / Pre-Evaluation)
# =========================================================================
print("\n" + "="*80)
print(" VERIFICATION: ACTIVE MODEL LOAD CHECK (PRE-EVALUATION)")
print("-" * 80)
print(f"  - Model Class Name:       {type(model).__name__}")
print(f"  - Tokenizer Class Name:   {type(tokenizer).__name__}")
print(f"  - Model Config Path/Name: {model.config.name_or_path}")
print("="*80)

assert "Roberta" in type(model).__name__, "CRITICAL ERROR: Evaluation model is not RoBERTa!"
assert "Roberta" in type(tokenizer).__name__, "CRITICAL ERROR: Evaluation tokenizer is not RoBERTa!"
assert "roberta" in model.config.name_or_path.lower(), "CRITICAL ERROR: Evaluation config is not roberta-base!"

# Generate Predictions on Test Set
print("\n[Evaluation] Generating predictions for the Test set...")
predictions_output = trainer.predict(tokenized_test)
logits = predictions_output.predictions

# Apply Softmax to get raw probabilities
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
test_preds = np.argmax(logits, axis=-1)
y_true = test_df['label'].values

# Calculate Overall Test Metrics
acc_all = accuracy_score(y_true, test_preds)
fpr_all = 0.0
cm = confusion_matrix(y_true, test_preds)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    fpr_all = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# Calculate Source-by-Source breakdown
print(f"\n[Source Breakdown] Evaluation on Test set:")
test_eval_df = test_df.copy()
test_eval_df['pred'] = test_preds
# Store AI class probabilities
test_eval_df['ai_prob'] = probs[:, 1]

unique_sources = sorted(test_eval_df['source'].unique())

header = f"{'Source / Origin':<32} | {'Rows':<8} | {'Accuracy':<10} | {'FPR':<8}"
print("-" * len(header))
print(header)
print("-" * len(header))

persuade_fpr_roberta = 0.0
for src in unique_sources:
    src_df = test_eval_df[test_eval_df['source'] == src]
    y_true_src = src_df['label'].values
    y_pred_src = src_df['pred'].values
    
    src_acc = accuracy_score(y_true_src, y_pred_src)
    
    # 1. Compute FPR manually to prevent sklearn warnings on single-class subsets
    num_negative = np.sum(y_true_src == 0)
    if num_negative == 0:
        src_fpr = np.nan
        fpr_str = "N/A (no human docs)"
    else:
        s_fp = np.sum((y_true_src == 0) & (y_pred_src == 1))
        src_fpr = s_fp / num_negative
        fpr_str = f"{src_fpr:7.4f}"
        
    if src == "persuade_corpus":
        persuade_fpr_roberta = src_fpr
        
    print(f"{src:<32} | {len(src_df):<8,d} | {src_acc:10.4f} | {fpr_str:<8}")
print("-" * len(header))

# =========================================================================
# CRITICAL ANALYSIS: TOP 25 HIGH-CONFIDENCE FALSE POSITIVES IN PERSUADE
# =========================================================================
print("\n" + "="*80)
print(" ANALYSIS: TOP 25 HIGH-CONFIDENCE FALSE POSITIVES (HUMAN FLAGGED AS AI)")
print("=" * 80)
# Filter for persuade_corpus records that are HUMAN (label=0) but PREDICTED AS AI (pred=1)
false_positives = test_eval_df[
    (test_eval_df['source'] == 'persuade_corpus') & 
    (test_eval_df['label'] == 0) & 
    (test_eval_df['pred'] == 1)
]

if len(false_positives) > 0:
    # Sort descending by the AI probability score (confidence of error)
    top_fps = false_positives.sort_values(by='ai_prob', ascending=False).head(25)
    
    print(f"Total false positives found in persuade_corpus: {len(false_positives)}")
    print(f"Displaying top {len(top_fps)} by AI confidence score:\n")
    
    col_header = f"{'Idx':<6} | {'AI Prob':<8} | {'Text Preview (First 150 chars)':<60}"
    print(col_header)
    print("-" * len(col_header))
    
    for idx, (_, row) in enumerate(top_fps.iterrows(), 1):
        clean_preview = row['text'].replace('\n', ' ').strip()[:147] + "..."
        print(f"{idx:<6} | {row['ai_prob']:8.4f} | {clean_preview:<60}")
else:
    print("Excellent! No False Positives detected in persuade_corpus on the Test set.")
print("=" * 80)

# Render Direct Three-Way Comparison Table
print("\n" + "="*85)
print("        THREE-WAY COMPARISON: BASELINE VS. DISTILBERT VS. ROBERTA-BASE")
print("=" * 85)
print(f"{'Metric':<35} | {'TF-IDF Baseline':<16} | {'DistilBERT':<12} | {'RoBERTa-base (Target)'}")
print("-" * 85)
print(f"{'Overall Test Accuracy':<35} | {'0.9469':<16} | {'0.9722':<12} | {acc_all:.4f}")
print(f"{'Overall Test FPR':<35} | {'0.0729':<16} | {'0.0528':<12} | {fpr_all:.4f}")
print(f"{'persuade_corpus FPR (Unseen Topics)':<35} | {'0.1765':<16} | {'0.1875':<12} | {persuade_fpr_roberta:.4f}")
print("=" * 85)

# %% [markdown]
# ## CELL 6: Save Model & Sentence-Level Scorer
# - Save fine-tuned weights and configurations.
# - Define `predict_sentence_scores` function for Phase 5 sentence highlights.
# - Execute a prediction test.

# %%
print(f"\n[Save] Saving final fine-tuned model and tokenizer to: {FINAL_SAVE_DIR}...")
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)
trainer.save_model(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print("[Save] Final model saved successfully!")

def predict_sentence_scores(text: str, model_obj, tokenizer_obj) -> list[dict]:
    """
    Splits an input document into individual sentences and computes the AI-probability score for each.
    This serves as the foundation for the sentence-level highlights in Phase 5.
    """
    # 1. Segment document into sentences using NLTK (with a robust fallback in case of tokenization issues)
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, AttributeError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
                
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"[predict_sentence_scores] NLTK sentence tokenization failed ({e}). Falling back to regex splitter.")
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
    results = []
    
    # 2. Put model in evaluation mode
    model_obj.eval()
    device = next(model_obj.parameters()).device
    
    # 3. Predict for each sentence
    with torch.no_grad():
        for sent in sentences:
            sent_cleaned = sent.strip()
            if not sent_cleaned:
                continue
                
            # Tokenize sentence
            inputs = tokenizer_obj(
                sent_cleaned, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                padding="max_length", 
                return_tensors="pt"
            )
            # Move tokens to GPU/CPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model_obj(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probability distributions
            probs_tensor = torch.softmax(logits, dim=-1)
            probs_numpy = probs_tensor.cpu().numpy()[0]
            prob_ai = float(probs_numpy[1])  # Class 1 = AI Probability
            
            results.append({
                "sentence": sent_cleaned,
                "score": prob_ai
            })
            
    return results

# Inference Verification Run on a mock document containing mixed sources
print("\n[Inference Verification] Running predict_sentence_scores on mock document:")
mock_document = (
    "Studying history is very important for the future of our civilization. "
    "However, this document was generated by a large language model predicting tokens. "
    "Therefore, some sentences look completely robotic while others look natural. "
    "Students should always write their own essays."
)

sentence_predictions = predict_sentence_scores(mock_document, model, tokenizer)
for i, pred in enumerate(sentence_predictions, 1):
    print(f"  Sentence {i}: AI Prob = {pred['score']:.4f} | \"{pred['sentence'][:70]}...\"")
print("\n[Inference Verification] Successfully validated sentence-level pipeline!")
