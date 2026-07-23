# %% [markdown]
# # Phase 5: Sentence-Level Multi-LLM Academic AI Detector
# 
# This notebook is designed to run on Google Colab (T4 GPU) to fine-tune `roberta-base` on a sentence-level academic dataset.
# Key features:
# 1. Trained on **individual sentences** (overcoming document padding mismatch).
# 2. Ingests arXiv text with generations from **Claude-4.5, Gemini-3, GPT-5, and open-source models (GPT-oss)**.
# 3. Optimized with **MAX_LENGTH = 64** for fast, VRAM-safe training (batch size 32).
# 4. Evaluates accuracy and False Positive Rate (FPR) for each LLM generator separately.
# 5. Saves the final model weights to Google Drive under `models/roberta-sentence-academic/`.

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
DATA_DIR = "/content/drive/MyDrive/ai-text-detector/data/processed/"
OUTPUT_DIR = "/content/drive/MyDrive/ai-text-detector/models/roberta-sentence-checkpoints/"
FINAL_SAVE_DIR = "/content/drive/MyDrive/ai-text-detector/models/roberta-sentence-academic-v2/"

MODEL_NAME = "roberta-base"
MAX_LENGTH = 64  # Optimized for individual sentences (under 50 words)
BATCH_SIZE = 64  # Increased batch size for faster processing of 120,000 sentences
EPOCHS = 3
LEARNING_RATE = 3e-5
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print(f"\n[Config] Active Hyperparameters:")
print(f"  - Model Target:             {MODEL_NAME}")
print(f"  - Sentence Max Length:      {MAX_LENGTH} tokens")
print(f"  - Training Batch Size:      {BATCH_SIZE}")
print(f"  - Epochs:                   {EPOCHS}")
print(f"  - Learning Rate:            {LEARNING_RATE}")
print(f"  - Final Save Path:          {FINAL_SAVE_DIR}")

# %% [markdown]
# ## CELL 2: Load and Tokenize Datasets
# - Load train, val, and test splits.
# - Apply RoBERTa fast tokenizer.

# %%
# Resolve paths (local fallback if Google Drive is not mounted)
local_data_dir = "data/processed"
active_data_dir = DATA_DIR if os.path.exists(DATA_DIR) else local_data_dir

train_path = os.path.join(active_data_dir, "train.parquet")
val_path = os.path.join(active_data_dir, "val.parquet")
test_path = os.path.join(active_data_dir, "test.parquet")

if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
    print(f"\n[Error] Preprocessed Parquet splits not found at: {active_data_dir}")
    print("Please run 'python src/prepare_dataset.py' locally first and upload splits to Drive.")
    sys.exit(1)

print(f"\n[Data] Loading Parquet splits from: {active_data_dir}...")
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

print(f"  - Train Set Size: {len(train_df):,d} sentences")
print(f"  - Val Set Size:   {len(val_df):,d} sentences")
print(f"  - Test Set Size:  {len(test_df):,d} sentences")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load Tokenizer
print(f"\n[Tokenizer] Loading tokenizer for: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_function(examples):
    # RoBERTa does not expect or support token_type_ids, fast tokenizer handles this
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length"
    )

print("[Tokenizer] Tokenizing train/val splits...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
print("[Tokenizer] Tokenization complete!")

# %% [markdown]
# ## CELL 3: Model and WeightedTrainer Setup
# - Load model and verify its architecture.
# - Implement custom WeightedTrainer to handle class balance (even though balanced).
# - Set training arguments.

# %%
# Load Sequence Classification Model (Binary Classification)
print(f"\n[Model] Ingesting model checkpoint: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# =========================================================================
# CRITICAL MODEL VERIFICATION POINT 1 (Pre-Training Setup)
# =========================================================================
print("\n" + "="*80)
print(" VERIFICATION: ACTIVE MODEL LOAD CHECK (PRE-TRAINING)")
print("-" * 80)
print(f"  - Model Class Name:       {type(model).__name__}")
print(f"  - Tokenizer Class Name:   {type(tokenizer).__name__}")
print(f"  - Model Config Path/Name: {model.config.name_or_path}")
print("=" * 80)

# Check model architecture class directly
assert "RobertaForSequenceClassification" in type(model).__name__, \
    f"BUG ALERT: Model type is {type(model).__name__}, not RobertaForSequenceClassification! Aborting execution."

# Calculate Class Weights dynamically for loss balancing
labels = train_df['label'].values
class_counts = np.bincount(labels)
total_samples = len(labels)
class_weights = total_samples / (len(class_counts) * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n[Model] Class distribution in Train: Human = {class_counts[0]}, AI = {class_counts[1]}")
print(f"[Model] Computed loss weights: Human = {class_weights[0]:.4f}, AI = {class_weights[1]:.4f}")

# Define Custom Trainer overriding loss computation to support weighted CrossEntropyLoss
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation supporting class weights.
        Compatible with transformers v4.46+ which expects num_items_in_batch.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Global False Positive Rate
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    }

# Setup Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),  # Enable mixed-precision training if GPU is available
    label_smoothing_factor=0.1,      # Regularization to prevent overconfident false positives
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
print("\n[Training] Starting RoBERTa-base sentence fine-tuning...")
train_result = trainer.train()
print("[Training] Training completed!")
print(train_result.metrics)

# %% [markdown]
# ## CELL 5: Multi-LLM Evaluation & Comparison Table
# - Re-verify active model architecture.
# - Evaluate test set and compute source-specific breakdowns (FPR for Claude, Gemini, GPT-5, etc.).
# - Print comparison table.

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
print("=" * 80)

assert "RobertaForSequenceClassification" in type(model).__name__, \
    "BUG ALERT: Model type changed! Active model is not RoBERTa. Aborting evaluation."

print("\n[Evaluation] Generating predictions for the Test set...")
test_dataset_tok = test_dataset.map(tokenize_function, batched=True)
test_predictions = trainer.predict(test_dataset_tok)

# Extract logits and labels
logits = test_predictions.predictions
y_true = test_predictions.label_ids
y_pred = np.argmax(logits, axis=-1)

# Compute soft probabilities
import torch.nn.functional as F
logits_tensor = torch.tensor(logits)
probs_tensor = F.softmax(logits_tensor, dim=-1)
ai_probs = probs_tensor[:, 1].numpy()

# Store predictions back into Test Dataframe
test_eval_df = test_df.copy()
test_eval_df['pred'] = y_pred
test_eval_df['ai_prob'] = ai_probs

# Calculate Overall Test Metrics
acc_all = accuracy_score(y_true, y_pred)
precision_all = precision_score(y_true, y_pred, zero_division=0)
recall_all = recall_score(y_true, y_pred, zero_division=0)
f1_all = f1_score(y_true, y_pred, zero_division=0)

tn_all = np.sum((y_true == 0) & (y_pred == 0))
fpr_all = np.sum((y_true == 0) & (y_pred == 1)) / (np.sum(y_true == 0))

print("\n" + "="*80)
print(" GLOBAL EVALUATION SUMMARY ON TEST SET (SENTENCE-LEVEL)")
print("-" * 80)
print(f"  - Overall Sentence Accuracy:  {acc_all:.4f}")
print(f"  - Overall Precision:          {precision_all:.4f}")
print(f"  - Overall Recall (TPR):       {recall_all:.4f}")
print(f"  - Overall F1-Score:           {f1_all:.4f}")
print(f"  - Overall False Positive Rate: {fpr_all:.4f} ({np.sum((y_true == 0) & (y_pred == 1))} / {np.sum(y_true == 0)})")
print("=" * 80)

# Generate Source breakdown (accuracies and FPR per model: Human, Gemini, Claude, GPT-5)
unique_sources = test_eval_df['source'].unique()
print("\n[Source Breakdown] Evaluation on Test set:")
header = f"{'Source / Model':<32} | {'Rows':<8} | {'Accuracy':<10} | {'FPR':<8}"
print(header)
print("-" * len(header))

for src in unique_sources:
    src_df = test_eval_df[test_eval_df['source'] == src]
    y_true_src = src_df['label'].values
    y_pred_src = src_df['pred'].values
    
    src_acc = accuracy_score(y_true_src, y_pred_src)
    
    # Compute FPR manually
    num_negative = np.sum(y_true_src == 0)
    if num_negative == 0:
        src_fpr = np.nan
        fpr_str = "N/A (no human docs)"
    else:
        s_fp = np.sum((y_true_src == 0) & (y_pred_src == 1))
        src_fpr = s_fp / num_negative
        fpr_str = f"{src_fpr:7.4f}"
        
    print(f"{src:<32} | {len(src_df):<8,d} | {src_acc:10.4f} | {fpr_str:<8}")
print("-" * len(header))

# Render Direct Three-Way Comparison Table
print("\n" + "="*85)
print("        THREE-WAY COMPARISON: BASELINE VS. DISTILBERT VS. ROBERTA-SENTENCE")
print("=" * 85)
print(f"{'Metric':<35} | {'TF-IDF Baseline':<16} | {'DistilBERT (Phase 3)':<20} | {'RoBERTa-Sentence (Target)'}")
print("-" * 85)
print(f"{'Overall Test Accuracy':<35} | {'0.9735':<16} | {'0.9722':<20} | {acc_all:.4f}")
print(f"{'Overall Test FPR':<35} | {'0.0170':<16} | {'0.0528':<20} | {fpr_all:.4f}")
print("=" * 85)

# %% [markdown]
# ## CELL 6: Save Model & Sentence-Level Scorer
# - Save fine-tuned weights and configurations.
# - Define `predict_sentence_scores` function.
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
    Uses a robust fallback in case of NLTK tokenization issues.
    """
    # 1. Segment document into sentences using NLTK
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
    "In this study, we present a novel approach to unsupervised anomaly detection. "
    "We utilize a deep autoencoder trained exclusively on normal samples. "
    "However, this sentence was generated by a large language model predicting tokens to deceive. "
    "Additionally, Claude or Gemini can write scientific abstracts with high academic fluency. "
    "Future work will explore scaling to multi-modal research verification."
)

sentence_predictions = predict_sentence_scores(mock_document, model, tokenizer)
for i, pred in enumerate(sentence_predictions, 1):
    print(f"  Sentence {i}: AI Prob = {pred['score']:.4f} | \"{pred['sentence'][:70]}...\"")
print("\n[Inference Verification] Successfully validated sentence-level pipeline!")
