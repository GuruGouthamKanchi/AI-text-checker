# %% [markdown]
# # Phase 3: Transformer Model Fine-Tuning (DistilBERT Baseline)
# 
# This notebook is designed to run on Google Colab (utilizing a free T4 GPU) to fine-tune `distilbert-base-uncased` on the preprocessed dataset.
# The objective is to validate our Hugging Face training pipeline, compute class-weighted loss to counter class imbalance, and evaluate generalization performance on unseen topics (especially looking for a drop in the `persuade_corpus` False Positive Rate compared to our TF-IDF baseline).

# %% [markdown]
# ## CELL 1: Environment Setup & Configurations
# - Install Hugging Face and PyTorch packages.
# - Mount Google Drive to load data and save model checkpoints.
# - Configure training hyperparameters.

# %%
# 1. Install required packages in Google Colab (if running in Colab)
# Uncomment the line below when running in Google Colab:
# !pip install -q transformers[torch] datasets accelerate evaluate pandas pyarrow scikit-learn

import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 2. Mount Google Drive
try:
    from google.colab import drive
    print("[Setup] Mounting Google Drive...")
    drive.mount('/content/drive')
except ImportError:
    print("[Setup] Google Drive mount skipped (running outside Google Colab environment).")

# 3. Global Hyperparameters and Paths Configuration
# Adjust paths to match your Google Drive structure.
DATA_DIR = "/content/drive/MyDrive/ai-text-detector/data/processed/"
OUTPUT_DIR = "/content/drive/MyDrive/ai-text-detector/models/distilbert-checkpoints/"
MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
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
# - Load train, validation, and test Parquet splits.
# - Convert pandas DataFrames into Hugging Face `Dataset` structures.
# - Apply tokenization and padding.

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

print(f"  - Loaded Train: {len(train_df):,} rows")
print(f"  - Loaded Val:   {len(val_df):,} rows")
print(f"  - Loaded Test:  {len(test_df):,} rows")

# Convert to HF Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load Tokenizer
print(f"\n[Tokenizer] Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Tokenize text with truncation and padding to MAX_LENGTH
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
# ## CELL 3: Model, Custom Trainer & Metrics Setup
# - Load sequence classification model.
# - Compute dynamic class weights to address 64/36 class imbalance.
# - Build custom Trainer subclass with weighted CrossEntropyLoss.
# - Formulate evaluation metrics (specifically computing FPR).

# %%
import torch.nn as nn
from transformers import EvalPrediction

# 1. Initialize model
print(f"\n[Model] Ingesting model checkpoint {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 2. Compute Class Weights for Imbalance Correction
labels_train = train_df['label'].values
class_counts = np.bincount(labels_train)
total_samples = len(labels_train)
# Inverse frequency scaling
class_weights = total_samples / (2.0 * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"[Model] Class balance in train: Human={class_counts[0]} ({class_counts[0]/total_samples*100:.1f}%), AI={class_counts[1]} ({class_counts[1]/total_samples*100:.1f}%)")
print(f"[Model] Calculated Class Weights: Human (0)={class_weights[0]:.4f}, AI (1)={class_weights[1]:.4f}")

# 3. Custom Trainer Subclass to support CrossEntropy loss weighting
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights tensor to model training device
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply weights tensor to CrossEntropyLoss
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 4. Define compute_metrics containing False Positive Rate (FPR)
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
# Metric for best model is tied to F1-score to maximize detection precision/recall, 
# but validation loss or FPR are also monitored.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Mixed precision on T4 GPU
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    seed=SEED,
    logging_steps=100,
    report_to="none"  # Disable online reporting tools (Weights & Biases, etc.)
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
print("\n[Training] Starting DistilBERT fine-tuning...")
train_result = trainer.train()
print("[Training] Training completed!")
print(train_result.metrics)

# %% [markdown]
# ## CELL 5: Test Evaluation & Baseline Comparison
# - Generate predictions on Test Set.
# - Break down Accuracy and FPR by individual data source.
# - Render a direct comparison table between the TF-IDF Baseline and DistilBERT.

# %%
print("\n[Evaluation] Generating predictions for the Test set...")
predictions_output = trainer.predict(tokenized_test)
logits = predictions_output.predictions
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

unique_sources = sorted(test_eval_df['source'].unique())

header = f"{'Source / Origin':<32} | {'Rows':<8} | {'Accuracy':<10} | {'FPR':<8}"
print("-" * len(header))
print(header)
print("-" * len(header))

persuade_fpr_distilbert = 0.0
for src in unique_sources:
    src_df = test_eval_df[test_eval_df['source'] == src]
    y_true_src = src_df['label']
    y_pred_src = src_df['pred']
    
    src_acc = accuracy_score(y_true_src, y_pred_src)
    
    # Calculate source FPR
    src_cm = confusion_matrix(y_true_src, y_pred_src)
    if src_cm.shape == (2, 2):
        s_tn, s_fp, s_fn, s_tp = src_cm.ravel()
        src_fpr = s_fp / (s_fp + s_tn) if (s_fp + s_tn) > 0 else 0.0
    else:
        src_y_true = np.array(y_true_src)
        src_y_pred = np.array(y_pred_src)
        s_tn = np.sum((src_y_true == 0) & (src_y_pred == 0))
        s_fp = np.sum((src_y_true == 0) & (src_y_pred == 1))
        src_fpr = s_fp / (s_fp + s_tn) if (s_fp + s_tn) > 0 else np.nan
        
    fpr_str = f"{src_fpr:7.4f}" if not np.isnan(src_fpr) else "N/A"
    
    # Track persuade_corpus specifically for comparison
    if src == "persuade_corpus":
        persuade_fpr_distilbert = src_fpr
        
    print(f"{src:<32} | {len(src_df):<8,d} | {src_acc:10.4f} | {fpr_str:<8}")
print("-" * len(header))

# Render Direct Before/After Comparison Table
# Baseline metrics are pulled from Phase 2 results.
print("\n" + "="*80)
print("             COMPARISON: TF-IDF BASELINE VS. DISTILBERT TRANSFORMER")
print("=" * 80)
print(f"{'Metric':<35} | {'TF-IDF Baseline':<18} | {'DistilBERT Transformer'}")
print("-" * 80)
print(f"{'Overall Test Accuracy':<35} | {'0.9469':<18} | {acc_all:.4f}")
print(f"{'Overall Test FPR':<35} | {'0.0729':<18} | {fpr_all:.4f}")
print(f"{'persuade_corpus FPR (Unseen Topics)':<35} | {'0.1765':<18} | {persuade_fpr_distilbert:.4f}")
print("=" * 80)

# %% [markdown]
# ## CELL 6: Save Model & Tokenizer
# - Save fine-tuned model weights and tokenizer configurations.
# - Output files to Google Drive folder for future reload.

# %%
final_save_path = os.path.join(OUTPUT_DIR, "distilbert-final")
print(f"\n[Save] Saving final fine-tuned model and tokenizer to: {final_save_path}...")
os.makedirs(final_save_path, exist_ok=True)

trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("[Save] Final model saved successfully!")
