# %% [markdown]
# # Phase 6: DeBERTa-v3 Disentangled Attention Multi-LLM Academic AI Detector
# 
# This notebook fine-tunes `microsoft/deberta-v3-large` (or `microsoft/deberta-v3-base`) on sentence and document-level academic AI detection.
# Key architectural improvements of DeBERTa-v3 over RoBERTa:
# 1. **Disentangled Attention**: Represents content and position in separate vectors ($Content \text{--} Content$, $Content \text{--} Position$, $Position \text{--} Content$).
# 2. **Replaced Token Detection (RTD) Pre-training**: Highly sensitive to subtle statistical anomalies and synthetic token transitions in modern LLMs (GPT-4, Claude 3.5, LLaMA-3).
# 3. **Enhanced Mask Decoder (EMD)**: Incorporates absolute positions right before softmax, capturing document structure in PDFs and academic papers.

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

# Download NLTK resources
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except (LookupError, AttributeError):
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[Warning] Failed to download NLTK resource {resource}: {e}")

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    print("[Setup] Mounting Google Drive...")
    drive.mount('/content/drive')
except ImportError:
    print("[Setup] Google Drive mount skipped (running outside Google Colab).")

# Global Configuration
DATA_DIR = "/content/drive/MyDrive/ai-text-detector/data/processed/"
OUTPUT_DIR = "/content/drive/MyDrive/ai-text-detector/models/deberta-v3-checkpoints/"
FINAL_SAVE_DIR = "models/deberta-v3-academic/"

MODEL_NAME = "microsoft/deberta-v3-large"  # High-accuracy disentangled attention backbone
MAX_LENGTH = 128  # Extended token length for rich academic context
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='binary', zero_division=0)
    rec = recall_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

def main():
    print(f"[DeBERTa Trainer] Initializing fine-tuning with backbone '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DeBERTa Trainer] Compute Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    print(f"[DeBERTa Trainer] Model backbone successfully instantiated.")
    print(f"[DeBERTa Trainer] Ready for fine-tuning on academic AI detection dataset.")

if __name__ == "__main__":
    main()
