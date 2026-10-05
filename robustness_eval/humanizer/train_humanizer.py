import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any, Optional

class HumanizerDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_text = f"humanize: {pair['input_text']}"
        tgt_text = pair['target_text']

        input_encoding = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

def load_training_pairs(dataset_source: Optional[str] = None) -> List[Dict[str, str]]:
    """Loads paired AI-to-Human text samples from Hugging Face or local adversarial JSONL."""
    pairs = [
        {
            "input_text": "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features.",
            "target_text": "I worked on database query optimization and schema design to cut API response latency by 25% across 10+ core features."
        },
        {
            "input_text": "Furthermore, it is crucial to delve into the multifaceted structure of transformer models.",
            "target_text": "We explored the complex structure of transformer models in detail."
        },
        {
            "input_text": "Leveraging comprehensive natural language processing techniques is paramount for establishing robust baseline metrics.",
            "target_text": "Using thorough natural language processing techniques is key to building strong baseline metrics."
        },
        {
            "input_text": "The experiments were meticulously conducted to foster empirical validity.",
            "target_text": "We carefully ran the experiments to ensure reliable results."
        }
    ]

    # Try loading HuggingFace dataset if available
    try:
        from datasets import load_dataset
        print("[train_humanizer] Attempting to fetch HuggingFace dataset 'dmitva/human_ai_generated_text'...")
        hf_ds = load_dataset("dmitva/human_ai_generated_text", split="train[:50]")
        for item in hf_ds:
            ai_txt = item.get("ai_text") or item.get("generated_text")
            human_txt = item.get("human_text") or item.get("text")
            if ai_txt and human_txt:
                pairs.append({"input_text": ai_txt, "target_text": human_txt})
        print(f"[train_humanizer] Loaded {len(pairs)} paired training samples from HuggingFace.")
    except Exception as e:
        print(f"[train_humanizer] Notice: HuggingFace dataset auto-fetch fallback ({e}). Using curated paired dataset.")

    return pairs

def train_t5_humanizer(
    base_model_name: str = "google/flan-t5-base",
    output_dir: str = "models/t5-humanizer-v1",
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 3e-4
) -> str:
    """
    Fine-tunes a T5 / FLAN-T5 model on AI-to-Human text pairs and saves local checkpoint.
    """
    print(f"[train_humanizer] Starting T5 Humanizer Fine-Tuning...")
    print(f"[train_humanizer] Base model: {base_model_name}")
    print(f"[train_humanizer] Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_humanizer] Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
    except Exception as e:
        print(f"[train_humanizer] Could not load base model '{base_model_name}': {e}. Falling back to 't5-small'.")
        base_model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)

    pairs = load_training_pairs()
    dataset = HumanizerDataset(pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print(f"[train_humanizer] Executing fine-tuning loop for {epochs} epoch(s)...")
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[train_humanizer] Epoch {epoch + 1}/{epochs} Complete. Average Loss: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("=" * 60)
    print(f"[train_humanizer] SUCCESS: T5 Humanizer Model saved to '{output_dir}'.")
    print("=" * 60)

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PyTorch T5 Humanizer Model.")
    parser.add_argument("--base-model", type=str, default="t5-small", help="Base model architecture (t5-small, google/flan-t5-base).")
    parser.add_argument("--output-dir", type=str, default="models/t5-humanizer-v1", help="Output model directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    args = parser.parse_args()

    train_t5_humanizer(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
