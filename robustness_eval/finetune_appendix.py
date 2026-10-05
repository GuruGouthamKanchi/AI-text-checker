import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, List, Dict
from robustness_eval.config import DEFAULT_CONFIG, RobustnessConfig
from robustness_eval.datasets.adversarial_pairs import AdversarialDataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def load_fine_tuning_data(
    adversarial_path: str = "robustness_eval/datasets/adversarial_v1.jsonl"
) -> List[Dict[str, int]]:
    sample_data = []

    human_texts = [
        "The experiments were conducted in accordance with institutional guidelines and approved protocols.",
        "We measured the temperature variance across five independent trial runs under constant pressure.",
        "These initial observations suggest that further empirical investigation is warranted.",
        "The data collection procedure followed standard double-blind review criteria.",
        "Figure 3 illustrates the relationship between input voltage and resulting signal distortion."
    ]
    for text in human_texts:
        sample_data.append({"text": text, "label": 0})

    if os.path.exists(adversarial_path):
        adv_ds = AdversarialDataset.load_from_jsonl(adversarial_path)
        for pair in adv_ds.pairs:
            sample_data.append({"text": pair.original_text, "label": 1})
            sample_data.append({"text": pair.rewritten_text, "label": 1})
    else:
        ai_texts = [
            "Furthermore, it is crucial to delve into the multifaceted structure of transformer models.",
            "Leveraging comprehensive natural language processing techniques is paramount for establishing robust baseline metrics.",
            "The intricate tapestry of deep neural architectures unveils pivotal patterns."
        ]
        for text in ai_texts:
            sample_data.append({"text": text, "label": 1})

    return sample_data

def finetune_robust_model(
    base_model_path: str = "models/modernbert-academic",
    adversarial_data_path: str = "robustness_eval/datasets/adversarial_v1.jsonl",
    output_model_dir: str = "models/modernbert-academic-robust",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5
) -> str:
    """
    Continues fine-tuning the base model checkpoint on a mix of original + adversarial examples.
    Saves the result as a new, separate model artifact without overwriting the currently deployed model file.
    """
    print(f"[finetune_appendix] Starting robust fine-tuning...")
    print(f"[finetune_appendix] Base model path: {base_model_path}")
    print(f"[finetune_appendix] Output artifact directory: {output_model_dir}")

    if not os.path.exists(base_model_path):
        print(f"[finetune_appendix] Local model '{base_model_path}' not found. Falling back to 'answerdotai/ModernBERT-base'.")
        base_model_path = "answerdotai/ModernBERT-base"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_list = load_fine_tuning_data(adversarial_path=adversarial_data_path)
    print(f"[finetune_appendix] Loaded {len(data_list)} training samples (mixed original + adversarial).")

    texts = [item["text"] for item in data_list]
    labels = [item["label"] for item in data_list]

    dataset = TextClassificationDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    print(f"[finetune_appendix] Executing pure PyTorch fine-tuning loop on {device}...")
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[finetune_appendix] Epoch {epoch + 1}/{epochs} Complete. Average Loss: {avg_loss:.4f}")

    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print("=" * 60)
    print(f"[finetune_appendix] SUCCESS: New robust model artifact saved to:")
    print(f"                    {output_model_dir}")
    print(f"[finetune_appendix] NOTE: The deployed production model in '{base_model_path}' remains unmodified.")
    print(f"[finetune_appendix] To promote this model to production, update 'model_dir' in src/app.py.")
    print("=" * 60)

    return output_model_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune robust model artifact on adversarial rewrites.")
    parser.add_argument("--base-model", type=str, default="models/modernbert-academic", help="Base checkpoint path.")
    parser.add_argument("--adversarial-data", type=str, default="robustness_eval/datasets/adversarial_v1.jsonl", help="Adversarial dataset path.")
    parser.add_argument("--output-dir", type=str, default="models/modernbert-academic-robust", help="Output directory for new model artifact.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size.")
    args = parser.parse_args()

    finetune_robust_model(
        base_model_path=args.base_model,
        adversarial_data_path=args.adversarial_data,
        output_model_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
