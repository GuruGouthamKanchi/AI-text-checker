import os
import argparse
from typing import List
from robustness_eval.rewriters import combined_rewrite
from robustness_eval.datasets.adversarial_pairs import AdversarialPair, AdversarialDataset

DEFAULT_SAMPLE_AI_TEXTS = [
    "Furthermore, it is crucial to delve into the multifaceted structure of transformer models. Moreover, meticulous parameter tuning underscores the overall performance improvements.",
    "Leveraging comprehensive natural language processing techniques is paramount for establishing robust baseline metrics. In conclusion, these findings foster deep domain insights.",
    "The intricate tapestry of deep neural architectures unveils pivotal patterns. Consequently, this study illuminates key indicators of synthetic generation.",
    "To demystify these complex systems, we embark on a thorough empirical investigation. It is a testament to the power of modern machine learning frameworks.",
    "The dynamic interplay between perplexity and burstiness yields a holistic model. Seamlessly integrating these metrics fosters superior detection capabilities."
]

def load_ai_training_samples(input_path: str = "data/processed/val.parquet", max_samples: int = 50) -> List[str]:
    """
    Reuses existing AI-labeled training/val dataset without duplicating or modifying it.
    Filters for AI-generated text (label == 1 or generated text).
    """
    if not os.path.exists(input_path):
        # Fallback if input_path doesn't exist
        print(f"[build_dataset] Input dataset path '{input_path}' not found. Using default sample set.")
        return DEFAULT_SAMPLE_AI_TEXTS[:max_samples]

    try:
        import pandas as pd
        if input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
        elif input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            print(f"[build_dataset] Unsupported format '{input_path}'. Using default sample set.")
            return DEFAULT_SAMPLE_AI_TEXTS[:max_samples]

        # Look for label column
        label_col = None
        for col in ["label", "generated", "is_ai"]:
            if col in df.columns:
                label_col = col
                break

        text_col = None
        for col in ["text", "sentence", "content"]:
            if col in df.columns:
                text_col = col
                break

        if label_col and text_col:
            ai_df = df[df[label_col] == 1]
            samples = ai_df[text_col].dropna().astype(str).tolist()
            if samples:
                return samples[:max_samples]

        if text_col:
            samples = df[text_col].dropna().astype(str).tolist()
            return samples[:max_samples]

    except Exception as e:
        print(f"[build_dataset] Exception loading dataset ({e}). Using default sample set.")

    return DEFAULT_SAMPLE_AI_TEXTS[:max_samples]

def build_adversarial_dataset(
    input_path: str = "data/processed/val.parquet",
    output_path: str = "robustness_eval/datasets/adversarial_v1.jsonl",
    max_samples: int = 50,
    rewrite_method: str = "combined_v1"
) -> AdversarialDataset:
    """
    Loads AI-labeled samples, applies combined_rewrite(), and stores the result
    in a namespaced JSONL file separate from production training data.
    """
    ai_texts = load_ai_training_samples(input_path=input_path, max_samples=max_samples)
    dataset = AdversarialDataset()

    print(f"[build_dataset] Processing {len(ai_texts)} AI-labeled samples with combined_rewrite()...")
    for text in ai_texts:
        clean_orig = text.strip()
        if not clean_orig:
            continue

        rewritten = combined_rewrite(clean_orig)
        pair = AdversarialPair(
            original_text=clean_orig,
            rewritten_text=rewritten,
            source_label="ai",
            rewrite_method=rewrite_method,
            metadata={"source_file": input_path}
        )
        dataset.add_pair(pair)

    # Ensure target directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset.save_to_jsonl(output_path)
    print(f"[build_dataset] Successfully saved {len(dataset.pairs)} adversarial pairs to '{output_path}'.")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build adversarial dataset pairs for robustness evaluation.")
    parser.add_argument("--input", type=str, default="data/processed/val.parquet", help="Path to input training/val dataset.")
    parser.add_argument("--output", type=str, default="robustness_eval/datasets/adversarial_v1.jsonl", help="Output .jsonl path.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum AI samples to process.")
    args = parser.parse_args()

    build_adversarial_dataset(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples
    )
