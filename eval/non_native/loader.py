"""
eval/non_native/loader.py
Loader for non-native English writing samples (e.g., TOEFL, IELTS, ICLE, or user-supplied essays).
Strictly adheres to Project Rule 2:
- Does NOT fabricate, invent, or simulate non-native writing samples.
- Safely ingests files if provided by the user in this folder.
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional

NON_NATIVE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_non_native_samples() -> pd.DataFrame:
    """
    Scans eval/non_native/ for user-supplied text files (.txt, .csv, .parquet, .jsonl).
    Returns a DataFrame with columns: ['sample_id', 'text', 'source', 'label'] where label=0 (human).
    If no samples are found, returns an empty DataFrame with a clear notification.
    """
    records = []

    # 1. Check for parquet files
    parquet_files = glob.glob(os.path.join(NON_NATIVE_DIR, "*.parquet"))
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if 'text' in df.columns:
                for idx, row in df.iterrows():
                    records.append({
                        "sample_id": f"non_native_pq_{os.path.basename(pf)}_{idx}",
                        "text": row['text'],
                        "source": row.get('source', 'user_non_native_parquet'),
                        "label": 0
                    })
        except Exception as e:
            print(f"[NonNativeLoader] Warning reading {pf}: {e}")

    # 2. Check for csv files
    csv_files = glob.glob(os.path.join(NON_NATIVE_DIR, "*.csv"))
    for cf in csv_files:
        try:
            df = pd.read_csv(cf)
            if 'text' in df.columns:
                for idx, row in df.iterrows():
                    records.append({
                        "sample_id": f"non_native_csv_{os.path.basename(cf)}_{idx}",
                        "text": row['text'],
                        "source": row.get('source', 'user_non_native_csv'),
                        "label": 0
                    })
        except Exception as e:
            print(f"[NonNativeLoader] Warning reading {cf}: {e}")

    # 3. Check for raw txt files
    txt_files = glob.glob(os.path.join(NON_NATIVE_DIR, "*.txt"))
    for tf in txt_files:
        fname = os.path.basename(tf)
        if fname.lower().startswith("readme"):
            continue
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    records.append({
                        "sample_id": f"non_native_txt_{fname}",
                        "text": content,
                        "source": "user_non_native_txt",
                        "label": 0
                    })
        except Exception as e:
            print(f"[NonNativeLoader] Warning reading {tf}: {e}")

    if not records:
        print("[NonNativeLoader] No user-supplied non-native English writing samples found in eval/non_native/.")
        print("[NonNativeLoader] Per Project Rule 2, no synthetic samples are fabricated. Returning empty DataFrame.")
        return pd.DataFrame(columns=['sample_id', 'text', 'source', 'label'])

    out_df = pd.DataFrame(records)
    print(f"[NonNativeLoader] Loaded {len(out_df)} non-native English writing samples.")
    return out_df

if __name__ == "__main__":
    df = load_non_native_samples()
    print(f"Sample count: {len(df)}")
