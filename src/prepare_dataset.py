import os
import sys
import pandas as pd

# Ensure that the src directory is in the python path for absolute/relative import safety
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from clean_text import clean_text
from data_loaders import load_daigt, load_coai_academic, load_hc3
from dedupe import remove_near_duplicates
from split import topic_aware_split

def main():
    print("=" * 70)
    print(" AI-Generated-Text Detector: Preprocessing Pipeline (Phase 1) ")
    print("=" * 70)
    
    # 1. Establish project directory structure
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    daigt_path = os.path.join(raw_dir, "daigt_v2_train.csv")
    dfs = []
    
    # Try loading DAIGT V2
    daigt_loaded = False
    if os.path.exists(daigt_path):
        try:
            df_daigt = load_daigt(daigt_path)
            dfs.append(df_daigt)
            daigt_loaded = True
        except Exception as e:
            print(f"[Error] Failed to load DAIGT V2 dataset: {e}")
    else:
        print(f"\n[Notice] Raw DAIGT file not found at: {daigt_path}")
        print("  - To run with the full dataset, download 'daigt_v2_train.csv' from Kaggle")
        print("    and place it inside the 'data/raw/' directory.")
        print("  - Proceeding with Hugging Face arXiv abstracts dataset only...")
        
    # Load COAI academic dataset
    try:
        df_coai = load_coai_academic()
        if len(df_coai) > 0:
            dfs.append(df_coai)
    except Exception as e:
        print(f"[Error] Failed to load COAI academic dataset: {e}")
        
    # Load HC3 dataset
    try:
        df_hc3 = load_hc3()
        if len(df_hc3) > 0:
            dfs.append(df_hc3)
    except Exception as e:
        print(f"[Error] Failed to load HC3 dataset: {e}")
        
    if not dfs:
        print("\n[CRITICAL] No dataset could be loaded. Please ensure you have internet access for COAI academic dataset.")
        sys.exit(1)
        
    # Concatenate all loaded dataframes
    print("\n[Pipeline] Concatenating loaded datasets...")
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"[Pipeline] Ingested combined dataset size: {len(df_combined)} rows.")
    
    # 2. Clean Text
    print("\n[Pipeline] Cleaning and normalizing text column...")
    df_combined['text'] = df_combined['text'].apply(clean_text)
    
    # Filter out any resulting empty rows
    df_combined = df_combined[df_combined['text'].str.strip() != ""]
    print(f"[Pipeline] Text cleaning completed. Remaining paragraphs: {len(df_combined)}")
    
    # 3. Segment paragraphs into individual sentences using NLTK
    print("\n[Pipeline] Segmenting paragraphs into individual sentences...")
    import nltk
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, AttributeError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
                
    sentence_records = []
    for idx, row in df_combined.iterrows():
        text = row['text']
        label = row['label']
        source = row['source']
        topic_id = row['topic_id']
        
        try:
            sents = nltk.sent_tokenize(text)
        except Exception:
            # Fallback regex split
            import re
            sents = re.split(r'(?<=[.!?])\s+', text)
            sents = [s.strip() for s in sents if s.strip()]
            
        for s in sents:
            s_clean = s.strip()
            # Filter out extremely short sentences (less than 15 characters or 3 words)
            if len(s_clean) >= 15 and len(s_clean.split()) >= 3:
                sentence_records.append({
                    'text': s_clean,
                    'label': label,
                    'source': source,
                    'topic_id': topic_id
                })
                
    df_sentences = pd.DataFrame(sentence_records)
    print(f"[Pipeline] Generated {len(df_sentences)} raw sentences.")
    
    # 4. Exact Deduplication
    print("[Pipeline] Running exact deduplication on sentences...")
    before_dedup = len(df_sentences)
    df_sentences = df_sentences.drop_duplicates(subset=['text'])
    print(f"[Pipeline] Deduplication completed. Removed {before_dedup - len(df_sentences)} duplicate sentences. Remaining: {len(df_sentences)}")
    
    # 5. Stratified Domain Balancing (Sample 20,000 human and 20,000 AI sentences per domain)
    def get_domain(source: str) -> str:
        academic_sources = [
            'human', 'anthropic/claude-haiku-4.5', 'google/gemini-3-flash-preview', 
            'openai/gpt-5-nano', 'gpt-oss-120b', 'openai/gpt-oss-20b'
        ]
        qa_sources = ['hc3_human', 'hc3_chatgpt']
        if source in academic_sources:
            return 'academic'
        elif source in qa_sources:
            return 'qa'
        else:
            return 'essay'
            
    df_sentences['domain'] = df_sentences['source'].apply(get_domain)
    
    print("\n[Pipeline] Stratified Domain Counts before sampling:")
    for d in ['academic', 'essay', 'qa']:
        d_df = df_sentences[df_sentences['domain'] == d]
        h_len = len(d_df[d_df['label'] == 0])
        a_len = len(d_df[d_df['label'] == 1])
        print(f"  - Domain '{d:<8}': Human = {h_len:>6,d} | AI = {a_len:>6,d}")
        
    sampled_dfs = []
    # Heavily focus training on the Academic Journal domain, using essays/QA as minor style regularization
    domain_targets = {
        'academic': 50000,
        'essay': 5000,
        'qa': 5000
    }
    
    for d, target in domain_targets.items():
        d_df = df_sentences[df_sentences['domain'] == d]
        h_sents = d_df[d_df['label'] == 0]
        a_sents = d_df[d_df['label'] == 1]
        
        n_samples = min(target, len(h_sents), len(a_sents))
        print(f"  - Sampling {n_samples:,d} human and {n_samples:,d} AI sentences for domain '{d}'...")
        
        sampled_h = h_sents.sample(n=n_samples, random_state=42)
        sampled_a = a_sents.sample(n=n_samples, random_state=42)
        sampled_dfs.extend([sampled_h, sampled_a])
        
    df_balanced = pd.concat(sampled_dfs, ignore_index=True)
    total_retained = len(df_balanced)
    
    # 6. Report Final Label Balance
    print("\n" + "="*45)
    print(" PREPROCESSED DATASET LABEL BALANCE")
    print("-" * 45)
    num_class = total_retained // 2
    print(f"  Human-written (0): {num_class:>6d} ({num_class/total_retained*100:6.2f}%)")
    print(f"  AI-generated (1):  {num_class:>6d} ({num_class/total_retained*100:6.2f}%)")
    print(f"  Total records:      {total_retained:>6d}")
    print("="*45)
    
    # 7. Topic-Aware Splitting
    train_df, val_df, test_df = topic_aware_split(
        df_balanced, 
        topic_col="topic_id", 
        train_frac=0.8, 
        val_frac=0.1, 
        test_frac=0.1, 
        random_seed=42
    )
    
    # 6. Save Splits as Parquet
    print("\n[Pipeline] Saving splits as Parquet files to data/processed/...")
    
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    
    try:
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        print(f"  - Saved Train split to: {train_path} ({os.path.getsize(train_path)/1024/1024:6.2f} MB)")
        print(f"  - Saved Val split to:   {val_path} ({os.path.getsize(val_path)/1024/1024:6.2f} MB)")
        print(f"  - Saved Test split to:  {test_path} ({os.path.getsize(test_path)/1024/1024:6.2f} MB)")
        print("\n[Pipeline] SUCCESS: Dataset preprocessing completed and written!")
    except Exception as e:
        print(f"[Error] Failed to write processed Parquet files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
