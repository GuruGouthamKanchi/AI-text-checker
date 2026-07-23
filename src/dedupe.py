import re
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def remove_near_duplicates(
    df: pd.DataFrame, 
    text_col: str = "text", 
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    Identifies and removes duplicate and near-duplicate rows from a DataFrame.
    
    Processing Steps:
    1. First Pass: Exact duplicate removal using pandas `drop_duplicates` (highly optimized and fast).
    2. Second Pass: Fuzzy near-duplicate removal using MinHash and Locality Sensitive Hashing (LSH).
       - Tokenizes text into word 3-grams (shingles).
       - Builds 128-permutation MinHash signatures.
       - Uses MinHashLSH to query for candidates with Jaccard similarity >= threshold.
       - Validates candidates with a strict Jaccard check to prevent false positives.
    
    Parameters:
    - df: Input pandas DataFrame
    - text_col: The column containing text to deduplicate
    - threshold: Jaccard similarity threshold (0.0 to 1.0) above which text is considered a duplicate.
    
    Returns:
    - Cleaned DataFrame with duplicate rows removed.
    """
    initial_count = len(df)
    if initial_count == 0:
        return df

    # --- Phase 1: Exact Duplicate Removal ---
    print(f"\n[Dedupe] Initiating exact duplicate check on column '{text_col}'...")
    df_exact_cleaned = df.drop_duplicates(subset=[text_col]).copy()
    exact_dup_count = initial_count - len(df_exact_cleaned)
    print(f"[Dedupe] Removed {exact_dup_count} exact duplicates. Remaining: {len(df_exact_cleaned)}")
    
    if len(df_exact_cleaned) == 0:
        return df_exact_cleaned

    # --- Phase 2: Near-Duplicate Removal ---
    print(f"[Dedupe] Initiating fuzzy near-duplicate check (Jaccard similarity >= {threshold})...")
    
    num_perm = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # Reset index for clean lookup and list tracking
    df_temp = df_exact_cleaned.reset_index(drop=True)
    
    keep_indices = []
    # Dictionary to cache the token shingle sets of the kept documents to allow precise Jaccard checks
    kept_shingles = {}
    
    for idx, row in tqdm(df_temp.iterrows(), total=len(df_temp), desc="Fuzzy deduplication"):
        text = str(row[text_col])
        
        # 1. Shingle generation (word 3-grams)
        # Strip punctuation for robust similarity mapping
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()
        if len(words) < 3:
            shingles = set(words)
        else:
            shingles = set(" ".join(words[i:i+3]) for i in range(len(words)-2))
            
        if not shingles:
            # Keep empty rows or let them through if any (though usually removed by exact check)
            keep_indices.append(idx)
            continue
            
        # 2. MinHash signature generation
        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
            
        # 3. Query LSH for similar candidates
        candidates = lsh.query(m)
        
        # 4. Strict verification check (LSH is probabilistic; checking Jaccard prevents false positives)
        is_duplicate = False
        for cand_idx in candidates:
            cand_shingles = kept_shingles[cand_idx]
            
            # Calculate actual Jaccard Similarity
            intersection = len(shingles.intersection(cand_shingles))
            union = len(shingles.union(cand_shingles))
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            if jaccard_sim >= threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            keep_indices.append(idx)
            # Insert into LSH and cache shingles for future comparisons
            lsh.insert(idx, m)
            kept_shingles[idx] = shingles
            
    deduped_df = df_temp.iloc[keep_indices].reset_index(drop=True)
    fuzzy_dup_count = len(df_temp) - len(deduped_df)
    total_removed = initial_count - len(deduped_df)
    
    print(f"[Dedupe] Fuzzy deduplication completed:")
    print(f"  - Near-duplicates removed: {fuzzy_dup_count}")
    print(f"  - Total records retained: {len(deduped_df)} (Removed {total_removed} rows in total, {total_removed/initial_count*100:.1f}% reduction)")
    
    return deduped_df
