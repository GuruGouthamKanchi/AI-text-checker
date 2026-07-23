import numpy as np
import pandas as pd

def topic_aware_split(
    df: pd.DataFrame, 
    topic_col: str = "topic_id", 
    train_frac: float = 0.8, 
    val_frac: float = 0.1, 
    test_frac: float = 0.1, 
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train, val, and test splits by unique topic_id values.
    This guarantees that no topic/prompt appears in more than one split, preventing data leakage.
    
    The algorithm:
    1. Extracts all unique topic_ids.
    2. Shuffles them using a fixed random seed.
    3. Allocates topics to splits on-the-fly using a deficit-balancing strategy
       to ensure the resulting row counts closely match the target ratios.
    4. Asserts that the splits are disjoint.
    
    Parameters:
    - df: Input pandas DataFrame.
    - topic_col: Name of the column containing topic/prompt identifiers.
    - train_frac: Fraction of data rows targeted for training (default 0.8).
    - val_frac: Fraction of data rows targeted for validation (default 0.1).
    - test_frac: Fraction of data rows targeted for testing (default 0.1).
    - random_seed: Seed for reproducibility.
    
    Returns:
    - A tuple of (train_df, val_df, test_df)
    """
    # Verify fractions sum to 1.0 (approx)
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, currently: {total_frac}")
        
    total_rows = len(df)
    if total_rows == 0:
        return df.copy(), df.copy(), df.copy()
        
    print(f"\n[Split] Initiating topic-aware group split (Train: {train_frac}, Val: {val_frac}, Test: {test_frac})...")
    
    # 1. Gather unique topics and their size (row count) distributions
    topic_counts = df[topic_col].value_counts().to_dict()
    unique_topics = list(topic_counts.keys())
    
    # 2. Shuffle unique topics deterministically
    rng = np.random.RandomState(random_seed)
    shuffled_topics = list(unique_topics)
    rng.shuffle(shuffled_topics)
    
    # 3. Deficit-balancing allocation
    train_topics, val_topics, test_topics = [], [], []
    train_rows, val_rows, test_rows = 0, 0, 0
    
    train_target = train_frac * total_rows
    val_target = val_frac * total_rows
    test_target = test_frac * total_rows
    
    for topic in shuffled_topics:
        size = topic_counts[topic]
        
        # Calculate how much each split deviates from its absolute row target
        d_train = train_target - train_rows
        d_val = val_target - val_rows
        d_test = test_target - test_rows
        
        # Assign to the partition with the greatest remaining row deficit
        max_deficit = max(d_train, d_val, d_test)
        if max_deficit == d_train:
            train_topics.append(topic)
            train_rows += size
        elif max_deficit == d_val:
            val_topics.append(topic)
            val_rows += size
        else:
            test_topics.append(topic)
            test_rows += size
            
    # 4. Generate the split DataFrames
    train_df = df[df[topic_col].isin(train_topics)].reset_index(drop=True)
    val_df = df[df[topic_col].isin(val_topics)].reset_index(drop=True)
    test_df = df[df[topic_col].isin(test_topics)].reset_index(drop=True)
    
    # 5. Sanity Check / Disjointness Assertions
    train_topic_set = set(train_df[topic_col].unique())
    val_topic_set = set(val_df[topic_col].unique())
    test_topic_set = set(test_df[topic_col].unique())
    
    # Assert pairwise intersections are completely empty (no leakage)
    assert train_topic_set.isdisjoint(val_topic_set), "CRITICAL ERROR: Topic overlap found between Train and Val!"
    assert train_topic_set.isdisjoint(test_topic_set), "CRITICAL ERROR: Topic overlap found between Train and Test!"
    assert val_topic_set.isdisjoint(test_topic_set), "CRITICAL ERROR: Topic overlap found between Val and Test!"
    
    # Print detailed stats
    print(f"[Split] Topic-aware split completed successfully:")
    print(f"  - Train Set: {len(train_df)} rows ({len(train_df)/total_rows*100:.2f}%) | {len(train_topic_set)} unique topics")
    print(f"  - Val Set:   {len(val_df)} rows ({len(val_df)/total_rows*100:.2f}%) | {len(val_topic_set)} unique topics")
    print(f"  - Test Set:  {len(test_df)} rows ({len(test_df)/total_rows*100:.2f}%) | {len(test_topic_set)} unique topics")
    print(f"  - Verification: Split overlap assert passed. Total output rows: {len(train_df) + len(val_df) + len(test_df)} / {total_rows}")
    
    return train_df, val_df, test_df
