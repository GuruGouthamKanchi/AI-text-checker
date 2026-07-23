import os
import re
import hashlib
import pandas as pd
from datasets import load_dataset

def _normalize_topic(topic_str: str) -> str:
    """
    Normalizes a topic or prompt string to a clean identifier.
    Converts to lowercase, removes special characters, and joins with underscores.
    """
    if not isinstance(topic_str, str) or not topic_str.strip():
        return "unknown_topic"
    # Convert to lowercase and remove non-alphanumeric/non-space chars
    cleaned = re.sub(r'[^a-zA-Z0-9\s]+', '', topic_str).strip().lower()
    # Replace whitespace sequences with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:64]  # Truncate to avoid overly long identifiers


def _hash_fallback_topic(text: str) -> str:
    """
    Generates a deterministic hash for the first 10 words of the text.
    Used as a fallback topic ID if no explicit prompt/topic column is found.
    """
    if not isinstance(text, str) or not text.strip():
        return "empty_text_hash"
    words = text.split()[:10]
    words_str = " ".join(words).lower()
    return "hash_" + hashlib.sha256(words_str.encode("utf-8")).hexdigest()[:16]


def load_daigt(path: str) -> pd.DataFrame:
    """
    Loads and standardizes the DAIGT V2 Kaggle dataset.
    
    Standardized schema returned:
    - text: str (raw document content)
    - label: int (0 = human-written, 1 = AI-generated)
    - source: str (sub-source dataset identifier, e.g. "persuade_corpus", "chatgpt")
    - topic_id: str (normalized identifier for the essay topic/prompt to support leak-free splits)
    """
    print(f"\n[DAIGT Loader] Ingesting raw DAIGT V2 dataset from: {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DAIGT raw dataset not found at {path}. "
            "Please ensure you download 'daigt_v2_train.csv' and place it in the raw folder."
        )
    
    # Ingest CSV file
    df = pd.read_csv(path)
    print(f"[DAIGT Loader] Loaded CSV file with shape {df.shape}")
    
    # 1. Defensive Text Column Discovery
    text_cols = ['text', 'essay', 'input_text', 'document']
    text_col = next((c for c in text_cols if c in df.columns), None)
    if not text_col:
        raise ValueError(f"Could not find a text column in DAIGT columns: {list(df.columns)}")
    
    # 2. Defensive Label Column Discovery
    label_cols = ['generated', 'label', 'class', 'is_ai']
    label_col = next((c for c in label_cols if c in df.columns), None)
    if not label_col:
        raise ValueError(f"Could not find a label column in DAIGT columns: {list(df.columns)}")
    
    # 3. Defensive Source Column Discovery
    source_cols = ['source', 'prompt_name', 'prompt_id']
    source_col = next((c for c in source_cols if c in df.columns), None)
    
    # 4. Defensive Topic/Prompt Column Discovery
    topic_cols = ['prompt_name', 'prompt_id', 'prompt', 'instructions']
    topic_col = next((c for c in topic_cols if c in df.columns), None)
    
    # Construct standard dataframe
    standard_df = pd.DataFrame()
    standard_df['text'] = df[text_col].astype(str)
    standard_df['label'] = df[label_col].astype(int)
    
    # Assign source
    if source_col:
        standard_df['source'] = df[source_col].astype(str)
    else:
        standard_df['source'] = "daigt_v2_default"
        
    # Assign topic ID for splitting
    if topic_col:
        standard_df['topic_id'] = df[topic_col].apply(lambda x: _normalize_topic(str(x)))
    else:
        # Fall back to hash of first 10 words to prevent leakage on identical starts
        standard_df['topic_id'] = standard_df['text'].apply(_hash_fallback_topic)
        
    # Print diagnostics
    total = len(standard_df)
    counts = standard_df['label'].value_counts()
    human_cnt = counts.get(0, 0)
    ai_cnt = counts.get(1, 0)
    print(f"[DAIGT Loader] Standardization results:")
    print(f"  - Total records loaded: {total}")
    print(f"  - Human-written records (0): {human_cnt} ({human_cnt/total*100:.1f}%)")
    print(f"  - AI-generated records (1): {ai_cnt} ({ai_cnt/total*100:.1f}%)")
    print(f"  - Unique topics/prompts resolved: {standard_df['topic_id'].nunique()}")
    
    return standard_df


def load_hc3() -> pd.DataFrame:
    """
    Loads and standardizes the Hello-SimpleAI/HC3 dataset from Hugging Face.
    
    Each raw row has a 'question', 'human_answers' list, and 'chatgpt_answers' list.
    We unpack each human and ChatGPT answer into a separate row.
    
    Standardized schema returned:
    - text: str (answer content)
    - label: int (0 = human-written, 1 = AI-generated)
    - source: str ("hc3_human" or "hc3_chatgpt")
    - topic_id: str (normalized question text used as the prompt identifier)
    """
    print("\n[HC3 Loader] Downloading/loading Hello-SimpleAI/HC3 from Hugging Face...")
    
    try:
        # Load the 'all.jsonl' file directly from the HC3 repository
        dataset = load_dataset("json", data_files="https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl")
    except Exception as e:
        print(f"[HC3 Loader] ERROR: Failed to load HC3 from Hugging Face: {e}")
        print("[HC3 Loader] Returning an empty DataFrame as a fallback.")
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic_id'])
    
    # We unpack both human and ChatGPT answers
    records = []
    
    # Standard split name is "train" (HC3 has train only)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    
    for row in dataset[split_name]:
        question = row.get('question', '')
        human_ans_list = row.get('human_answers', [])
        chatgpt_ans_list = row.get('chatgpt_answers', [])
        
        # Normalize question to serve as our topic group identifier
        topic_id = _normalize_topic(question)
        
        # Unpack human answers
        for ans in human_ans_list:
            if isinstance(ans, str) and ans.strip():
                records.append({
                    'text': ans,
                    'label': 0,
                    'source': 'hc3_human',
                    'topic_id': topic_id
                })
                
        # Unpack ChatGPT answers
        for ans in chatgpt_ans_list:
            if isinstance(ans, str) and ans.strip():
                records.append({
                    'text': ans,
                    'label': 1,
                    'source': 'hc3_chatgpt',
                    'topic_id': topic_id
                })
                
    standard_df = pd.DataFrame(records)
    
    # Print diagnostics
    total = len(standard_df)
    if total > 0:
        counts = standard_df['label'].value_counts()
        human_cnt = counts.get(0, 0)
        ai_cnt = counts.get(1, 0)
        print(f"[HC3 Loader] Standardization results:")
        print(f"  - Total records loaded: {total}")
        print(f"  - Human-written answers (0): {human_cnt} ({human_cnt/total*100:.1f}%)")
        print(f"  - AI-generated answers (1): {ai_cnt} ({ai_cnt/total*100:.1f}%)")
        print(f"  - Unique questions resolved: {standard_df['topic_id'].nunique()}")
    else:
        print("[HC3 Loader] Empty dataset loaded.")
        
    return standard_df


def load_arxiv_abstracts() -> pd.DataFrame:
    """
    Loads and standardizes the NicolaiSivesind/ChatGPT-Research-Abstracts dataset from Hugging Face.
    
    Each row has a 'title', a 'real_abstract' (human-written), and a 'generated_abstract' (AI-generated).
    We unpack both human and ChatGPT abstracts into separate rows.
    
    Standardized schema returned:
    - text: str (abstract content)
    - label: int (0 = human-written, 1 = AI-generated)
    - source: str ("arxiv_human" or "arxiv_chatgpt")
    - topic_id: str (normalized research paper title used as the prompt identifier)
    """
    print("\n[ArXiv Loader] Downloading/loading NicolaiSivesind/ChatGPT-Research-Abstracts from Hugging Face...")
    
    try:
        dataset = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts")
    except Exception as e:
        print(f"[ArXiv Loader] ERROR: Failed to load arXiv dataset from Hugging Face: {e}")
        print("[ArXiv Loader] Returning an empty DataFrame as a fallback.")
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic_id'])
    
    records = []
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    
    for row in dataset[split_name]:
        title = row.get('title', '')
        real_abs = row.get('real_abstract', '')
        gen_abs = row.get('generated_abstract', '')
        
        # Normalize the research paper title to serve as our topic group identifier
        topic_id = _normalize_topic(title)
        
        # Unpack human abstract
        if isinstance(real_abs, str) and real_abs.strip():
            records.append({
                'text': real_abs,
                'label': 0,
                'source': 'arxiv_human',
                'topic_id': topic_id
            })
            
        # Unpack ChatGPT abstract
        if isinstance(gen_abs, str) and gen_abs.strip():
            records.append({
                'text': gen_abs,
                'label': 1,
                'source': 'arxiv_chatgpt',
                'topic_id': topic_id
            })
            
    standard_df = pd.DataFrame(records)
    
    # Print diagnostics
    total = len(standard_df)
    if total > 0:
        counts = standard_df['label'].value_counts()
        human_cnt = counts.get(0, 0)
        ai_cnt = counts.get(1, 0)
        print(f"[ArXiv Loader] Standardization results:")
        print(f"  - Total records loaded: {total}")
        print(f"  - Human-written abstracts (0): {human_cnt} ({human_cnt/total*100:.1f}%)")
        print(f"  - AI-generated abstracts (1): {ai_cnt} ({ai_cnt/total*100:.1f}%)")
        print(f"  - Unique papers resolved: {standard_df['topic_id'].nunique()}")
    else:
        print("[ArXiv Loader] Empty dataset loaded.")
        
    return standard_df


def load_coai_academic() -> pd.DataFrame:
    """
    Loads and standardizes the coai/ai-text-detection-training dataset from Hugging Face.
    
    Features in coai dataset:
    - text: raw text content (paragraphs / abstracts)
    - label: 0 (human), 1 (AI)
    - model: generator name (e.g. 'human', 'google/gemini-3-flash-preview', 'anthropic/claude-haiku-4.5', 'openai/gpt-5-nano', 'gpt-oss-120b')
    - source: 'arxiv' or 'arxiv_paraphrase'
    - arxiv_id: identifier of the source arXiv paper (excellent for topic-aware splitting to avoid leakage)
    """
    print("\n[COAI Loader] Downloading/loading coai/ai-text-detection-training from Hugging Face...")
    
    try:
        from datasets import concatenate_datasets
        dataset = load_dataset("coai/ai-text-detection-training")
        combined_ds = concatenate_datasets([dataset['train'], dataset['test']])
    except Exception as e:
        print(f"[COAI Loader] ERROR: Failed to load COAI dataset from Hugging Face: {e}")
        print("[COAI Loader] Returning an empty DataFrame as a fallback.")
        return pd.DataFrame(columns=['text', 'label', 'source', 'topic_id'])
        
    records = []
    for row in combined_ds:
        text = row.get('text', '')
        label = row.get('label', 0)
        model = row.get('model', 'unknown')
        arxiv_id = row.get('arxiv_id', '')
        
        # If arxiv_id is missing, default to a hashed version of the text to prevent invalid topic groupings
        if not isinstance(arxiv_id, str) or not arxiv_id.strip():
            import hashlib
            arxiv_id = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
            
        topic_id = _normalize_topic(arxiv_id)
        
        if isinstance(text, str) and text.strip():
            records.append({
                'text': text,
                'label': label,
                'source': model,  # Map generator model as the 'source' for detailed breakdowns
                'topic_id': topic_id
            })
            
    standard_df = pd.DataFrame(records)
    
    # Print diagnostics
    total = len(standard_df)
    if total > 0:
        print(f"[COAI Loader] Standardization results:")
        print(f"  - Total records loaded: {total}")
        counts = standard_df['label'].value_counts()
        human_cnt = counts.get(0, 0)
        ai_cnt = counts.get(1, 0)
        print(f"  - Human paragraphs (0): {human_cnt} ({human_cnt/total*100:.1f}%)")
        print(f"  - AI paragraphs (1):    {ai_cnt} ({ai_cnt/total*100:.1f}%)")
        print(f"  - Unique papers resolved: {standard_df['topic_id'].nunique()}")
        print("  - Generator models distribution:")
        for model_name, cnt in standard_df['source'].value_counts().items():
            print(f"    * {model_name}: {cnt}")
    else:
        print("[COAI Loader] Empty dataset loaded.")
        
    return standard_df
