import re
import unicodedata
import nltk

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text for ML consumption:
    - Normalizes Unicode representations using NFKC.
    - Removes Unicode control characters (except tabs and newlines).
    - Maps smart quotes and dashes to standard ASCII representations.
    - Strips HTML tags and common markdown markers (bold, italic, links, headers).
    - Normalizes multiple spaces and excessive newlines.
    - Strips leading and trailing whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize Unicode (NFKC handles some spacing anomalies and composition)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Strip control characters (Cc category) while preserving tabs and newlines
    text = "".join(
        ch for ch in text 
        if unicodedata.category(ch) != "Cc" or ch in ("\n", "\r", "\t")
    )
    
    # 3. Normalize smart quotes and dashes to standard ASCII equivalents
    quote_map = {
        '“': '"', '”': '"', '″': '"', '‟': '"',
        '‘': "'", '’': "'", '′': "'", '‛': "'",
        '–': '-', '—': '-', '−': '-'
    }
    for q, r in quote_map.items():
        text = text.replace(q, r)
        
    # 4. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 5. Remove common markdown artifacts
    # Strip headers (e.g. "# Heading")
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Strip bold/italic wrappers (e.g. **bold**, _italic_)
    text = re.sub(r'\*\*|__|\*|_', '', text)
    # Strip links: [label](url) -> label
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove markdown code block markers but keep content
    text = re.sub(r'```(?:[a-zA-Z0-9]+)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 6. Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs and multiple horizontal spaces with a single space
    text = re.sub(r'[ \t\u00a0\u2000-\u200a]+', ' ', text)
    # Normalize multiple newlines (3 or more down to 2, preserving paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def _ensure_nltk_punkt():
    """
    Programmatically downloads NLTK's punkt tokenizer resource if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, AttributeError):
        try:
            # Download punkt silently
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"[Warning] Failed to automatically download NLTK 'punkt' package: {e}")
            print("[Warning] A regex fallback will be used if NLTK sentence splitting fails.")


# Ensure punkt is loaded
_ensure_nltk_punkt()


def split_into_sentences(text: str) -> list[str]:
    """
    Splits a document text into a list of constituent sentences.
    
    Why NLTK sent_tokenize?
    1. Lightweight: Avoids installing large neural model pipelines (e.g., SpaCy's en_core_web_sm) 
       which consume hundreds of megabytes and slow down initialization.
    2. CPU Efficient: Uses the unsupervised Punkt sentence tokenizer algorithm, which is highly 
       accurate for standard prose (emails, essays, reports) and runs quickly on CPU.
    3. Self-contained: Easy to reuse or package as a single function for Google Colab/scripts.
    """
    if not text or not text.strip():
        return []
    
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback regex sentence splitter if NLTK resource download was blocked/offline
        # Splits on period/exclamation/question mark followed by whitespace and a capital letter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
