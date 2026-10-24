# Long Document Evaluation Directory (`eval/long_docs/`)

This directory is designed for evaluating long-context document detection across documents containing 1,000 to 8,000 tokens, comparing:
1. **RoBERTa 512-Token Sliding Window (with Chunk-level Max / Mean / Top-K aggregation)**
2. **ModernBERT 8k Native Full-Context Processing**

### Compliance with Project Rule 2
Per Project Rule 2:
- If full-text human papers or generated full AI papers are not yet provided in `eval/long_docs/human_papers/` and `eval/long_docs/ai_papers/`, **we do not simulate or fabricate synthetic token streams**.
- Instead, the infrastructure provides:
  - `eval/long_docs/download_arxiv.py`: Script to download open-access arXiv papers.
  - `eval/long_docs/ai_paper_generation_template.py`: Standardized prompt template for generating full papers using local or API models.

### Evaluation Workflow
Once papers are populated in `human_papers/` and `ai_papers/`:
Run `eval/long_docs/eval_long_docs.py` to benchmark ModernBERT 8k vs RoBERTa 512 sliding window.
