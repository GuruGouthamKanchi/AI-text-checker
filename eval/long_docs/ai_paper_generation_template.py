"""
eval/long_docs/ai_paper_generation_template.py
Prompt template and generation utility for user to produce full-length AI research papers (1k-8k tokens).
Strictly complies with Project Rule 2:
- Does NOT execute or fabricate synthetic outputs unless explicitly triggered with an available model.
- Provides standardized, reproducible prompts for generating long-form papers across topics.
"""

import os
import json

LONG_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
AI_PAPERS_DIR = os.path.join(LONG_DOCS_DIR, "ai_papers")
os.makedirs(AI_PAPERS_DIR, exist_ok=True)

PROMPT_TEMPLATES = {
    "academic_full_paper": """You are an academic researcher writing a comprehensive, formal scientific paper on the topic: "{topic}".
Write a full-length, rigorous manuscript (between 2,000 and 5,000 words) formatted with the following sections:
1. Title and Abstract
2. Introduction and Problem Formulation
3. Related Work and Theoretical Foundations
4. Methodology (Architecture, Mathematical Formulations, Loss Objectives)
5. Experimental Setup and Benchmark Datasets
6. Results, Statistical Significance, and Ablation Studies
7. Discussion, Limitations, and Future Work
8. Conclusion

Maintain a formal, objective, academic tone throughout. Use precise terminology, equations where applicable, and thorough scholarly transitions. Do not truncate or summarize; write out the full prose for each section.""",

    "academic_survey": """Write an exhaustive literature review and taxonomical survey on recent breakthroughs in "{topic}".
The paper must be between 3,000 and 6,000 words, thoroughly comparing different methodologies, detailing tradeoffs, and discussing future directions. Format with structured numbered sections and academic style."""
}

TOPICS = [
    "Diffusion Models for Multimodal Video Generation",
    "Mechanistic Interpretability of Sparse Autoencoders in Large Language Models",
    "Post-Training Quantization Strategies for Edge Inference",
    "Retrieval-Augmented Generation: Mitigating Hallucinations in Medical QA Systems",
    "Zero-Knowledge Proofs for Verifiable Machine Learning Execution"
]

def print_prompts():
    print("=" * 75)
    print(" REPRODUCIBLE AI FULL-PAPER GENERATION PROMPT TEMPLATES (1k - 8k Tokens)")
    print("=" * 75)
    print("Use these standardized prompts with your local or commercial LLM (e.g. Claude, GPT-4, Llama 3):")
    print(f"Save the generated full papers to: {AI_PAPERS_DIR}/<topic_name>.txt\n")

    for i, top in enumerate(TOPICS, 1):
        prompt = PROMPT_TEMPLATES["academic_full_paper"].format(topic=top)
        print(f"--- [Template {i}] Topic: {top} ---")
        print(f"PROMPT (first 250 chars):\n{prompt[:250]}...\n")

if __name__ == "__main__":
    print_prompts()
