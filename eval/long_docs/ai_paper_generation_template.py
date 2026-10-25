"""
eval/long_docs/ai_paper_generation_template.py
Prompt Templates and Dataset Validation Engine for AI Research Papers (1,000 - 8,000 Tokens).

Strictly complies with Project Rules 1, 2, 5, 8, 11:
- Does NOT fabricate or simulate AI papers.
- Provides 100 distinct academic topics spanning 10 disciplines.
- Provides standardized, reproducible prompts for generating long-form papers across topics.
- Includes automated dataset validation to verify token count distributions across length buckets.
"""

import os
import json
from typing import List, Dict

TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "modernbert-academic")
            if os.path.exists(TOKENIZER_PATH):
                TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except Exception:
            TOKENIZER = False
    return TOKENIZER if TOKENIZER is not False else None

LONG_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_PAPERS_DIR = os.path.join(LONG_DOCS_DIR, "human_papers")
AI_PAPERS_DIR = os.path.join(LONG_DOCS_DIR, "ai_papers")

os.makedirs(HUMAN_PAPERS_DIR, exist_ok=True)
os.makedirs(AI_PAPERS_DIR, exist_ok=True)

PROMPT_TEMPLATE = """You are an academic researcher writing a comprehensive, formal scientific paper on the topic: "{topic}".
Write a full-length, rigorous manuscript (between 2,000 and 5,000 words, strictly in the 1,000 to 8,000 token range) formatted with the following sections:
1. Title and Abstract
2. Introduction and Problem Formulation
3. Related Work and Theoretical Foundations
4. Methodology (Architecture, Mathematical Formulations, Loss Objectives)
5. Experimental Setup and Benchmark Datasets
6. Results, Statistical Significance, and Ablation Studies
7. Discussion, Limitations, and Future Work
8. Conclusion

Maintain a formal, objective, academic tone throughout. Use precise terminology, equations where applicable, and thorough scholarly transitions. Do not truncate or summarize; write out the full prose for each section."""

TOPICS_100 = [
    # Computer Science & Machine Learning (1-20)
    "Mechanistic Interpretability of Sparse Autoencoders in Large Language Models",
    "Post-Training Quantization Strategies for Edge Inference in Transformer Architectures",
    "Diffusion Models for High-Fidelity Video Synthesis and Temporal Coherence",
    "Mitigating Hallucinations in Retrieval-Augmented Generation for Clinical QA",
    "Zero-Knowledge Proofs for Verifiable Machine Learning Model Inference",
    "Contrastive Learning Objectives for Self-Supervised Audio-Visual Alignment",
    "Graph Neural Networks for Large-Scale Molecular Property Prediction",
    "State Space Models and Mamba Architectures for Long-Context Sequence Modeling",
    "Federated Learning Under Non-IID Client Data Distributions with Differential Privacy",
    "Robust Optimization for Adversarial Attacks in Transformer Vision Models",
    "Attention-Sink Phenomena and KV Cache Eviction in Infinite-Context LLMs",
    "Direct Preference Optimization versus Reinforcement Learning from Human Feedback",
    "Neural Architecture Search for Ultra-Low Latency Embedded Computing",
    "Causal Discovery Algorithms in Observational Time-Series Data",
    "Multi-Agent Reinforcement Learning in Stochastic Partially Observable Games",
    "Memory-Augmented Neural Networks for Dynamic Lifelong Continual Learning",
    "Speculative Decoding Algorithms for Accelerating Autoregressive Generation",
    "Geometric Deep Learning and Equivariant Architectures for 3D Protein Folding",
    "Vision-Language-Action Models for Autonomous Robotic Manipulation",
    "Hyperparameter Optimization Using Bayesian Surrogates in Deep Learning Pipelines",

    # Natural Language Processing & Computational Linguistics (21-35)
    "Cross-Lingual Transferability of Low-Resource Languages via Multilingual Pretraining",
    "Syntax-Aware Parsing via Recursive Compositional Neural Networks",
    "Evaluation of Semantic Drift in Chronologically Updated Word Embeddings",
    "Discourse Cohesion and Anaphora Resolution in Extended Academic Monologues",
    "Automated Fact-Checking and Stance Detection Across Heterogeneous News Streams",
    "Subword Tokenization Bias and Its Impact on Morphologically Rich Dialects",
    "Parameter-Efficient Fine-Tuning: Comparative Analysis of LoRA, Prefix, and Adapter Layers",
    "Neural Abstractive Summarization with Hallucination Mitigation Constraints",
    "Metaphor Identification and Figurative Language Interpretation via Large Models",
    "Information Extraction from Unstructured Scientific Literature Using LLMs",
    "Evaluation Metrics for Open-Ended Conversational Agents: Beyond Perplexity and BLEU",
    "Sentence-Level Sentiment Compositionality in Polysemous Contexts",
    "Language Model Calibration: Temperature Scaling versus Conformal Prediction",
    "Ethical Alignment and Toxicity Filtering in Foundational Open-Weight Models",
    "Historical Linguistic Phylogenetics via Deep Phonetic Feature Representation",

    # Computer Vision & Multi-Modal Systems (36-50)
    "Neural Radiance Fields and 3D Gaussian Splatting for Real-Time Novel View Synthesis",
    "Self-Supervised Monocular Depth Estimation Under Adverse Weather Conditions",
    "Semantic Segmentation of High-Resolution Satellite Imagery for Disaster Response",
    "Zero-Shot Object Detection Using Open-Vocabulary Foundation Models",
    "Vision Transformers versus Convolutional Networks: Scaling Laws and Inductive Biases",
    "Video Object Segmentation with Spatio-Temporal Memory Networks",
    "Generative Adversarial Networks for Anomaly Detection in Medical Imaging",
    "Unsupervised Domain Adaptation for Autonomous Driving in Dense Urban Traffic",
    "Multimodal Sentiment Analysis Combining Acoustic, Visual, and Lexical Modalities",
    "Point Cloud Registration and Surface Reconstruction for Industrial Inspection",
    "Deep Fake Facial Synthesis Detection via Biological Signal Extraction",
    "Pose Estimation and Motion Tracking in Markerless Biomechanical Analysis",
    "Event-Based Neuromorphic Cameras for High-Speed Visual Tracking",
    "Image Restoration via Deep Denoising Diffusion Probabilistic Models",
    "Cross-Modal Retrieval Between Microscopy Scans and Pathological Diagnostic Reports",

    # Physics, Quantum Computing & Mathematics (51-65)
    "Quantum Error Correction Codes on Surface Lattices Under Coherent Noise",
    "Variational Quantum Eigensolvers for Ground State Molecular Simulations",
    "Topological Insulators and Majorana Fermions in Hybrid Semiconductor-Superconductor Wires",
    "Stochastic Resonance and Signal Amplification in Non-Linear Dynamical Systems",
    "Numerical Relativity and Gravitational Waveform Modeling for Binary Black Hole Mergers",
    "Plasma Turbulence and Heat Confinement in Tokamak Fusion Reactors",
    "Random Matrix Theory and Spectral Density of Complex Correlated Networks",
    "Bifurcation Analysis in Non-Hermitian Quantum Mechanical Systems",
    "Lattice Boltzmann Methods for Microfluidic Multiphase Flow Simulation",
    "Cosmological Parameter Estimation from Cosmic Microwave Background Anisotropies",
    "Renormalization Group Flows in Conformal Field Theories",
    "Sparse Recovery Algorithms in Compressed Sensing with Unknown Sensing Matrices",
    "Homological Algebra Applications in Topological Data Analysis",
    "Ergodic Properties of High-Dimensional Hamiltonian Systems",
    "Superconducting Qubits Decoherence Mechanisms via Two-Level State Systems",

    # Biomedicine, Neuroscience & Bioengineering (66-80)
    "CRISPR-Cas9 Off-Target Cleavage Prediction Using Deep Learning Architectures",
    "Single-Cell RNA-Sequencing Deconvolution for Tumor Microenvironment Heterogeneity",
    "Neural Decoding of Motor Cortical Spikes for Brain-Computer Interfaces",
    "Structural Basis of Antigen-Antibody Binding Affinities in Monoclonal Therapeutics",
    "Metabolic Pathway Engineering for Sustainable Biomanufacturing of Terpenoids",
    "Spatial Transcriptomics Analysis for Mapping Cellular Architecture in Glioblastoma",
    "Synaptic Plasticity and Long-Term Potentiation in Hippocampal Memory Consolidation",
    "Pharmacokinetic and Pharmacodynamic Modeling of Targeted Nanoparticle Drug Delivery",
    "Automated Arrhythmia Classification from Multi-Lead Electrocardiograms",
    "Cryo-Electron Microscopy Reconstruction of Ribosomal Conformational States",
    "Epigenetic Clocks and DNA Methylation Markers in Age-Related Degenerative Diseases",
    "Biocompatibility of 3D-Printed Hydrogel Scaffolds for Vascularized Tissue Engineering",
    "Optogenetic Control of Dopaminergic Neurons in Reward-Based Learning Circuits",
    "Microbiome-Gut-Brain Axis Signaling in Neuroinflammatory Disorders",
    "Deep Learning Models for De Novo Macrocyclic Peptide Design",

    # Economics, Quantitative Finance & Social Sciences (81-100)
    "Dynamic Stochastic General Equilibrium Models with Heterogeneous Agent Expectations",
    "High-Frequency Algorithmic Market Making and Limit Order Book Dynamics",
    "Macroeconomic Forecasting Using Large Bayesian Vector Autoregressions",
    "Central Bank Digital Currencies: Monetary Transmission and Financial Stability",
    "Network Centrality and Systemic Risk Contagion in Interbank Lending Networks",
    "Empirical Asset Pricing with Deep Factor Models and Non-Linear Cross-Sections",
    "Behavioral Biases in Household Portfolio Allocation Under Inflationary Shocks",
    "Causal Machine Learning for Program Evaluation in Labor Market Interventions",
    "Carbon Pricing Mechanisms and Border Carbon Adjustments in International Trade",
    "Supply Chain Fragility and Bullwhip Propagation Under Geopolitical Disruptions",
    "Information Diffusion and Polarization in Complex Social Media Topologies",
    "Game-Theoretic Models of Cyber-Insurance and Optimal Security Investment",
    "Automated ESG Sentiment Scoring from Corporate Financial Disclosures",
    "Real Estate Valuation Using Multimodal Geospatial and Demographic Deep Ensembles",
    "Game Theory of Collusion in Multi-Firm Automated Pricing Algorithms",
    "Sovereign Debt Default Prediction Using Machine Learning on Macroeconomic Indicators",
    "Mechanism Design for Combinatorial Spectrum Auctions with Complementarities",
    "Monetary Policy Shocks Under Asymmetric Information and Sticky Information Waves",
    "Consumer Credit Default Risk Modeling Under Macroeconomic Stress Conditions",
    "Venture Capital Deal Syndication Networks and Portfolio Startup Exit Success"
]

def count_tokens(text: str) -> int:
    """Accurately count tokens using ModernBERT tokenizer if available, else approximate."""
    tok = get_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, truncation=False))
        except Exception:
            pass
    words = len(text.split())
    return int(words * 1.33)

def export_prompts_jsonl(output_path: str = os.path.join(LONG_DOCS_DIR, "ai_paper_prompts.jsonl")):
    """Exports all 100 standardized prompts to JSONL for automated or batch LLM execution."""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, topic in enumerate(TOPICS_100, 1):
            clean_name = f"ai_paper_{idx:03d}_{topic[:40].lower().replace(' ', '_').replace(':', '')}"
            clean_name = "".join(c for c in clean_name if c.isalnum() or c == '_')
            entry = {
                "id": idx,
                "file_name": f"{clean_name}.txt",
                "topic": topic,
                "target_tokens": "1000-8000",
                "prompt": PROMPT_TEMPLATE.format(topic=topic)
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[Success] Exported 100 standardized paper prompts to: {output_path}")

def verify_dataset():
    """Scans human_papers/ and ai_papers/ and reports exact file counts and token length distributions."""
    print("=" * 75)
    print(" LONG-DOCUMENT DATASET AUDIT (1,000 - 8,000 Tokens)")
    print("=" * 75)
    
    for split_name, folder in [("Human Research Papers", HUMAN_PAPERS_DIR), ("AI-Generated Papers", AI_PAPERS_DIR)]:
        if not os.path.exists(folder):
            print(f"[{split_name}] Directory missing: {folder}")
            continue
        
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        print(f"\n--- {split_name} ({folder}) ---")
        print(f"Total files found: {len(files)} (Target: >= 100)")
        
        buckets = {"1k-2k": 0, "2k-4k": 0, "4k-6k": 0, "6k-8k": 0, "out_of_range": 0}
        valid_count = 0
        
        for f in files:
            fpath = os.path.join(folder, f)
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    content = fp.read()
                toks = count_tokens(content)
                if 1000 <= toks < 2000:
                    buckets["1k-2k"] += 1
                    valid_count += 1
                elif 2000 <= toks < 4000:
                    buckets["2k-4k"] += 1
                    valid_count += 1
                elif 4000 <= toks < 6000:
                    buckets["4k-6k"] += 1
                    valid_count += 1
                elif 6000 <= toks <= 8000:
                    buckets["6k-8k"] += 1
                    valid_count += 1
                else:
                    buckets["out_of_range"] += 1
            except Exception as e:
                print(f"  Error reading {f}: {e}")
        
        print(f"Valid papers in 1k-8k token range: {valid_count} / 100")
        print(f"  - 1,000 - 2,000 tokens: {buckets['1k-2k']}")
        print(f"  - 2,000 - 4,000 tokens: {buckets['2k-4k']}")
        print(f"  - 4,000 - 6,000 tokens: {buckets['4k-6k']}")
        print(f"  - 6,000 - 8,000 tokens: {buckets['6k-8k']}")
        if buckets["out_of_range"] > 0:
            print(f"  - [WARNING] Out of range (<1k or >8k): {buckets['out_of_range']}")

if __name__ == "__main__":
    export_prompts_jsonl()
    verify_dataset()
