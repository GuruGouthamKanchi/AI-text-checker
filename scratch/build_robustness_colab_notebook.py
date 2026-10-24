"""
scratch/build_robustness_colab_notebook.py
Builds eval/run_robustness_colab.ipynb for Google Colab GPU execution.
"""

import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_md(source_text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        })

    def add_code(source_text):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        })

    # Cell 1: Markdown Title
    add_md("""# Adversarial Robustness & Closed-Loop Retraining Pipeline (Colab GPU Version)
### Project: AI-Generated Text Detector
**Adherence to Project Rules:**
1. Zero Hallucination: All metrics originate strictly from code execution and are saved to `results/`.
2. Hardware: Runs with CUDA acceleration (T4/A100).
3. Split Strictness: Attacks built strictly within each split (train/val/test). Zero test sample contamination.
4. Retraining: Evaluates Round 0 baseline, Round 1 Cross-Entropy, Round 2 Contrastive (lambda sweep {0.1, 0.2, 0.5}).
5. Generalization: Holds out an entire attack family (`llm_prompt_heavy` & `commercial_manual`) from training.
6. Multi-seed: Sweeps 3 random seeds (42, 43, 44) reporting Mean +/- Std.
""")

    # Cell 2: Dependencies & GPU Setup
    add_code("""# CELL 1: Environment & GPU Setup
!pip install -q transformers datasets accelerate pandas pyarrow scikit-learn matplotlib

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU:      {torch.cuda.get_device_name(0)}")
    !nvidia-smi
""")

    # Cell 3: Drive Mount & Directory Auto-Detection
    add_code("""# CELL 2: Google Drive Mount & Directory Auto-Detection
import os
import sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    candidate_paths = [
        "/content/drive/MyDrive/AI_Text_Checker",
        "/content/drive/MyDrive/ai-text-detector",
        "/content/drive/MyDrive/AI-text-checker",
        "/content/drive/MyDrive/ai_text_detector",
        "/content"
    ]
    found = False
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "data", "processed", "val.parquet")):
            os.chdir(p)
            print(f"[Success] Found dataset! Active working directory: {os.getcwd()}")
            found = True
            break
            
    if not found:
        print("[!] Could not auto-detect 'data/processed/val.parquet'. Please set path below.")
except Exception as e:
    print(f"Drive mount skipped: {e}")

# Ensure required directories exist
os.makedirs("eval", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("robustness_eval/datasets", exist_ok=True)
print(f"Current Directory: {os.getcwd()}")
""")

    # Cell 4: Run Robustness Study
    with open("robustness_eval/run_robustness_study.py", "r", encoding="utf-8") as f:
        study_code = f.read()

    add_code(f"""# CELL 3: Robustness Evaluation & Closed-Loop Retraining Suite
# Generates attacks, runs baseline, fine-tunes across seeds and lambda values,
# measures held-out generalization, and exports all CSVs, JSONs, and figures.

{study_code}

if __name__ == "__main__":
    main()
""")

    # Cell 5: Display Figures & Markdown
    add_code("""# CELL 4: Inline Display of Figures & Robustness Summary
from IPython.display import Image, display, Markdown

print("=" * 80)
print(" FIGURE 1: ADVERSARIAL DEGRADATION & RECOVERY")
print("=" * 80)
if os.path.exists("results/figures/robustness_degradation.png"):
    display(Image("results/figures/robustness_degradation.png", width=750))

print("\\n" + "=" * 80)
print(" FIGURE 2: RETRAINING ROUNDS TRAJECTORY")
print("=" * 80)
if os.path.exists("results/figures/retrain_rounds.png"):
    display(Image("results/figures/retrain_rounds.png", width=750))

print("\\n" + "=" * 80)
print(" FIGURE 3: CONTRASTIVE LAMBDA SWEEP")
print("=" * 80)
if os.path.exists("results/figures/lambda_sweep.png"):
    display(Image("results/figures/lambda_sweep.png", width=750))

print("\\n" + "=" * 80)
print(" ROBUSTNESS SUMMARY REPORT")
print("=" * 80)
if os.path.exists("results/ROBUSTNESS_SUMMARY.md"):
    with open("results/ROBUSTNESS_SUMMARY.md", "r", encoding="utf-8") as f:
        display(Markdown(f.read()))
""")

    # Cell 6: Zip & Download
    add_code("""# CELL 5: Package and Download Robustness Results
import shutil
from google.colab import files

shutil.make_archive("robustness_results_archive", "zip", "results")
print("Successfully created robustness_results_archive.zip.")
try:
    files.download("robustness_results_archive.zip")
except Exception as e:
    print(f"Download prompt skipped: {e}")
""")

    out_path = "eval/run_robustness_colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated Colab notebook at: {out_path}")

if __name__ == "__main__":
    build_notebook()
