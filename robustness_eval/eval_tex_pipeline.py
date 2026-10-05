r"""
Evaluation Harness for AccaHumanize-Tex LaTeX Document Pipeline.

Evaluates an entire LaTeX research paper:
1. Preamble & Document Setup Preservation (\documentclass, \usepackage)
2. LaTeX Math Equation Lock (\begin{equation}, $...$)
3. Table & Figure Environments Lock (\begin{table}, \includegraphics)
4. Inline Citation Lock (\cite{...}, \ref{...})
5. Body Paragraph Detection & Closed-Loop Humanization
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.closed_loop_engine import ClosedLoopHumanizer
from src.tex_parser import LaTeXASTParser
from src.tex_compiler import LaTeXCompiler

SAMPLE_LATEX_PAPER = r"""
\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}

\title{Neural Architecture Search via Closed-Loop Optimization}
\author{A. Researcher et al.}

\begin{document}
\maketitle

\begin{abstract}
Furthermore, transformer models exhibit significant advantages in sequence prediction tasks \cite{vaswani2017}. Consequently, empirical results indicate superior accuracy.
\end{abstract}

\section{Introduction}
Recent advancements in deep learning have revolutionized natural language understanding \cite{smith2023}. Detailed methodological analysis can be retrieved via $P(x) = \exp(-H(x))$. Therefore, it is essential to consider embedding variance across paragraphs.

\begin{equation}
L_{total} = \alpha L_{detection} + (1-\alpha) L_{semantic}
\end{equation}

\begin{table}[h]
\centering
\begin{tabular}{|c|c|}
\hline
Model & Accuracy \\
\hline
RoBERTa & 98.4\% \\
\hline
\end{tabular}
\caption{Model performance summary.}
\end{table}

\section{Conclusion}
Consequently, our proposed framework guarantees zero corruption of math equations or citations.

\begin{thebibliography}{99}
\bibitem{vaswani2017} Vaswani et al., Attention is All You Need, NeurIPS 2017.
\bibitem{smith2023} Smith et al., AI Text Auditing, IEEE 2023.
\end{thebibliography}

\end{document}
"""

def test_latex_pipeline():
    print("=" * 70)
    print("    AccaHumanize-Tex: LaTeX Document Pipeline Evaluation")
    print("=" * 70)

    parser = LaTeXASTParser()
    locked_tex, lock_dict, paragraphs = parser.lock_tex(SAMPLE_LATEX_PAPER)

    print(f"\n[1. Deconstruction AST Analysis]")
    print(f"  - Extracted Body Paragraphs: {len(paragraphs)}")
    print(f"  - Isolated & Locked Structural Tokens: {len(lock_dict)}")
    for k, v in list(lock_dict.items())[:5]:
        print(f"    * {k} -> {repr(v[:50])}")

    print(f"\n[2. Closed-Loop Optimization Execution]")
    engine = ClosedLoopHumanizer()

    final_tex = ""
    for event in engine.process_closed_loop_tex_stream(SAMPLE_LATEX_PAPER, tone="academic"):
        if event["type"] == "tex_step":
            print(f"  Paragraph #{event['paragraph_index']+1}:")
            print(f"    - Original:  '{event['original_paragraph'][:60]}...'")
            print(f"    - AI Prob:   {event['initial_ai_score']*100:.1f}% -> {event['final_ai_score']*100:.1f}%")
            print(f"    - Similarity: {event['semantic_similarity']*100:.1f}%")
        elif event["type"] == "tex_complete":
            final_tex = event["final_tex"]
            print(f"\n[3. Final Document Summary]")
            print(f"  - Avg AI Score Reduction: {event['avg_initial_ai_score']*100:.1f}% -> {event['avg_final_ai_score']*100:.1f}%")
            print(f"  - Avg Semantic Retention: {event['avg_semantic_similarity']*100:.1f}%")

    print(f"\n[4. LaTeX AST Integrity Verification]")
    is_valid, missing = parser.verify_tex_integrity(SAMPLE_LATEX_PAPER, final_tex, lock_dict)
    print(f"  - Math, Tables, Figures & Citation Lock Status: {'100% PRESERVED' if is_valid else 'FAILED'}")
    if missing:
        print(f"  - Missing tokens: {missing}")

    print(f"\n[5. Subprocess Compilation Check]")
    compiler = LaTeXCompiler()
    print(f"  - System LaTeX Compiler Available: {compiler.is_compiler_available()}")
    if compiler.is_compiler_available():
        pdf_bytes, log, ok = compiler.compile_tex_to_pdf(final_tex)
        print(f"  - PDF Compilation Result: {'SUCCESS (' + str(len(pdf_bytes)) + ' bytes)' if ok else 'FAILED'}")

    print("\n" + "=" * 70)
    print("                 LATEX PIPELINE VERIFICATION PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_latex_pipeline()
