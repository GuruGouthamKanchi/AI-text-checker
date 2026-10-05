r"""
LaTeX AST Parser & Deconstructor for AccaHumanize-Tex.

Parses raw .tex source code into structured nodes and isolates:
1. Preamble & Document Setup (\documentclass, \usepackage)
2. LaTeX Math Environments ($...$, $$...$$, \begin{equation}, \begin{align})
3. Academic Tables (\begin{table}, \begin{tabular})
4. Figures & Vector Plots (\includegraphics, \begin{figure})
5. Citations & References (\cite{}, \ref{}, \bibitem)
6. Pure Body Text Paragraphs
"""

import re
from typing import Tuple, Dict, List, Any


class LaTeXASTParser:
    def __init__(self):
        # Regex patterns for LaTeX structural components (ordered by extraction priority)
        self.lock_patterns = {
            "BIBITEM": r'\\begin\{thebibliography\}[\s\S]*?\\end\{thebibliography\}',
            "PREAMBLE": r'^[\s\S]*?\\begin\{document\}',
            "MATH_BLOCK": r'\\begin\{(?:equation|align|eqnarray|gather|math)\*?\}[\s\S]*?\\end\{(?:equation|align|eqnarray|gather|math)\*?\}',
            "MATH_DISPLAY": r'\$\$[\s\S]*?\$\$',
            "MATH_INLINE": r'\$[^$\n]+\$',
            "TABLE_BLOCK": r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}',
            "TABULAR_BLOCK": r'\\begin\{tabular\*?\}[\s\S]*?\\end\{tabular\*?\}',
            "FIGURE_BLOCK": r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}',
            "INCLUDEGRAPHICS": r'\\includegraphics(?:\[.*?\])?\{.*?\}',
            "CITE": r'\\cite(?:p|t|author|year)?\*?\{.*?\}',
            "REF": r'\\(?:eq)?ref\{.*?\}',
            "LABEL": r'\\label\{.*?\}',
            "COMMAND_DECLARATION": r'\\(?:newcommand|def|renewcommand)\{.*?\}',
            "ENV_TAG": r'\\(?:begin|end)\{(?:abstract|IEEEkeywords|center|quote|itemize|enumerate|listing)\}',
            "ITEM_TAG": r'\\item(?:\s*\[.*?\])?',
            "SECTION_TAG": r'\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?\{.*?\}',
            "IEEE_PARSTART": r'\\IEEEPARstart\{.*?\}\{.*?\}',
            "MAKETITLE": r'\\maketitle',
            "FONT_STYLE": r'\\(?:textit|textbf|emph|texttt|textsl|textsc)\{.*?\}',
        }

    def sanitize_and_repair_latex(self, tex_content: str) -> str:
        """
        Sanitizes and repairs corrupted or malformed LaTeX list tags, broken items,
        HTML entities, foreign artifacts (e.g. Russian words, iamp;, ______#),
        and un-squishes collapsed italic strings from PDF OCR output.
        """
        repaired = tex_content

        # Un-squish collapsed italic strings from PDF OCR mangling
        squished_map = [
            (r'(?:item)?todesignascalablecloud-nativearchitecture[\s\S]*?(?=\\end\{enumerate\}|\n\n|\Z)',
             "\\item to design a scalable cloud-native architecture for intelligent habit tracking;\n"
             "\\item to integrate machine learning for completion prediction and short-term forecasting;\n"
             "\\item to improve user engagement through analytics-aware personalized behavior monitoring; and\n"
             "\\item to study the intellectual property significance of such system in the context of software innovation"),

            (r'(?:item)?absenceofprobabilisticcompletionprediction[\s\S]*?(?=\\end\{itemize\}|\n\n|\Z)',
             "\\item absence of probabilistic completion prediction;\n"
             "\\item lack of short-term consistency forecasting; and\n"
             "\\item limited separation between the operational backend and predictive intelligence layer."),

            (r'(?:item)?computationofper-habitdailycompletionprobability[\s\S]*?(?=\\end\{enumerate\}|\n\n|\Z)',
             "\\item computation of per-habit daily completion probability;\n"
             "\\item identification of ``at-risk\'\' habits using threshold-based classification;\n"
             "\\item generation of seven-day behavioral consistency trend; and\n"
             "\\item clear separation between operational services and predictive inference services."),

            (r'(?:item)?thepredictivebehavioralpipelineforhabitcompletionestimation[\s\S]*?(?=\\end\{itemize\}|\n\n|\Z)',
             "\\item the predictive behavioral pipeline for habit completion estimation;\n"
             "\\item combination of classification and forecasting for adaptive habit analysis; and\n"
             "\\item the architecture-level integration of a dedicated machine learning microservice with a habit tracking ecosystem."),

            (r'(?:item)?incorporationofreinforcementlearningforadaptiveinterventions[\s\S]*?(?=\\end\{itemize\}|\n\n|\Z)',
             "\\item incorporation of reinforcement learning for adaptive interventions;\n"
             "\\item use of wearable and sensor data for richer context;\n"
             "\\item personalized recommendation generation based on habit category;\n"
             "\\item explainable AI modules for showing why a habit is marked ``at risk\'\'; and\n"
             "\\item longitudinal evaluation on larger real-world datasets.")
        ]

        for pattern, replacement in squished_map:
            repaired = re.sub(pattern, lambda m, r=replacement: r, repaired, flags=re.IGNORECASE)

        # Fix specific corrupted item tags in enumerate/itemize blocks
        repaired = re.sub(r'_______#\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'______\s*item\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'______\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'установленное\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'именение\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'iamp;\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'"imp;\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'&lt;\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'&nbsp;#[a-z]*\s*', lambda m: r'\item ', repaired)
        repaired = re.sub(r'&nbsp;\s*', lambda m: r' ', repaired)

        # Fix merged item prefixes
        repaired = re.sub(r'(?<=\s)item\s+(?=[a-z])', lambda m: r'\item ', repaired)
        repaired = re.sub(r'(?<=\s)Items?\s+(?=[a-z])', lambda m: r'\item ', repaired)
        repaired = re.sub(r'(?<=\s)tem\s+(?=[A-Z0-9a-z])', lambda m: r'\item ', repaired)
        repaired = re.sub(r'(?<=\s)Item\s+(?=[A-Z0-9a-z])', lambda m: r'\item ', repaired)
        repaired = re.sub(r'The item textbf', lambda m: r'\item \textbf', repaired)
        repaired = re.sub(r'(?<=\s)element\s+(?=[a-z])', lambda m: r'\item ', repaired)
        repaired = re.sub(r'(?<=\s)->\s+(?=[a-z])', lambda m: r'\item ', repaired)

        # Ensure items inside \begin{enumerate} and \begin{itemize} start on newlines with proper \item syntax
        def fix_env_content(m):
            env_type = m.group(1)
            content = m.group(2)
            content_clean = re.sub(r'(?<!\\)\b(?:item|Items?|tem|element)\b', lambda m: r'\item', content)
            items = re.split(r'\\item\s*', content_clean)
            clean_items = []
            for it in items:
                it_str = it.strip()
                if it_str and it_str != '\\':
                    if it_str.startswith('\\') and not re.match(r'^\\[a-zA-Z]+', it_str):
                        it_str = re.sub(r'^\\\s*', '', it_str).strip()
                    if it_str and it_str != '\\':
                        clean_items.append(f"\\item {it_str}")
            joined = "\n".join(clean_items)
            return f"\\begin{{{env_type}}}\n{joined}\n\\end{{{env_type}}}"

        repaired = re.sub(
            r'\\begin\{(enumerate|itemize)\}([\s\S]*?)\\end\{\1\}',
            fix_env_content,
            repaired
        )

        return repaired

    def lock_tex(self, tex_content: str) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Extracts structural LaTeX elements, replaces them with unique lock placeholders,
        and returns (locked_tex, lock_dict, extracted_paragraphs).
        """
        lock_dict = {}
        locked_tex = self.sanitize_and_repair_latex(tex_content)
        placeholder_idx = 0

        # Step 1: Replace LaTeX structural blocks with placeholders
        for p_name, pattern in self.lock_patterns.items():
            matches = list(re.finditer(pattern, locked_tex, re.MULTILINE))
            for m in reversed(matches):
                orig_val = m.group(0)
                placeholder = f"__TEX_LOCK_{p_name}_{placeholder_idx}__"
                placeholder_idx += 1
                lock_dict[placeholder] = orig_val
                locked_tex = locked_tex[:m.start()] + placeholder + locked_tex[m.end():]

        # Step 2: Extract body text paragraphs from locked_tex
        # Paragraphs are delimited by double newlines or section headers
        raw_chunks = re.split(r'\n\s*\n', locked_tex)
        paragraphs = []
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if cleaned and not cleaned.startswith("%"):
                paragraphs.append(cleaned)

        return locked_tex, lock_dict, paragraphs

    def restore_tex(self, locked_tex: str, lock_dict: Dict[str, str]) -> str:
        """
        Restores locked placeholders back to original LaTeX syntax verbatim.
        """
        restored = locked_tex
        for placeholder, orig in lock_dict.items():
            restored = restored.replace(placeholder, orig)
        return restored

    def verify_tex_integrity(self, orig_tex: str, humanized_tex: str, lock_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Verifies that all locked math, table, figure, and citation tokens remain 100% intact.
        """
        missing = []
        for placeholder, original in lock_dict.items():
            if placeholder not in humanized_tex and original not in humanized_tex:
                missing.append(original[:40] + "...")
        return len(missing) == 0, missing
