"""
LaTeX AST Re-Assembler & PDF Compiler Wrapper for AccaHumanize-Tex.

Re-assembles optimized LaTeX AST nodes and provides subprocess compilation to PDF
using pdflatex or xelatex when available on host OS.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from typing import Tuple, Optional, Dict


class LaTeXCompiler:
    def __init__(self):
        # Locate available LaTeX compiler binary on system PATH
        self.compiler_bin = (
            shutil.which("pdflatex") or
            shutil.which("xelatex") or
            shutil.which("lualatex")
        )

    def is_compiler_available(self) -> bool:
        """Returns True if pdflatex or xelatex binary is installed on system PATH."""
        return self.compiler_bin is not None

    def compile_tex_to_pdf(self, tex_content: str, filename: str = "document.tex") -> Tuple[Optional[bytes], str, bool]:
        """
        Compiles .tex source string to PDF bytes using pdflatex.
        Returns (pdf_bytes, compile_log_stdout, is_success).
        """
        if not self.is_compiler_available():
            msg = "[LaTeX Compiler] pdflatex/xelatex binary not found on PATH. Output will be generated as refined .tex source file."
            return None, msg, False

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file_path = os.path.join(tmpdir, filename)
            with open(tex_file_path, "w", encoding="utf-8") as f:
                f.write(tex_content)

            try:
                cmd = [
                    self.compiler_bin,
                    "-interaction=nonstopmode",
                    "-output-directory", tmpdir,
                    tex_file_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)

                pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
                pdf_path = os.path.join(tmpdir, pdf_filename)

                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pf:
                        pdf_bytes = pf.read()
                    return pdf_bytes, result.stdout, True
                else:
                    return None, f"LaTeX compilation failed:\n{result.stdout[-1000:]}", False

            except Exception as e:
                return None, f"LaTeX compilation subprocess error: {str(e)}", False
