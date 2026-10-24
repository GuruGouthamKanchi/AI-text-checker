"""
scripts/generate_claude_digest.py
Automated Repository Bundler for Claude (Markdown .md format).
Generates a token-optimized, syntax-highlighted Markdown digest of the entire repository
(every source file, documentation, configuration, and result summary) for direct consumption by Claude.
"""

import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(WORKSPACE_ROOT, "CLAUDE_DIGEST.md")

INCLUDED_EXTENSIONS = {
    ".py", ".md", ".tex", ".json", ".csv", ".bat", ".sh", ".txt", ".ts", ".tsx", ".js", ".jsx", ".css", ".html"
}

EXCLUDED_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", "node_modules", ".next",
    "data", "models", ".gemini", "dist", "build", ".system_generated"
}

EXT_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".tex": "latex",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".bat": "batch",
    ".txt": "text",
}

def should_include_file(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    if basename in {
        "package-lock.json", "yarn.lock", "CLAUDE_DIGEST.txt", "CLAUDE_DIGEST.md",
        "cache_val_preds.parquet", "cache_test_preds.parquet"
    }:
        return False
    _, ext = os.path.splitext(basename)
    return ext.lower() in INCLUDED_EXTENSIONS

def generate_digest():
    print(f"Generating token-optimized Markdown repository digest for Claude from: {WORKSPACE_ROOT}...")
    digest_lines = []

    digest_lines.append("# AI-Text-Checker: Monolithic Repository Digest for Claude\n")
    digest_lines.append("> **Repository:** https://github.com/GuruGouthamKanchi/AI-text-checker")
    digest_lines.append("> **Architecture Guide:** See `CLAUDE.md` for complete architecture diagrams and system contracts.\n")
    digest_lines.append("---\n")

    file_count = 0
    total_bytes = 0

    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        # Exclude ignored directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in sorted(files):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, WORKSPACE_ROOT).replace("\\", "/")

            if should_include_file(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Truncate very large CSV/JSON lines to keep tokens lean
                    lines = content.splitlines()
                    if len(lines) > 500 and (file.endswith(".csv") or file.endswith(".json")):
                        content = "\n".join(lines[:500]) + f"\n... [Truncated {len(lines)-500} additional lines for brevity]"

                    _, ext = os.path.splitext(file)
                    lang = EXT_TO_LANG.get(ext.lower(), "text")

                    digest_lines.append(f"## File: `{rel_path}`\n")
                    digest_lines.append(f"```{lang}")
                    digest_lines.append(content)
                    digest_lines.append("```\n")

                    file_count += 1
                    total_bytes += len(content.encode("utf-8"))
                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(digest_lines))

    print(f"[Success] Bundled {file_count} files ({total_bytes / 1024:.1f} KB) into Markdown format: {OUTPUT_FILE}")
    print("Markdown formatting maximizes BPE compression, saving tokens compared to plain text!")

if __name__ == "__main__":
    generate_digest()
