"""
scripts/generate_claude_digest.py
Automated Repository Bundler for Claude.
Generates a clean, monolithic, token-optimized textual snapshot of the entire repository
(every source file, documentation, configuration, and result summary) for direct consumption by Claude.
"""

import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(WORKSPACE_ROOT, "CLAUDE_DIGEST.txt")

INCLUDED_EXTENSIONS = {
    ".py", ".md", ".tex", ".json", ".csv", ".bat", ".sh", ".txt", ".ts", ".tsx", ".js", ".jsx", ".css", ".html"
}

EXCLUDED_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", "node_modules", ".next",
    "data", "models", ".gemini", "dist", "build", ".system_generated"
}

def should_include_file(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    if basename in {"package-lock.json", "yarn.lock", "CLAUDE_DIGEST.txt", "cache_val_preds.parquet", "cache_test_preds.parquet"}:
        return False
    _, ext = os.path.splitext(basename)
    return ext.lower() in INCLUDED_EXTENSIONS

def generate_digest():
    print(f"Generating monolithic repository digest for Claude from: {WORKSPACE_ROOT}...")
    digest_lines = []

    digest_lines.append("=" * 80)
    digest_lines.append(" AI-TEXT-CHECKER: COMPLETE REPOSITORY DIGEST FOR CLAUDE")
    digest_lines.append("=" * 80)
    digest_lines.append(f"Repository: https://github.com/GuruGouthamKanchi/AI-text-checker\n\n")

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

                    digest_lines.append("\n" + "=" * 80)
                    digest_lines.append(f"FILE: {rel_path}")
                    digest_lines.append("=" * 80)
                    digest_lines.append(content)

                    file_count += 1
                    total_bytes += len(content.encode("utf-8"))
                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(digest_lines))

    print(f"\n[Success] Bundled {file_count} files ({total_bytes / 1024:.1f} KB) into: {OUTPUT_FILE}")
    print("You can upload this single file directly to any Claude Chat or Claude Project!")

if __name__ == "__main__":
    generate_digest()
