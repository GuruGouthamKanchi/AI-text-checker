"""
robustness_eval/manual_inputs/ingest_manual_attacks.py
Ingestion script for manual inputs from commercial tools (QuillBot, Undetectable.ai, etc.).
In strict compliance with project guidelines:
- Commercial tools are NEVER scraped or automated.
- This script ingests pairs pasted or provided manually by the user.
- Outputs are saved with explicit tool provenance and labeled separately in all benchmark reports.
"""

import os
import json
import argparse
from typing import List, Dict, Any

MANUAL_INPUTS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMERCIAL_JSONL_PATH = os.path.join(MANUAL_INPUTS_DIR, "commercial_attacks.jsonl")

def ingest_manual_entry(original_text: str, rewritten_text: str, tool_name: str, split: str = "test", notes: str = "") -> Dict[str, Any]:
    """Appends a single manual commercial attack pair to the commercial dataset."""
    entry = {
        "original_text": original_text.strip(),
        "rewritten_text": rewritten_text.strip(),
        "tool_name": tool_name.strip(),
        "split": split.strip().lower(),
        "source_label": "ai",
        "attack_family": "commercial_manual",
        "notes": notes.strip()
    }
    
    with open(COMMERCIAL_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    print(f"[Ingest] Successfully logged manual commercial pair for tool '{tool_name}' ({split} split).")
    return entry

def load_commercial_attacks(split: str = "test") -> List[Dict[str, Any]]:
    """Loads all manual commercial attacks matching the requested split."""
    if not os.path.exists(COMMERCIAL_JSONL_PATH):
        return []
    
    entries = []
    with open(COMMERCIAL_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("split", "test") == split.lower():
                        entries.append(data)
                except Exception:
                    continue
    return entries

def batch_ingest_from_json(json_file_path: str):
    """Batch imports manual entries from a user-supplied JSON or JSONL file."""
    if not os.path.exists(json_file_path):
        print(f"[Error] File not found: {json_file_path}")
        return
    
    count = 0
    with open(json_file_path, "r", encoding="utf-8") as f:
        # Check if json or jsonl
        content = f.read().strip()
        if content.startswith("["):
            items = json.loads(content)
        else:
            items = [json.loads(line) for line in content.split("\n") if line.strip()]
            
    for item in items:
        orig = item.get("original_text", "")
        rewr = item.get("rewritten_text", "")
        tool = item.get("tool_name", item.get("tool", "QuillBot"))
        split = item.get("split", "test")
        notes = item.get("notes", "")
        if orig and rewr:
            ingest_manual_entry(orig, rewr, tool, split, notes)
            count += 1
            
    print(f"[Batch Ingest] Imported {count} commercial attack pairs into {COMMERCIAL_JSONL_PATH}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest manual commercial tool attack samples.")
    parser.add_argument("--batch-file", type=str, default=None, help="Path to JSON/JSONL file with manual samples.")
    parser.add_argument("--tool", type=str, default="QuillBot", help="Commercial tool name (e.g. QuillBot, Undetectable.ai).")
    parser.add_argument("--orig", type=str, default=None, help="Original AI-generated text string.")
    parser.add_argument("--rewritten", type=str, default=None, help="Commercial tool rewritten text string.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, val, test).")
    args = parser.parse_args()
    
    if args.batch_file:
        batch_ingest_from_json(args.batch_file)
    elif args.orig and args.rewritten:
        ingest_manual_entry(args.orig, args.rewritten, args.tool, args.split)
    else:
        print("Usage: python ingest_manual_attacks.py --orig '...' --rewritten '...' --tool 'QuillBot' --split test")
        print("   or: python ingest_manual_attacks.py --batch-file samples.json")
