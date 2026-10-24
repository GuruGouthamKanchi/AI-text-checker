# Non-Native English Writing Evaluation Directory (`eval/non_native/`)

This directory is dedicated to evaluating false positive rates (FPR) on non-native English writing (e.g., TOEFL, IELTS, ICLE, or international student academic essays).

### Project Rule Compliance (Rule 2)
- **Zero Simulation / Fabrication:** Per Project Rule 2, AI models or agents must **never** synthesize, fabricate, or simulate non-native writing samples.
- **User-Supplied Data:** Users can place `.parquet`, `.csv`, or `.txt` files containing verified human non-native writing into this directory.

### Ingestion Schema
When providing data:
1. **Parquet / CSV Format:** Must contain a column named `text` (and optionally `source` or `proficiency_level`).
2. **Text Files (`.txt`):** Individual essay files placed directly in this directory will be automatically ingested as human writing (`label = 0`).

The loader script is located at `eval/non_native/loader.py` and is automatically integrated into the evaluation suite.
