# AI-Generated Text Detector - Phase 1: Data Preprocessing

This repository implements **Phase 1: Data Preprocessing** for a sentence-level AI-generated text detector. The goal of this phase is to load multiple raw datasets, normalize and clean the text, perform near-duplicate detection and removal, and partition the data into train, validation, and test splits without topic leakage.

## Project Structure

```text
ai-text-detector/
├── data/
│   ├── raw/               # Raw downloaded files (gitignored)
│   └── processed/         # Cleaned parquet outputs (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loaders.py    # Ingestion functions for DAIGT and HC3
│   ├── clean_text.py      # Unicode normalization, HTML/markdown removal, sentence segmenter
│   ├── dedupe.py          # Exact duplicate and MinHash LSH fuzzy deduplication
│   ├── split.py           # Leak-free topic-aware splitting
│   └── prepare_dataset.py # Orchestrator that puts it all together
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### 1. Prerequisites & Installation

Verify that you are using Python 3.12+ (tested on Python 3.12.4 on Windows).

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

This project uses two datasets:
1. **DAIGT V2 Train Dataset**: 
   - Download the dataset manually from Kaggle (e.g. from the AI-generated essay detection challenges).
   - Place the CSV file in `data/raw/` and name it `daigt_v2_train.csv`.
   - The loader is designed defensively to handle variations in the columns `text`, `label` / `generated`, `prompt_name`, and `source`.
2. **HC3 (Hello-SimpleAI/HC3)**:
   - This dataset is automatically downloaded from Hugging Face via the `datasets` library. No manual download is required.

### 3. Running the Pipeline

Execute the main orchestration script:
```bash
python src/prepare_dataset.py
```

### 4. Pipeline Steps
When you run the script, it will execute the following steps:
1. **Load Raw Data**: Ingests both DAIGT (from `data/raw/daigt_v2_train.csv`) and HC3 (streamed/downloaded from Hugging Face).
2. **Combine**: Standardizes the schemas to `text`, `label` (0 = human, 1 = AI), `source`, and `topic_id`.
3. **Clean Text**: Applies text normalization, HTML/markdown removal, smart quote mapping, and sentence tokenization prep.
4. **Deduplication**: Filters out exact matches followed by fuzzy near-duplicates using **MinHash LSH** (Jaccard similarity threshold of `0.9`).
5. **Topic-Aware Split**: Group-splits unique `topic_id`s into a `80% / 10% / 10%` Train/Val/Test distribution, guaranteeing zero prompt/topic leakage across partitions.
6. **Save**: Exports the split datasets as `train.parquet`, `val.parquet`, and `test.parquet` in `data/processed/`.

## Key Preprocessing Details

### Sentence Segmentation
Even though Phase 1 saves document-level datasets, we include a robust sentence splitter `split_into_sentences` in `src/clean_text.py` leveraging NLTK's `sent_tokenize`. This enables Phase 5 to break essays into sentences, run sentence-level predictions, and highlight specific AI sentences (like Turnitin).

### Near-Duplicate Detection (MinHash LSH)
To avoid training models on near-identical templates or repeated answers, we run fuzzy deduplication. We build word 3-gram MinHash signatures with 128 permutations, query an LSH index to group similar texts, and discard near-duplicates (Jaccard distance < 0.1).

### Topic Leakage Prevention
Models can easily overfit to specific essay prompts (e.g., memorizing keywords about "Car-free cities"). By splitting unique `topic_id` prompts instead of individual rows, we guarantee that the validation and test sets evaluate generalization to unseen topics.

## Supported File Formats

This app supports analyzing academic documents in the following formats:
1. **PDF (`.pdf`)**: Native text layers are extracted directly using `pypdf`.
2. **Word (`.docx`)**: Body paragraphs and table data are extracted using `python-docx`.
3. **Legacy Word (`.doc`)**: Legacy binary documents are converted to `.docx` in the background using LibreOffice.

### System Prerequisites for Legacy DOC Support
To upload `.doc` files, **LibreOffice** must be installed on the host system:
* **Windows**: Download and install LibreOffice from [libreoffice.org](https://www.libreoffice.org/download/download/). The app will auto-detect standard installations in `C:\Program Files\LibreOffice`.
* **macOS / Linux**: Install LibreOffice via your package manager (e.g., `brew install libreoffice` or `sudo apt-get install libreoffice`). Make sure the `soffice` binary is added to your environment `PATH`.

*Fallback:* If you do not wish to install LibreOffice, you can manually open the legacy `.doc` file in Microsoft Word or Google Docs, save/export it as a `.docx` file, and upload the `.docx` file directly.

## Running the Project

### First-Time Setup
Before running the application for the first time, run the installation script once. This checks system prerequisites, configures the virtual environment, installs dependencies, and prepares the workspace.

- **Windows**: Double-click `install.bat`
- **Mac/Linux**: Make executable and run: `chmod +x install.sh && ./install.sh`

Once the installation completes, you can start the application using the scripts below.

Simple double-clickable startup and shutdown scripts are provided for both Windows (batch) and Mac/Linux (bash) environments. The backend FastAPI service runs on port `8000`, and the frontend Next.js dev server runs on port `3000`.

### Windows Script Instructions
- **Start All Services**: Double-click `start_all.bat` (starts backend first, then frontend).
- **Stop All Services**: Double-click `stop_all.bat` (terminates both backend and frontend processes, cleaning up ports 8000 and 3000).
- **Independent Servers**:
  - Start Backend: `start_backend.bat`
  - Stop Backend: `stop_backend.bat`
  - Start Frontend: `start_frontend.bat`
  - Stop Frontend: `stop_frontend.bat`

### Mac/Linux Script Instructions
Make sure the scripts are executable (the `start_all.sh` script does this automatically).
- **Start All Services**: `./start_all.sh`
- **Stop All Services**: `./stop_all.sh`
- **Independent Servers**:
  - Start Backend: `./start_backend.sh`
  - Stop Backend: `./stop_backend.sh`
  - Start Frontend: `./start_frontend.sh`
  - Stop Frontend: `./stop_frontend.sh`

### Expected URLs
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)


