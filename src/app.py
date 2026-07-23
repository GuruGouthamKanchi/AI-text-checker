import os
import re
import uuid
import shutil
import tempfile
import sys
from contextlib import asynccontextmanager
import numpy as np

# Ensure root folder is in the python path for importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.highlighter import (
    segment_sentences,
    run_predictions,
    smooth_predictions,
    calculate_explainability_metrics,
    analyze_llm_signatures,
    extract_citations,
    verify_citation
)

from src.document_parser import (
    extract_text,
    DocumentParsingError,
    PasswordProtectedError,
    MissingDependencyError
)

# Global variables for pre-loaded model and tokenizer
model_dir = "models/roberta-sentence-academic-v2"
model = None
tokenizer = None
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    print("[FastAPI Startup] Checking local model weights...")
    model_exists = (
        os.path.exists(model_dir) and 
        os.path.exists(os.path.join(model_dir, "config.json"))
    )
    
    if model_exists:
        try:
            print(f"[FastAPI Startup] Loading RoBERTa-v2 model from '{model_dir}' in memory...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()
            model.to(device)
            print(f"[FastAPI Startup] Model loaded successfully on {device}!")
        except Exception as e:
            print(f"[FastAPI Startup] Error preloading model: {e}. Falling back to simulation.")
    else:
        print(f"[FastAPI Startup] Model weights not found at '{model_dir}'. App will run in SIMULATION MODE.")
        
    yield
    print("[FastAPI Shutdown] Unloading model...")

app = FastAPI(
    title="VeriPaper AI Text Detector API",
    description="Local verification service for identifying AI-generated writing in academic research papers.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for next.js dev server on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

def run_analysis_pipeline(doc_text: str, filename: str, page_count: int):
    # Clean up whitespace
    doc_text = doc_text.strip()
    if not doc_text:
        raise HTTPException(
            status_code=400,
            detail="The document contains no readable text."
        )

    # Segment document into individual sentences
    sentences = segment_sentences(doc_text)
    total_sentences = len(sentences)
    if total_sentences == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid sentences could be extracted from the document."
        )

    # Calculate word count
    word_count = len([w for w in doc_text.split() if w.strip()])

    # Run classification model
    try:
        predictions, is_simulated = run_predictions(
            sentences=sentences,
            model_dir=model_dir,
            preloaded_model=model,
            preloaded_tokenizer=tokenizer
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    # Apply 3-sentence rolling average smoothing
    predictions = smooth_predictions(predictions)

    # Calculate explainability metrics
    lexical_diversity, structural_burstiness = calculate_explainability_metrics(sentences)

    # Run advanced detection features (LLM signature analysis & Citation audit)
    llm_signatures = analyze_llm_signatures(doc_text)
    
    from src.citation_parser import audit_citations_integrity
    audit = audit_citations_integrity(doc_text)
    
    verified_citations = []
    for ref in audit["references"]:
        verified_citations.append({
            "reference": ref["raw_text"],
            "status": ref["status"],
            "details": ref["details"],
            "doi": ref["doi"]
        })
    citation_health = audit["health_score"]

    # Calculate metric labels & colors matching production thresholds
    # Lexical Diversity
    if lexical_diversity >= 68.0:
        lex_label = "Rich Vocabulary (Human-like)"
    elif lexical_diversity >= 58.0:
        lex_label = "Standard Vocabulary"
    else:
        lex_label = "Repetitive Vocabulary (AI-like)"
        
    # Structural Burstiness
    if structural_burstiness >= 45.0:
        burst_label = "Dynamic Structure (Human-like)"
    elif structural_burstiness >= 30.0:
        burst_label = "Moderate Variation"
    else:
        burst_label = "Uniform Structure (AI-like)"

    # Calculate confidence tiers and aggregate counts
    processed_sentences = []
    flagged_high = 0
    flagged_mid = 0
    unflagged = 0
    
    paragraph_index = 0
    raw_paragraphs = doc_text.split("\n\n")
    sentence_idx = 0
    
    for p_idx, p_text in enumerate(raw_paragraphs):
        p_sentences = segment_sentences(p_text)
        for p_sent in p_sentences:
            if sentence_idx >= len(predictions):
                break
                
            pred = predictions[sentence_idx]
            score = pred["score"]
            
            if score >= 0.80:
                tier = "high"
                flagged_high += 1
            elif score >= 0.60:
                tier = "medium"
                flagged_mid += 1
            else:
                tier = "unflagged"
                unflagged += 1
                
            processed_sentences.append({
                "text": pred["sentence"],
                "paragraph_index": p_idx,
                "ai_probability": score,
                "confidence_tier": tier
            })
            sentence_idx += 1

    while sentence_idx < len(predictions):
        pred = predictions[sentence_idx]
        score = pred["score"]
        if score >= 0.80:
            tier = "high"
            flagged_high += 1
        elif score >= 0.60:
            tier = "medium"
            flagged_mid += 1
        else:
            tier = "unflagged"
            unflagged += 1
            
        processed_sentences.append({
            "text": pred["sentence"],
            "paragraph_index": paragraph_index,
            "ai_probability": score,
            "confidence_tier": tier
        })
        sentence_idx += 1

    flagged_total = flagged_high + flagged_mid
    flagged_ratio = (flagged_total / total_sentences * 100) if total_sentences > 0 else 0.0
    overall_percentage = min(100.0, flagged_ratio * 2.5) if flagged_total > 0 else 0.0

    return {
        "metadata": {
            "filename": filename,
            "page_count": page_count,
            "word_count": word_count
        },
        "overall_ai_percentage": overall_percentage,
        "sentences": processed_sentences,
        "explainability": {
            "lexical_diversity": {
                "score": lexical_diversity,
                "label": lex_label
            },
            "structural_burstiness": {
                "score": structural_burstiness,
                "label": burst_label
            }
        },
        "llm_signatures": llm_signatures,
        "citation_audit": {
            "health_score": citation_health,
            "detected_style": audit["style"],
            "style_confidence": audit["confidence"],
            "in_text_marker_count": audit["in_text_marker_count"],
            "references": verified_citations,
            "unmatched_citations": audit["unmatched_citations"],
            "unreferenced_entries": audit["unreferenced_entries"]
        },
        "summary": {
            "high_confidence_count": flagged_high,
            "medium_confidence_count": flagged_mid,
            "unflagged_count": unflagged
        },
        "is_simulated": is_simulated
    }

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Ingests an academic research paper PDF, segments it into sentences,
    runs the local RoBERTa v2 classifier, applies rolling-average smoothing,
    calculates vocabulary/structure explainability, and returns details.
    """
    # 1. Validate file extension (case-insensitive)
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only academic manuscripts in PDF (.pdf), Word (.docx), or legacy Word (.doc) formats are supported."
        )

    # 2. Validate file size (limit 25MB)
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning of stream
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify file size: {str(e)}"
        )
        
    if size > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size limit exceeded. Uploaded manuscripts must be under 25MB."
        )

    # 3. Save uploaded file to a temporary file
    temp_dir = tempfile.gettempdir()
    safe_filename = os.path.basename(filename)
    temp_path = os.path.join(temp_dir, f"upload_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write uploaded file to temp buffer: {str(e)}"
        )

    # 4. Extract document text using the unified parser
    doc_text = ""
    page_count = 0
    try:
        doc_text, page_count = extract_text(temp_path)
    except PasswordProtectedError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except MissingDependencyError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except DocumentParsingError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during document parsing: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return run_analysis_pipeline(doc_text, file.filename, page_count)

@app.get("/sample")
async def analyze_sample(type: str):
    """
    Analyzes a preloaded sample PDF paper (human or AI).
    """
    if type == "human":
        file_path = "d:\\AI_Text_Checker\\EJ1172284.pdf"
        filename = "EJ1172284.pdf"
    elif type == "ai":
        file_path = "d:\\AI_Text_Checker\\LLM_Architectures_and_Training_Survey.pdf"
        filename = "LLM_Architectures_and_Training_Survey.pdf"
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid sample type. Select 'human' or 'ai'."
        )
        
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Sample file not found at path: {file_path}"
        )
        
    # Extract PDF text using pypdf
    doc_text = ""
    page_count = 0
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        page_count = len(reader.pages)
        
        text_pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
        doc_text = "\n".join(text_pages)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse sample PDF: {str(e)}"
        )

    return run_analysis_pipeline(doc_text, filename, page_count)

@app.post("/report")
async def download_report(
    file: UploadFile = File(None),
    sample_type: str = None
):
    """
    Ingests an uploaded document or sample paper, runs the forensics pipeline,
    and returns a Turnitin-style highlighted PDF report.
    """
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTasks
    
    if sample_type:
        if sample_type == "human":
            source_path = "d:\\AI_Text_Checker\\EJ1172284.pdf"
            filename = "EJ1172284.pdf"
        elif sample_type == "ai":
            source_path = "d:\\AI_Text_Checker\\LLM_Architectures_and_Training_Survey.pdf"
            filename = "LLM_Architectures_and_Training_Survey.pdf"
        else:
            raise HTTPException(status_code=400, detail="Invalid sample type.")
            
        if not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail="Sample file not found.")
            
        size = os.path.getsize(source_path)
        is_temp_file = False
        temp_path = source_path
    else:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded or sample selected.")
            
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".docx", ".doc"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Only PDF, DOCX, and DOC documents are supported."
            )
            
        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to verify file size: {str(e)}")
            
        if size > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size limit exceeded. Limit is 25MB.")
            
        temp_dir = tempfile.gettempdir()
        safe_filename = os.path.basename(filename)
        temp_path = os.path.join(temp_dir, f"report_upload_{uuid.uuid4().hex}_{safe_filename}")
        is_temp_file = True
        
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write uploaded file to temp buffer: {str(e)}")

    try:
        from src.document_parser import extract_text
        from src.report_generator import generate_pdf_report
        
        doc_text, page_count = extract_text(temp_path)
        doc_text_clean = doc_text.strip()
        if not doc_text_clean:
            raise HTTPException(status_code=400, detail="The document contains no readable text.")
            
        sentences = segment_sentences(doc_text_clean)
        total_sentences = len(sentences)
        if total_sentences == 0:
            raise HTTPException(status_code=400, detail="No valid sentences could be extracted.")
            
        word_count = len([w for w in doc_text_clean.split() if w.strip()])
        char_count = len(doc_text_clean)
        
        # Run predictions
        predictions, is_simulated = run_predictions(
            sentences=sentences,
            model_dir=model_dir,
            preloaded_model=model,
            preloaded_tokenizer=tokenizer
        )
        
        # Apply smoothing
        smoothed = smooth_predictions(predictions)
        
        # Compute breakdown metrics
        flagged_high = 0
        flagged_mid = 0
        unflagged = 0
        
        for pred in smoothed:
            score = pred["score"]
            if score >= 0.85:
                pred["confidence_tier"] = "high"
                flagged_high += 1
            elif score >= 0.50:
                pred["confidence_tier"] = "medium"
                flagged_mid += 1
            else:
                pred["confidence_tier"] = "none"
                unflagged += 1
                
        flagged_total = flagged_high + flagged_mid
        flagged_ratio = (flagged_total / total_sentences * 100) if total_sentences > 0 else 0.0
        overall_percentage = min(100.0, flagged_ratio * 2.5) if flagged_total > 0 else 0.0
        
        metrics = {
            "high_confidence_count": flagged_high,
            "medium_confidence_count": flagged_mid,
            "unflagged_count": unflagged,
            "file_size_bytes": size
        }
        
        # Generate the PDF report
        report_pdf_path = generate_pdf_report(
            source_file_path=temp_path,
            predictions=smoothed,
            overall_percentage=overall_percentage,
            metrics=metrics,
            total_words=word_count,
            char_count=char_count
        )
        
        bg_tasks = BackgroundTasks()
        bg_tasks.add_task(os.remove, report_pdf_path)
        
        safe_title = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
        if safe_title.endswith(".pdf"):
            download_name = safe_title.replace(".pdf", "_VeriPaper_Report.pdf")
        elif safe_title.endswith(".docx"):
            download_name = safe_title.replace(".docx", "_VeriPaper_Report.pdf")
        elif safe_title.endswith(".doc"):
            download_name = safe_title.replace(".doc", "_VeriPaper_Report.pdf")
        else:
            download_name = f"{safe_title}_VeriPaper_Report.pdf"
            
        return FileResponse(
            report_pdf_path,
            media_type="application/pdf",
            filename=download_name,
            background=bg_tasks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate forensic report: {str(e)}"
        )
    finally:
        if is_temp_file and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

