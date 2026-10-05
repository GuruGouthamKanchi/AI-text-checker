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

# Global variables for Hybrid Ensemble Engine (ModernBERT 8k + RoBERTa 512)
model_dir = os.environ.get("MODEL_DIR", "models/modernbert-academic")
roberta_dir = os.environ.get("ROBERTA_MODEL_DIR", "models/roberta-sentence-academic-v2")

model_modernbert = None
tokenizer_modernbert = None
model_roberta = None
tokenizer_roberta = None
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_modernbert, tokenizer_modernbert, model_roberta, tokenizer_roberta, device, model_dir, roberta_dir
    print(f"[FastAPI Startup] Initializing Hybrid Ensemble Engine (ModernBERT 8k + RoBERTa 512)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load ModernBERT (8k macro context)
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            print(f"[FastAPI Startup] Loading ModernBERT (8k) from '{model_dir}'...")
            tokenizer_modernbert = AutoTokenizer.from_pretrained(model_dir)
            model_modernbert = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
            model_modernbert.eval()
            print(f"[FastAPI Startup] ModernBERT loaded successfully!")
        except Exception as e:
            print(f"[FastAPI Startup] Error preloading ModernBERT: {e}")
            
    # Load RoBERTa (512 micro sentence precision)
    if os.path.exists(roberta_dir) and os.path.exists(os.path.join(roberta_dir, "config.json")):
        try:
            print(f"[FastAPI Startup] Loading RoBERTa-v2 (512) from '{roberta_dir}'...")
            tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_dir)
            model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_dir).to(device)
            model_roberta.eval()
            print(f"[FastAPI Startup] RoBERTa-v2 loaded successfully!")
        except Exception as e:
            print(f"[FastAPI Startup] Error preloading RoBERTa: {e}")

    if model_modernbert is not None and model_roberta is not None:
        print(f"[FastAPI Startup] HYBRID ENSEMBLE ENGINE ACTIVE on {device}! (65% ModernBERT + 35% RoBERTa)")
    elif model_modernbert is not None:
        print(f"[FastAPI Startup] ModernBERT Standalone active on {device}.")
    elif model_roberta is not None:
        print(f"[FastAPI Startup] RoBERTa Standalone active on {device}.")
    else:
        print(f"[FastAPI Startup] App running in SIMULATION MODE until fine-tuned weights exist.")
        
    yield
    print("[FastAPI Shutdown] Unloading Hybrid Ensemble Engine models...")

app = FastAPI(
    title="VeriPaper AI Text Detector API",
    description="Local verification service for identifying AI-generated writing in academic research papers.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Global prediction cache to ensure single-execution AI text detection
analysis_cache = {}

def split_into_smart_paragraphs(text: str) -> list[str]:
    """
    Intelligently splits raw extracted document text into structural paragraphs, headings,
    title metadata, affiliations, itemized lists, and reference entries.
    """
    if not text:
        return []
        
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Insert explicit double-newline breaks before structural section markers embedded inside lines
    text = re.sub(r'(?<=\S)\s*(\b(?:Abstract|Index Terms|1\s+Introduction|Introduction|Background|Related Work|Methodology|Experiments|Results|Discussion|Conclusion|References|Bibliography)\b[\s—\-\.\:]+)', r'\n\n\1', text, flags=re.IGNORECASE)
    
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    paragraphs = []
    
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) <= 1:
            paragraphs.append(block)
            continue
            
        current_chunk = []
        for line in lines:
            # Check for section headings, title numbers, affiliations, emails, keywords, and citations
            is_heading = bool(re.match(
                r'^(?:[0-9]+\.|\d+\s+[A-Z]|\b(?:abstract|introduction|background|related work|methodology|experiments|results|discussion|conclusion|references|works cited|bibliography|department|school|university|e-mail|email|keywords|author|authors|independent research|july \d{4}|june \d{4}|august \d{4}|january \d{4}|february \d{4}|march \d{4}|april \d{4}|may \d{4}|september \d{4}|october \d{4}|november \d{4}|december \d{4})\b)', 
                line, 
                re.IGNORECASE
            ))
            is_bracket_ref = bool(re.match(r'^\[\d+\]', line))
            is_author_ref = bool(re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.?(?:\s+&\s+[A-Z][a-z]+,?\s+[A-Z]\.?)?\s+\(\d{4}\)', line))
            is_bullet = bool(re.match(r'^(?:[\*\-\•]|\d+\.)\s+', line))
            
            # Treat short standalone title/author/metadata header lines as separate blocks
            is_short_header = (len(line) < 70 and not line.endswith("."))
            
            if (is_heading or is_bracket_ref or is_author_ref or is_bullet or is_short_header) and current_chunk:
                paragraphs.append(" ".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            paragraphs.append(" ".join(current_chunk))
            
    return [p for p in paragraphs if p.strip()]

def run_analysis_pipeline(doc_text: str, filename: str, page_count: int, page_images: list = None, extracted_figures: list = None, pdf_path: str = None):
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
            preloaded_model=model_modernbert,
            preloaded_tokenizer=tokenizer_modernbert,
            preloaded_roberta_model=model_roberta,
            preloaded_roberta_tokenizer=tokenizer_roberta,
            roberta_dir=roberta_dir
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
    raw_paragraphs = split_into_smart_paragraphs(doc_text)
    sentence_idx = 0

    
    for p_idx, p_text in enumerate(raw_paragraphs):
        p_sentences = segment_sentences(p_text)
        for p_sent in p_sentences:
            if sentence_idx >= len(predictions):
                break
                
            pred = predictions[sentence_idx]
            score = pred["score"]
            
            if score >= 0.75:
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
        if score >= 0.75:
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

    flagged_words = 0.0
    doc_words_total = 0

    for item in processed_sentences:
        text = item["text"]
        score = item["ai_probability"]
        words = len([w for w in text.split() if w.strip()])
        doc_words_total += words
        if score >= 0.75:
            flagged_words += words * 1.0
        elif score >= 0.60:
            flagged_words += words * max(0.85, score)
        elif score >= 0.50:
            flagged_words += words * 0.50

    if doc_words_total > 0:
        overall_percentage = round(min(100.0, (flagged_words / doc_words_total) * 100.0), 1)
    else:
        overall_percentage = 0.0

    cache_entry = {
        "filename": filename,
        "predictions": predictions,
        "processed_sentences": [
            {
                "sentence": s["text"],
                "score": s["ai_probability"],
                "confidence_tier": s["confidence_tier"] if s["confidence_tier"] != "unflagged" else "none"
            }
            for s in processed_sentences
        ],
        "overall_percentage": overall_percentage,
        "metrics": {
            "high_confidence_count": flagged_high,
            "medium_confidence_count": flagged_mid,
            "unflagged_count": unflagged
        },
        "word_count": word_count,
        "char_count": len(doc_text)
    }
    analysis_cache[filename] = cache_entry
    analysis_cache["_latest"] = cache_entry

    from src.section_analyzer import analyze_paper_sections
    section_analysis = analyze_paper_sections(doc_text, processed_sentences)

    if pdf_path and os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
        try:
            from src.report_generator import render_highlighted_pdf_page_images
            hl_images = render_highlighted_pdf_page_images(pdf_path, processed_sentences)
            if hl_images:
                page_images = hl_images
        except Exception as e:
            print(f"[run_analysis_pipeline] PDF visual highlight rendering warning: {e}")

    return {
        "paragraphs": raw_paragraphs,
        "text": doc_text,
        "metadata": {
            "filename": filename,
            "page_count": page_count,
            "word_count": word_count
        },
        "overall_ai_percentage": overall_percentage,
        "sentences": processed_sentences,
        "section_analysis": section_analysis,
        "page_images": page_images or [],
        "extracted_figures": extracted_figures or [],
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

    # 4. Extract document text & visual page renderings
    doc_text = ""
    page_count = 0
    page_images = []
    extracted_figures = []
    
    try:
        if ext == ".pdf":
            from src.document_parser import extract_pdf_visual_and_text
            pdf_data = extract_pdf_visual_and_text(temp_path)
            doc_text = pdf_data["text"]
            page_count = pdf_data["page_count"]
            page_images = pdf_data["page_images"]
            extracted_figures = pdf_data["extracted_figures"]
        else:
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

    try:
        return run_analysis_pipeline(doc_text, file.filename, page_count, page_images, extracted_figures, pdf_path=temp_path)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


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
        
    # Extract PDF text & visual page renderings using PyMuPDF
    doc_text = ""
    page_count = 0
    page_images = []
    extracted_figures = []
    try:
        from src.document_parser import extract_pdf_visual_and_text
        pdf_data = extract_pdf_visual_and_text(file_path)
        doc_text = pdf_data["text"]
        page_count = pdf_data["page_count"]
        page_images = pdf_data["page_images"]
        extracted_figures = pdf_data["extracted_figures"]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse sample PDF: {str(e)}"
        )

    return run_analysis_pipeline(doc_text, filename, page_count, page_images, extracted_figures, pdf_path=file_path)


@app.post("/report")
@app.get("/report")
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
        
        cache_key = sample_type if sample_type else (filename if filename in analysis_cache else "_latest")
        cached = analysis_cache.get(cache_key) or analysis_cache.get(filename) or analysis_cache.get("_latest")

        if cached and cached.get("processed_sentences"):
            print(f"[Cache Hit] Reusing pre-computed AI text detection results for '{filename}'. Skipping redundant neural model inference!")
            smoothed = cached["processed_sentences"]
            overall_percentage = cached["overall_percentage"]
            metrics = cached["metrics"]
            metrics["file_size_bytes"] = size
            word_count = cached["word_count"]
            char_count = cached["char_count"]
        else:
            print(f"[Cache Miss] Executing AI text detection for report generation on '{filename}'...")
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
                preloaded_model=model_modernbert,
                preloaded_tokenizer=tokenizer_modernbert,
                preloaded_roberta_model=model_roberta,
                preloaded_roberta_tokenizer=tokenizer_roberta,
                roberta_dir=roberta_dir
            )
            
            # Apply smoothing
            smoothed = smooth_predictions(predictions)
            
            # Compute breakdown metrics
            flagged_high = 0
            flagged_mid = 0
            unflagged = 0
            
            for pred in smoothed:
                score = pred["score"]
                if score >= 0.75:
                    pred["confidence_tier"] = "high"
                    flagged_high += 1
                elif score >= 0.60:
                    pred["confidence_tier"] = "medium"
                    flagged_mid += 1
                else:
                    pred["confidence_tier"] = "none"
                    unflagged += 1
                    
            ai_weighted_words = 0.0
            doc_words_total = 0

            for pred in smoothed:
                text = pred["sentence"]
                score = pred["score"]
                words = len([w for w in text.split() if w.strip()])
                doc_words_total += words
                if score >= 0.60:
                    ai_weighted_words += words * score

            if doc_words_total > 0:
                overall_percentage = round((ai_weighted_words / doc_words_total) * 100, 1)
            else:
                overall_percentage = 0.0
            
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


# Mount additive diagnostic APIRouter for adversarial robustness evaluation
try:
    from robustness_eval.router import router as robustness_router
    app.include_router(robustness_router)
except Exception as e:
    print(f"[Warning] Failed to mount robustness router: {e}")


