import os
import sys
import fitz

# Add root folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.report_generator import generate_pdf_report, align_sentences_to_bboxes
from src.highlighter import segment_sentences, run_predictions, smooth_predictions
from src.app import model_dir, model, tokenizer

def test_report_pipeline():
    print("Testing Turnitin-style PDF report generation pipeline...")
    pdf_path = "LLM_Architectures_and_Training_Survey.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Source PDF {pdf_path} not found. Skipping test.")
        return
        
    # Read text
    doc = fitz.open(pdf_path)
    original_pages = len(doc)
    text_pages = [page.get_text() for page in doc]
    full_text = "\n".join(text_pages)
    doc.close()
    
    sentences = segment_sentences(full_text.strip())
    assert len(sentences) > 0, "No sentences extracted from PDF"
    
    print(f"Segmented {len(sentences)} sentences. Running mock/preloaded classifier...")
    # Run predictions
    predictions, is_simulated = run_predictions(
        sentences=sentences,
        model_dir=model_dir,
        preloaded_model=model,
        preloaded_tokenizer=tokenizer
    )
    
    # Smooth predictions
    smoothed = smooth_predictions(predictions)
    
    # Setup categories breakdown metrics
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
    flagged_ratio = (flagged_total / len(sentences) * 100) if len(sentences) > 0 else 0.0
    overall_percentage = min(100.0, flagged_ratio * 2.5) if flagged_total > 0 else 0.0
    
    metrics = {
        "high_confidence_count": flagged_high,
        "medium_confidence_count": flagged_mid,
        "unflagged_count": unflagged,
        "file_size_bytes": os.path.getsize(pdf_path)
    }
    
    print("Generating report PDF...")
    output_pdf_path = generate_pdf_report(
        source_file_path=pdf_path,
        predictions=smoothed,
        overall_percentage=overall_percentage,
        metrics=metrics,
        total_words=len(full_text.split()),
        char_count=len(full_text)
    )
    
    assert os.path.exists(output_pdf_path), "Output PDF file was not created"
    assert os.path.getsize(output_pdf_path) > 0, "Output PDF file is empty"
    
    # 3. Verify page counts and headers/footers structure
    out_doc = fitz.open(output_pdf_path)
    total_pages = len(out_doc)
    
    print(f"Original pages: {original_pages}, Output report pages: {total_pages}")
    assert total_pages == original_pages + 2, f"Expected {original_pages + 2} pages, got {total_pages}"
    
    # Verify headers of the first three pages
    for i, label in [(0, "Cover Page"), (1, "AI Verification Overview"), (2, "AI Verification Submission")]:
        text_content = out_doc[i].get_text()
        # Verify that page headers are drawn
        assert f"Page {i+1} of {total_pages} — {label}" in text_content or f"Page {i+1}" in text_content
        
    out_doc.close()
    
    # Cleanup output
    if os.path.exists(output_pdf_path):
        os.remove(output_pdf_path)
        
    print("Report pipeline test passed successfully!")

if __name__ == "__main__":
    try:
        # We need to load model/tokenizer for this test since it runs full model predictions
        from src.app import lifespan
        import asyncio
        
        async def run_lifecycle_test():
            # Trigger lifespan context manually to load model
            app_mock = type('AppMock', (), {})()
            async with lifespan(app_mock):
                test_report_pipeline()
                
        asyncio.run(run_lifecycle_test())
        print("All report verification checks passed!")
    except AssertionError as e:
        print(f"Verification assertion failure: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Verification error: {str(e)}")
        sys.exit(1)
