import os
import sys
import re
import subprocess
import math
import uuid
import datetime
import shutil
import tempfile
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import classifiers and tokenizers from current project
from src.highlighter import segment_sentences, run_predictions, smooth_predictions
from src.document_parser import find_libreoffice, extract_text_from_doc

class ReportGenerationError(Exception):
    pass

def clean_text_for_alignment(text: str) -> str:
    """Removes all whitespaces, newlines, and control chars for robust alignment matching."""
    return re.sub(r'[\s\u200b-\u200d\ufeff\-\u2010-\u2015\u201c\u201d\u2018\u2019"\'.,\/#!$%\^&\*;:{}=\-_`~()\[\]]+', '', text).lower()

def align_sentences_to_bboxes(pdf_path: str, sentences: list[str]) -> list[dict]:
    """
    Extracts all words with their coordinates from a PDF, and aligns each 
    tokenized sentence to its exact bounding boxes on each page using a 
    character-level index mapping. This is layout-independent and handles page-breaks.
    """
    doc = fitz.open(pdf_path)
    words_list = []
    full_text_buffer = []
    char_to_word_idx = []
    
    # 1. Gather all words from all pages
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        words = page.get_text("words")
        # Sort words primarily by top position (y0), then left position (x0) to match reading order
        words = sorted(words, key=lambda w: (w[1], w[0]))
        
        for w in words:
            word_text = w[4]
            words_list.append({
                "text": word_text,
                "bbox": (w[0], w[1], w[2], w[3]),
                "page": page_idx
            })
            
            # Map index characters back to the current word
            full_text_buffer.append(word_text)
            full_text_buffer.append(" ")
            word_len = len(word_text)
            for _ in range(word_len + 1):  # include space character
                char_to_word_idx.append(len(words_list) - 1)
                
    full_doc_text = "".join(full_text_buffer)
    clean_doc = clean_text_for_alignment(full_doc_text)
    
    # Map index from cleaned doc back to the original full text positions
    clean_to_orig_idx = []
    orig_pos = 0
    for char in full_doc_text:
        char_clean = clean_text_for_alignment(char)
        if char_clean:
            clean_to_orig_idx.append(orig_pos)
        orig_pos += 1
        
    aligned_sentences = []
    search_start_pos = 0
    
    # 2. Match each sentence (or its constituent sub-parts if multi-line) in the text stream
    for sent in sentences:
        sub_parts = [p.strip() for p in re.split(r'[\n\r]+', sent) if p.strip()]
        if not sub_parts:
            sub_parts = [sent]

        bboxes_by_page = {}

        for part in sub_parts:
            part_clean = clean_text_for_alignment(part)
            if len(part_clean) < 3:
                continue
                
            match_idx = clean_doc.find(part_clean, search_start_pos)
            if match_idx == -1:
                match_idx = clean_doc.find(part_clean, 0)
                
            if match_idx != -1:
                try:
                    orig_start = clean_to_orig_idx[match_idx]
                    orig_end = clean_to_orig_idx[match_idx + len(part_clean) - 1]
                    
                    start_word_idx = char_to_word_idx[orig_start]
                    end_word_idx = char_to_word_idx[orig_end]
                    
                    for w_idx in range(start_word_idx, end_word_idx + 1):
                        word_info = words_list[w_idx]
                        p = word_info["page"]
                        if p not in bboxes_by_page:
                            bboxes_by_page[p] = []
                        bboxes_by_page[p].append(word_info["bbox"])
                        
                    search_start_pos = match_idx + len(part_clean)
                except Exception:
                    pass

        if bboxes_by_page:
            aligned_sentences.append({
                "sentence": sent,
                "pages_bboxes": bboxes_by_page
            })
                
    doc.close()
    return aligned_sentences

def draw_page_background(canvas, doc):
    """Draws the VeriPaper AI cream-bond background canvas color."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#FAF8F2"))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=True, stroke=False)
    canvas.restoreState()

def create_reportlab_pages(
    temp_pdf_path: str,
    filename: str,
    vpr_id: str,
    overall_percentage: float,
    metrics: dict,
    total_words: int,
    char_count: int,
    pdf_pages: int
) -> None:
    """Generates the Cover Page (Page 1) and Overview Page (Page 2) in ReportLab."""
    # Setup document template with cream background callback
    doc_template = SimpleDocTemplate(
        temp_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Brand Palette styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#7A2331"),  # oxblood
        spaceAfter=15,
        alignment=0
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#7A2331"),
        spaceBefore=10,
        spaceAfter=10,
        alignment=0
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#21242B"),
        spaceAfter=8
    )
    
    mono_style = ParagraphStyle(
        'ReportMono',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#21242B")
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#21242B")
    )
    
    story = []
    
    # --- PAGE 1: COVER PAGE ---
    # Brand logo representation
    story.append(Spacer(1, 40))
    story.append(Paragraph("<font size=16 color='#7A2331'><b>VeriPaper AI</b></font><font size=10 color='#7A2331'> • FORENSICS REPORT</font>", body_style))
    story.append(Spacer(1, 60))
    
    # Document Title
    clean_filename = os.path.basename(filename)
    display_title = clean_filename.replace('_', ' ').replace('.pdf', '').replace('.docx', '').replace('.doc', '')
    story.append(Paragraph(f"Forensic Manuscript Verification Report", title_style))
    story.append(Paragraph(f"<b>Manuscript:</b> {display_title}", body_style))
    story.append(Spacer(1, 40))
    
    # Details Box Table
    analysis_date = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    file_size_kb = f"{metrics.get('file_size_bytes', 0) / 1024:.1f} KB"
    
    details_data = [
        [Paragraph("<b>Verification ID</b>", body_style), Paragraph(vpr_id, mono_style)],
        [Paragraph("<b>Analysis Date</b>", body_style), Paragraph(analysis_date, body_style)],
        [Paragraph("<b>Source File Name</b>", body_style), Paragraph(clean_filename, body_style)],
        [Paragraph("<b>File Size</b>", body_style), Paragraph(file_size_kb, body_style)],
        [Paragraph("<b>Manuscript Volume</b>", body_style), Paragraph(f"{pdf_pages} pages", body_style)],
        [Paragraph("<b>Word Count</b>", body_style), Paragraph(f"{total_words} words", body_style)],
        [Paragraph("<b>Character Count</b>", body_style), Paragraph(f"{char_count} characters", body_style)]
    ]
    
    details_table = Table(details_data, colWidths=[150, 350])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FAF8F2")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DCD4C0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(Paragraph("DOCUMENT METADATA", section_style))
    story.append(details_table)
    story.append(Spacer(1, 60))
    
    # Visual Seal Icon in ReportLab
    story.append(Paragraph("<font color='#7A2331'>[ VERIFICATION CERTIFICATE PENDING REVIEW ]</font>", mono_style))
    story.append(PageBreak())
    
    # --- PAGE 2: AI VERIFICATION OVERVIEW ---
    story.append(Spacer(1, 20))
    story.append(Paragraph("AI VERIFICATION SUMMARY", section_style))
    story.append(Spacer(1, 10))
    
    # Score headline
    story.append(Paragraph(f"<font size=48 color='#7A2331'><b>{overall_percentage:.0f}%</b></font><font size=16 color='#7A2331'><b> detected as AI-generated</b></font>", title_style))
    story.append(Paragraph("This percentage represents the cumulative sentence-level portion flagged with high or medium neural synthetic probability.", body_style))
    story.append(Spacer(1, 20))
    
    # Caution Box
    caution_data = [[
        Paragraph("<b>Caution: Review Required</b><br/>"
                  "Statistical AI-detection tools estimate probability based on localized writing patterns (lexical style and structural burstiness). "
                  "This report is a decision-support indicator designed to guide review. It should not be used as absolute proof of AI authorship, "
                  "and requires human assessment alongside the author's reference materials.", callout_style)
    ]]
    caution_table = Table(caution_data, colWidths=[500])
    caution_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FAF8F2")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#B8862E")), # Amber border
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(caution_table)
    story.append(Spacer(1, 30))
    
    # Breakdown Table
    story.append(Paragraph("FORENSIC SEGMENT BREAKDOWN", section_style))
    
    # Calculate counts
    flagged_high = metrics.get("high_confidence_count", 0)
    flagged_mid = metrics.get("medium_confidence_count", 0)
    unflagged = metrics.get("unflagged_count", 0)
    total_sent = flagged_high + flagged_mid + unflagged
    
    high_pct = (flagged_high / total_sent * 100) if total_sent > 0 else 0
    mid_pct = (flagged_mid / total_sent * 100) if total_sent > 0 else 0
    human_pct = (unflagged / total_sent * 100) if total_sent > 0 else 0
    
    breakdown_data = [
        [Paragraph("<b>Category</b>", body_style), Paragraph("<b>Sentences</b>", body_style), Paragraph("<b>Ratio</b>", body_style)],
        [Paragraph("<font color='#BA2D22'>■</font> High Confidence AI", body_style), Paragraph(str(flagged_high), body_style), Paragraph(f"{high_pct:.1f}%", body_style)],
        [Paragraph("<font color='#B8862E'>■</font> Medium Confidence AI", body_style), Paragraph(str(flagged_mid), body_style), Paragraph(f"{mid_pct:.1f}%", body_style)],
        [Paragraph("■ Verified Scholarly Ink (Human)", body_style), Paragraph(str(unflagged), body_style), Paragraph(f"{human_pct:.1f}%", body_style)]
    ]
    
    breakdown_table = Table(breakdown_data, colWidths=[220, 140, 140])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FAF8F2")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DCD4C0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 40))
    
    # Disclaimer Paragraph
    story.append(Paragraph("<b>Report Disclaimer</b>", body_style))
    story.append(Paragraph(
        "This forensics document is generated dynamically by VeriPaper AI. "
        "It employs deep neural sentence classification to detect characteristic artifacts left by LLMs. "
        "Because human writing can occasionally mimic low-burstiness patterns, and AI writing can be heavily edited, "
        "the results must be treated as guidance. Institutional policies and manual investigation must dictate any "
        "scholarly integrity outcomes.", body_style
    ))
    
    # Build document
    doc_template.build(story, onFirstPage=draw_page_background, onLaterPages=draw_page_background)

def generate_pdf_report(
    source_file_path: str,
    predictions: list[dict],
    overall_percentage: float,
    metrics: dict,
    total_words: int,
    char_count: int
) -> str:
    """
    Main pipeline to overlay coordinates highlights onto the original document,
    generate Cover and Summary pages, and merge them with running headers/footers.
    """
    if not predictions:
        from src.document_parser import extract_text
        from src.app import model_dir, roberta_dir, model_modernbert, tokenizer_modernbert, model_roberta, tokenizer_roberta
        doc_text_raw, _ = extract_text(source_file_path)
        doc_text_clean = doc_text_raw.strip()
        sentences_for_pipeline = segment_sentences(doc_text_clean)
        predictions_pipeline, _ = run_predictions(
            sentences=sentences_for_pipeline,
            model_dir=model_dir,
            preloaded_model=model_modernbert,
            preloaded_tokenizer=tokenizer_modernbert,
            preloaded_roberta_model=model_roberta,
            preloaded_roberta_tokenizer=tokenizer_roberta,
            roberta_dir=roberta_dir
        )
        predictions = smooth_predictions(predictions_pipeline)

    # Ensure all predictions have normalized keys and confidence tiers matching the web app
    flagged_high = 0
    flagged_mid = 0
    unflagged = 0

    for pred in predictions:
        if "sentence" not in pred and "text" in pred:
            pred["sentence"] = pred["text"]
        score = pred.get("score", pred.get("ai_probability", 0.0))
        pred["score"] = float(score)

        tier = pred.get("confidence_tier")
        if not tier or tier == "unflagged":
            if score >= 0.75:
                tier = "high"
            elif score >= 0.60:
                tier = "medium"
            else:
                tier = "none"
        pred["confidence_tier"] = tier

        if tier == "high":
            flagged_high += 1
        elif tier == "medium":
            flagged_mid += 1
        else:
            unflagged += 1

    overall_percentage = round(float(overall_percentage), 1)
    metrics["file_size_bytes"] = os.path.getsize(source_file_path)

    is_docx = source_file_path.lower().endswith(".docx") or source_file_path.lower().endswith(".doc")
    temp_pdf_path = None
    
    # 1. If DOCX/DOC, convert to PDF first
    if is_docx:
        temp_dir = tempfile.gettempdir()
        if source_file_path.lower().endswith(".doc"):
            # doc legacy conversion
            pdf_tuple = extract_text_from_doc(source_file_path)
            # wait, extract_text_from_doc does conversion and returns text, but we need the actual converted PDF file!
            # Let's write a small helper to convert DOCX/DOC to PDF directly
        
        # Standard conversion helper
        soffice_bin = find_libreoffice()
        if not soffice_bin:
            raise ReportGenerationError(
                "LibreOffice is required to generate reports from Word (.doc/.docx) files but was not found."
            )
            
        with tempfile.TemporaryDirectory() as conv_dir:
            try:
                result = subprocess.run(
                    [soffice_bin, "--headless", "--convert-to", "pdf", "--outdir", conv_dir, source_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=25
                )
                if result.returncode != 0:
                    raise ReportGenerationError(f"LibreOffice conversion failed: {result.stderr}")
            except Exception as e:
                raise ReportGenerationError(f"LibreOffice PDF conversion execution failed: {str(e)}")
                
            base_name = os.path.splitext(os.path.basename(source_file_path))[0]
            conv_pdf = os.path.join(conv_dir, f"{base_name}.pdf")
            if not os.path.exists(conv_pdf):
                files = [f for f in os.listdir(conv_dir) if f.endswith(".pdf")]
                if files:
                    conv_pdf = os.path.join(conv_dir, files[0])
                else:
                    raise ReportGenerationError("Converted PDF file not found.")
                    
            # Move to a persistent temporary path
            temp_pdf_path = os.path.join(temp_dir, f"report_conv_{uuid.uuid4().hex}.pdf")
            shutil.copyfile(conv_pdf, temp_pdf_path)
            pdf_path_to_highlight = temp_pdf_path
    else:
        pdf_path_to_highlight = source_file_path

    # Create temporary copies and paths
    temp_dir = tempfile.gettempdir()
    highlighted_pdf_path = os.path.join(temp_dir, f"highlighted_{uuid.uuid4().hex}.pdf")
    rl_cover_pdf_path = os.path.join(temp_dir, f"cover_{uuid.uuid4().hex}.pdf")
    final_output_pdf_path = os.path.join(temp_dir, f"final_report_{uuid.uuid4().hex}.pdf")

    try:
        # 2. Extract and align sentences with coordinates in the PDF
        sentences_list = [pred["sentence"] for pred in predictions]
        aligned_data = align_sentences_to_bboxes(pdf_path_to_highlight, sentences_list)
        
        # 3. Draw highlights on a copy of the PDF
        shutil.copyfile(pdf_path_to_highlight, highlighted_pdf_path)
        dest_doc = fitz.open(highlighted_pdf_path)
        
        # Build normalized lookup for prediction scores (handles exact string, stripped string, and clean text keys)
        pred_map = {}
        for pred in predictions:
            text_key = pred.get("sentence", pred.get("text", ""))
            pred_map[text_key] = pred
            pred_map[text_key.strip()] = pred
            clean_k = clean_text_for_alignment(text_key)
            if clean_k:
                pred_map[clean_k] = pred
        
        # Colors matching the brand theme
        color_high = (186/255, 45/255, 34/255)    # hex #BA2D22 red
        color_mid = (184/255, 134/255, 46/255)    # hex #B8862E amber
        
        for aligned in aligned_data:
            sent = aligned["sentence"]
            pred = pred_map.get(sent) or pred_map.get(sent.strip()) or pred_map.get(clean_text_for_alignment(sent))
            if not pred:
                continue
                
            tier = pred.get("confidence_tier")
            score = float(pred.get("score", pred.get("ai_probability", 0.0)))
            if not tier or tier == "unflagged":
                if score >= 0.75:
                    tier = "high"
                elif score >= 0.60:
                    tier = "medium"
                else:
                    tier = "none"

            if tier == "high":
                color = color_high
            elif tier == "medium":
                color = color_mid
            else:
                continue  # no highlight for human text
                
            for page_idx, bboxes in aligned["pages_bboxes"].items():
                if page_idx >= len(dest_doc):
                    continue
                page = dest_doc[page_idx]
                
                # Draw semi-transparent rectangle on page for each bounding box
                for bbox in bboxes:
                    rect = fitz.Rect(bbox)
                    # To prevent drawing flat lines or invalid small rects
                    if rect.width > 2 and rect.height > 2:
                        page.draw_rect(
                            rect,
                            color=None,
                            fill=color,
                            fill_opacity=0.20,
                            width=0
                        )
                        
        temp_save_path = os.path.join(temp_dir, f"temp_save_{uuid.uuid4().hex}.pdf")
        dest_doc.save(temp_save_path)
        pdf_pages_count = len(dest_doc)
        dest_doc.close()
        shutil.copyfile(temp_save_path, highlighted_pdf_path)
        try:
            os.remove(temp_save_path)
        except Exception:
            pass

        # 4. Generate Page 1 and Page 2 using ReportLab
        vpr_id = f"vpr:oid::{uuid.uuid4().hex[:12]}"
        create_reportlab_pages(
            temp_pdf_path=rl_cover_pdf_path,
            filename=source_file_path,
            vpr_id=vpr_id,
            overall_percentage=overall_percentage,
            metrics=metrics,
            total_words=total_words,
            char_count=char_count,
            pdf_pages=pdf_pages_count
        )

        # 5. Merge PDF Pages: [ReportLab Cover & Summary] + [Highlighted Original PDF]
        final_doc = fitz.open()
        
        rl_doc = fitz.open(rl_cover_pdf_path)
        highlighted_doc = fitz.open(highlighted_pdf_path)
        
        # Insert ReportLab pages (Cover = Page 1, Overview = Page 2)
        final_doc.insert_pdf(rl_doc)
        # Insert original highlighted pages
        final_doc.insert_pdf(highlighted_doc)
        
        total_pages = len(final_doc)
        
        # 6. Apply Running Headers/Footers on all pages
        # Text/Rule colors matching our palette
        ink_color = (33/255, 36/255, 43/255)       # hex #21242B
        muted_gray = (120/255, 120/255, 120/255)
        rule_color = (220/255, 212/255, 192/255)   # hex #DCD4C0
        
        for idx in range(total_pages):
            page = final_doc[idx]
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Determine page type for running header label
            if idx == 0:
                header_label = "Cover Page"
            elif idx == 1:
                header_label = "AI Verification Overview"
            else:
                header_label = "AI Verification Submission"
                
            header_text = f"Page {idx + 1} of {total_pages} — {header_label}"
            
            # Draw Header running texts
            # Right-aligned header ID
            id_text = f"ID: {vpr_id}"
            id_len = fitz.get_text_length(id_text, fontname="courier", fontsize=8)
            page.insert_text(
                fitz.Point(width - 54 - id_len, 30),
                id_text,
                fontsize=8,
                fontname="courier",
                color=muted_gray
            )
            # Left-aligned header text
            page.insert_text(
                fitz.Point(54, 30),
                header_text,
                fontsize=8,
                fontname="helvetica",
                color=muted_gray
            )
            
            # Draw Header hairline divider rule
            page.draw_line(
                fitz.Point(54, 38),
                fitz.Point(width - 54, 38),
                color=rule_color,
                width=0.5
            )
            
            # Draw Footer hairline divider rule
            page.draw_line(
                fitz.Point(54, height - 38),
                fitz.Point(width - 54, height - 38),
                color=rule_color,
                width=0.5
            )
            
            # Draw Footer running texts
            # Left footer
            page.insert_text(
                fitz.Point(54, height - 28),
                "VeriPaper AI — Document Forensics Instrument",
                fontsize=8,
                fontname="helvetica",
                color=muted_gray
            )
            # Right footer
            page_num_text = f"Page {idx + 1}"
            page_len = fitz.get_text_length(page_num_text, fontname="helvetica-bold", fontsize=8)
            page.insert_text(
                fitz.Point(width - 54 - page_len, height - 28),
                page_num_text,
                fontsize=8,
                fontname="helvetica-bold",
                color=ink_color
            )

        final_doc.save(final_output_pdf_path)
        final_doc.close()
        rl_doc.close()
        highlighted_doc.close()
        
        return final_output_pdf_path
        
    finally:
        # Cleanup temporary files
        for path in [temp_pdf_path, highlighted_pdf_path, rl_cover_pdf_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

import base64

def render_highlighted_pdf_page_images(pdf_path: str, processed_sentences: list[dict]) -> list[dict]:
    """
    Renders high-res visual page images with sentence AI probability highlights
    (Red for High Risk, Yellow/Amber for Medium Risk) overlaid directly onto PyMuPDF pages
    for the web app PDF canvas mode.
    """
    if not pdf_path or not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        return []

    try:
        doc = fitz.open(pdf_path)
        sentences_list = [s.get("sentence", s.get("text", "")) for s in processed_sentences if s.get("confidence_tier") in ["high", "medium"]]
        
        # If no sentences flagged, return clean high-res rendered pages
        if not sentences_list:
            page_images = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                base64_img = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
                page_images.append({
                    "page_number": i + 1,
                    "image_data": base64_img,
                    "width": pix.width,
                    "height": pix.height
                })
            doc.close()
            return page_images

        aligned_data = align_sentences_to_bboxes(pdf_path, sentences_list)
        pred_map = {}
        for s in processed_sentences:
            text_k = s.get("sentence", s.get("text", ""))
            pred_map[text_k] = s
            pred_map[text_k.strip()] = s
            clean_k = clean_text_for_alignment(text_k)
            if clean_k:
                pred_map[clean_k] = s

        # Create temporary working copy to draw highlights
        temp_dir = tempfile.gettempdir()
        temp_hl_pdf = os.path.join(temp_dir, f"render_hl_{uuid.uuid4().hex}.pdf")
        shutil.copyfile(pdf_path, temp_hl_pdf)
        hl_doc = fitz.open(temp_hl_pdf)

        color_high = (186/255, 45/255, 34/255)    # hex #BA2D22 red
        color_mid = (184/255, 134/255, 46/255)    # hex #B8862E amber

        for aligned in aligned_data:
            sent = aligned["sentence"]
            pred = pred_map.get(sent) or pred_map.get(sent.strip()) or pred_map.get(clean_text_for_alignment(sent))
            if not pred:
                continue

            tier = pred.get("confidence_tier")
            score = float(pred.get("score", pred.get("ai_probability", 0.0)))
            if not tier or tier == "unflagged":
                if score >= 0.75:
                    tier = "high"
                elif score >= 0.60:
                    tier = "medium"
                else:
                    tier = "none"

            if tier == "high":
                color = color_high
            elif tier == "medium":
                color = color_mid
            else:
                continue

            for page_idx, bboxes in aligned["pages_bboxes"].items():
                if page_idx >= len(hl_doc):
                    continue
                page = hl_doc[page_idx]
                for bbox in bboxes:
                    rect = fitz.Rect(bbox)
                    if rect.width > 2 and rect.height > 2:
                        page.draw_rect(
                            rect,
                            color=None,
                            fill=color,
                            fill_opacity=0.25,
                            width=0
                        )

        # Render page pixmaps with highlights to base64 PNG
        page_images = []
        for i, page in enumerate(hl_doc):
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            base64_img = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            page_images.append({
                "page_number": i + 1,
                "image_data": base64_img,
                "width": pix.width,
                "height": pix.height
            })

        hl_doc.close()
        doc.close()
        if os.path.exists(temp_hl_pdf):
            try:
                os.remove(temp_hl_pdf)
            except Exception:
                pass

        return page_images
    except Exception as e:
        print(f"[render_highlighted_pdf_page_images] Warning: {e}")
        return []
