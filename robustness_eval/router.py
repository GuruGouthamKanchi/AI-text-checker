from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from robustness_eval.humanizer.humanizer_engine import humanize_text

router = APIRouter(prefix="/robustness", tags=["Adversarial Robustness & Text Humanizer"])

class RobustnessEvalRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text payload to humanize / evaluate against adversarial rewriting.")
    tone: Optional[str] = Field("resume", description="Target humanization tone: 'resume', 'academic', or 'professional'.")

class RobustnessEvalResponse(BaseModel):
    original_text: str = Field(..., description="Original input text.")
    humanized_text: str = Field(..., description="Adversarially humanized text.")
    rewritten_text: str = Field(..., description="Adversarially humanized text (alias).")
    original_ai_percentage: float = Field(..., description="Detector AI percentage score on original text.")
    humanized_ai_percentage: float = Field(..., description="Detector AI percentage score on humanized rewritten text.")
    rewritten_ai_percentage: float = Field(..., description="Detector AI percentage score on humanized rewritten text (alias).")
    robustness_confidence_delta: float = Field(..., description="Score degradation delta (original - rewritten).")
    resilience_verdict: str = Field(..., description="Diagnostic classification of model resilience against rewriting.")
    explanation: str = Field(..., description="Humanizer and robustness summary.")

@router.post("/evaluate", response_model=RobustnessEvalResponse)
@router.post("/humanize", response_model=RobustnessEvalResponse)
async def evaluate_robustness_endpoint(payload: RobustnessEvalRequest):
    """
    AI Text Humanizer & Robustness Diagnostic Endpoint:
    Rewrites AI text into natural personal human voice using active phrasing,
    cliché elimination, and neural seq2seq paraphrasing. Returns humanized text & AI scores.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload text cannot be empty."
        )

    try:
        tone = payload.tone or "resume"
        result = humanize_text(text=text, tone=tone)

        return RobustnessEvalResponse(
            original_text=result["original_text"],
            humanized_text=result["humanized_text"],
            rewritten_text=result["humanized_text"],
            original_ai_percentage=result["original_ai_percentage"],
            humanized_ai_percentage=result["humanized_ai_percentage"],
            rewritten_ai_percentage=result["humanized_ai_percentage"],
            robustness_confidence_delta=result["score_reduction"],
            resilience_verdict=result["resilience_verdict"],
            explanation=result["explanation"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI text humanization failed: {str(e)}"
        )

import json
from fastapi.responses import StreamingResponse
from robustness_eval.humanizer.humanizer_engine import get_humanizer_engine

@router.post("/humanize/stream")
async def humanize_stream_endpoint(payload: RobustnessEvalRequest):
    """
    Real-Time Server-Sent Events (SSE) Humanizer Streaming Endpoint.
    Streams humanized sentences line-by-line in real-time as PyTorch T5 model generates them.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload text cannot be empty."
        )

    tone = payload.tone or "resume"
    engine = get_humanizer_engine()

    def event_generator():
        for chunk in engine.humanize_stream(text=text, tone=tone):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/closed-loop-humanize/stream")
async def closed_loop_humanize_stream_endpoint(payload: RobustnessEvalRequest):
    """
    Closed-Loop Dual-Agent SSE Streaming Endpoint (AccaHumanize-CL).
    Combines AI Detector Critic and Humanizer Generator with token locking and semantic verification.
    """
    from src.closed_loop_engine import ClosedLoopHumanizer

    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload text cannot be empty."
        )

    tone = payload.tone or "academic"
    closed_loop_engine = ClosedLoopHumanizer()

    def event_generator():
        for event_data in closed_loop_engine.process_closed_loop_stream(text=text, tone=tone):
            yield f"data: {json.dumps(event_data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


from fastapi.responses import Response

@router.post("/humanize-tex/stream")
async def humanize_tex_stream_endpoint(payload: RobustnessEvalRequest):
    """
    LaTeX Document-Level SSE Streaming Endpoint (AccaHumanize-Tex).
    Deconstructs .tex AST, locks math/tables/citations, and streams paragraph updates.
    """
    from src.closed_loop_engine import ClosedLoopHumanizer

    tex_content = payload.text.strip()
    if not tex_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LaTeX content cannot be empty."
        )

    tone = payload.tone or "academic"
    engine = ClosedLoopHumanizer()

    def event_generator():
        for event_data in engine.process_closed_loop_tex_stream(tex_content=tex_content, tone=tone):
            yield f"data: {json.dumps(event_data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/humanize-tex/download")
async def humanize_tex_download_endpoint(payload: RobustnessEvalRequest):
    """
    Downloadable LaTeX Export Endpoint.
    Returns clean, format-preserved .tex source file (or compiled PDF if pdflatex is installed).
    """
    from src.closed_loop_engine import ClosedLoopHumanizer
    from src.tex_compiler import LaTeXCompiler

    tex_content = payload.text.strip()
    if not tex_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LaTeX content cannot be empty."
        )

    tone = payload.tone or "academic"
    compiler = LaTeXCompiler()
    
    # If content already contains LaTeX document structure, compile directly without re-running closed loop
    final_tex = tex_content
    if "\\begin{document}" in tex_content and "\\end{document}" in tex_content and not ("\\begin{abstract}" in tex_content and "Furthermore, it is crucial to delve into" in tex_content):
        pass
    else:
        engine = ClosedLoopHumanizer()
        for event in engine.process_closed_loop_tex_stream(tex_content=tex_content, tone=tone):
            if event["type"] == "tex_complete":
                final_tex = event["final_tex"]

    pdf_bytes, log, is_compiled = compiler.compile_tex_to_pdf(final_tex)

    if is_compiled and pdf_bytes:
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=refined_academic_paper.pdf"}
        )
    else:
        return Response(
            content=final_tex,
            media_type="application/x-tex",
            headers={"Content-Disposition": "attachment; filename=refined_academic_paper.tex"}
        )



