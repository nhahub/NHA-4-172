import base64
import io
import json
import logging
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from qdrant_client.http import models as qmodels

sys.path.append(str(Path(__file__).parent.parent))
from Multimodal_RAG_System import (
    load_vision_llm, unload_model, get_vram_usage_gb,
    get_qdrant_client, resolve_image_path, retrieve, expand_with_neighbor_context,
    check_citation_presence, generate_streaming, purge_memory,
    normalize_query_for_embedding,   # <-- add this
    RETRIEVAL_CFG, GENERATION_CFG, RAGAnswer, GenerationCancelled,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("genai_api")

THUMBNAIL_MAX_SIDE = 480
THUMBNAIL_JPEG_QUALITY = 82

INTERACTION_LOG_PATH = Path(__file__).parent / "interaction_log.jsonl"

# Single-user local deployment: only one generation can be in flight at a
# time (enforced by generation_lock), so a single shared cancel_event is
# sufficient — no need for per-request tracking.
generation_lock = threading.Lock()
cancel_event = threading.Event()


# ------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading vision LLM (resident for the server's lifetime)...")
    model, processor = load_vision_llm(GENERATION_CFG)
    app.state.vision_model = model
    app.state.vision_processor = processor
    logger.info(f"Ready. VRAM: {get_vram_usage_gb():.2f} GB")
    yield
    logger.info("Shutdown: unloading vision LLM...")
    unload_model(app.state.vision_model)


app = FastAPI(title="GenAI API", version="1.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    doc_id: str
    doc_type: str
    page_number: int
    thumbnail: str  # base64 data URI — small, fast-loading card preview


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refuses: bool
    cancelled: bool = False
    grounding_issues: list[str]
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    # vram_gb: float
    qdrant_connected: bool


# ------------------------------------------------------------
# Citation refinement
# ------------------------------------------------------------
def encode_thumbnail(image_path: str) -> str:
    image = Image.open(resolve_image_path(image_path)).convert("RGB")
    image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_chat_response(rag_answer: RAGAnswer, elapsed_seconds: float) -> ChatResponse:
    # Match structural variations: "📄 Reference:", "Reference:", or "Source pages used:" (case-insensitive)
    split_pattern = r"(?:📄\s*Reference:|Reference:|Source\s+pages\s+used:)"
    
    # Split the string safely regardless of which phrase the local model selected
    parts = re.split(split_pattern, rag_answer.answer, maxsplit=1, flags=re.IGNORECASE)
    clean_answer = parts[0].strip()

    cited_pairs = rag_answer.citation_check.get("cited", set())
    source_lookup = {(s["doc_id"], s["page_number"]): s for s in rag_answer.expanded_sources}

    citations = []
    for doc_id, page_number in cited_pairs:
        source = source_lookup.get((doc_id, page_number))
        if source is None:
            continue
        citations.append(Citation(
            doc_id=source["doc_id"],
            doc_type=source["doc_type"],
            page_number=source["page_number"],
            thumbnail=encode_thumbnail(source["image_path"]),
        ))
    citations.sort(key=lambda c: (c.doc_id, c.page_number))

    # Identify if the message represents an explicit validation refusal
    is_refusal = rag_answer.citation_check.get("refuses_to_answer", False)
    
    # Suppress downstream warning logs for legitimate empty citation states
    grounding_issues = rag_answer.citation_check.get("issues", [])
    if is_refusal or (not citations and not grounding_issues):
        grounding_issues = []
    elif not citations and grounding_issues:
        if any("No parseable references found" in issue for issue in grounding_issues):
            grounding_issues = []

    return ChatResponse(
        answer=clean_answer,
        citations=citations,
        refuses=is_refusal,
        cancelled=False,
        grounding_issues=grounding_issues,
        latency_seconds=round(elapsed_seconds, 2),
    )


def log_interaction(query: str, response: ChatResponse) -> None:
    """Every real user interaction is logged locally — cheap to capture now,
    useful later for expanding the Phase 6 evaluation set or mining real
    failure patterns instead of relying only on synthetic test queries."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": response.answer,
        "num_citations": len(response.citations),
        "refuses": response.refuses,
        "cancelled": response.cancelled,
        "grounding_issues": response.grounding_issues,
        "latency_seconds": response.latency_seconds,
    }
    try:
        with open(INTERACTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write interaction log (non-fatal)")


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_connected = True
    try:
        client = get_qdrant_client()
        client.get_collections()
        client.close()
    except Exception:
        qdrant_connected = False

    return HealthResponse(
        status="ok" if qdrant_connected else "degraded",
        # vram_gb=round(get_vram_usage_gb(), 2),
        qdrant_connected=qdrant_connected,
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    cancel_event.clear()

    acquired = generation_lock.acquire(timeout=1.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server is busy with another request. Please wait and try again.")

    def stream_generator():
        start = time.perf_counter()
        
        normalized_query = normalize_query_for_embedding(request.message)
        
        try:
            yield f"data: {json.dumps({'event': 'status', 'text': 'Searching the corpus...'})}\n\n"

            sources = retrieve(normalized_query, RETRIEVAL_CFG.default_top_k, RETRIEVAL_CFG)

            if cancel_event.is_set():
                raise GenerationCancelled()

            if not sources:
                elapsed = time.perf_counter() - start
                response = ChatResponse(
                    answer="No relevant pages found in the corpus for this question.",
                    citations=[], refuses=False, cancelled=False,
                    grounding_issues=[], latency_seconds=round(elapsed, 2),
                )
                log_interaction(request.message, response)
                yield f"data: {json.dumps({'event': 'final', 'data': response.model_dump()})}\n\n"
                return

            client = get_qdrant_client()
            try:
                expanded = expand_with_neighbor_context(
                    sources, client, RETRIEVAL_CFG.collection_name,
                    window=RETRIEVAL_CFG.neighbor_window, max_total_images=RETRIEVAL_CFG.max_total_images,
                )
            finally:
                client.close()

            if cancel_event.is_set():
                raise GenerationCancelled()

            purge_memory()
            yield f"data: {json.dumps({'event': 'status', 'text': 'Reading the pages...'})}\n\n"

            token_gen, result = generate_streaming(
                app.state.vision_model, app.state.vision_processor,
                normalized_query, expanded, GENERATION_CFG, cancel_event,
            )

            # Stream real tokens as produced. Withhold the "📄 Reference:"
            # footer from what the client sees while streaming — citations
            # are computed from the FULL text only after generation ends.
            emitted_len = 0
            full_text_buffer = ""
            marker_found = False

            for chunk in token_gen:
                full_text_buffer += chunk
                if marker_found:
                    continue
                marker_idx = full_text_buffer.find("📄")
                visible_end = marker_idx if marker_idx != -1 else len(full_text_buffer)
                if visible_end > emitted_len:
                    emitted_len = visible_end
                    yield f"data: {json.dumps({'event': 'token', 'text': full_text_buffer[:emitted_len]})}\n\n"
                if marker_idx != -1:
                    marker_found = True

            elapsed = time.perf_counter() - start

            if result["cancelled"]:
                response = ChatResponse(
                    answer="Generation stopped.", citations=[], refuses=False,
                    cancelled=True, grounding_issues=[], latency_seconds=round(elapsed, 2),
                )
                log_interaction(request.message, response)
                yield f"data: {json.dumps({'event': 'cancelled', 'data': response.model_dump()})}\n\n"
                return

            if result["error"]:
                logger.error(f"Generation error: {result['error']}")
                yield f"data: {json.dumps({'event': 'error', 'detail': 'Failed to generate an answer.'})}\n\n"
                return

            citation_check = check_citation_presence(result["full_text"], expanded)
            if citation_check["issues"]:
                logger.warning(f"Grounding check flagged this answer: {citation_check['issues']}")

            rag_answer = RAGAnswer(
                answer=result["full_text"], sources=sources,
                expanded_sources=expanded, citation_check=citation_check,
            )
            response = build_chat_response(rag_answer, elapsed)
            log_interaction(request.message, response)
            yield f"data: {json.dumps({'event': 'final', 'data': response.model_dump()})}\n\n"

        except GenerationCancelled:
            elapsed = time.perf_counter() - start
            response = ChatResponse(
                answer="Generation stopped.", citations=[], refuses=False,
                cancelled=True, grounding_issues=[], latency_seconds=round(elapsed, 2),
            )
            log_interaction(request.message, response)
            yield f"data: {json.dumps({'event': 'cancelled', 'data': response.model_dump()})}\n\n"
        except Exception:
            logger.exception("Error answering question")
            yield f"data: {json.dumps({'event': 'error', 'detail': 'Failed to generate an answer.'})}\n\n"
        finally:
            generation_lock.release()

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/api/chat/stop")
def stop_generation() -> dict:
    """Sets the shared cancel signal. The in-flight /api/chat call (if any)
    detects this at its next checkpoint — including mid-generation — and
    returns a clean cancelled=True response rather than being forcibly killed."""
    cancel_event.set()
    logger.info("Cancellation requested by client.")
    return {"status": "cancel_requested"}


@app.get("/api/citation-image")
def citation_image(doc_id: str, doc_type: str, page_number: int):
    """Serves the full-resolution original page image for lightbox zoom.
    Deliberately looks the page up by (doc_id, page_number) against Qdrant
    rather than trusting any client-supplied file path directly — avoids
    ever accepting an arbitrary path from the browser."""
    client = get_qdrant_client()
    try:
        points, _ = client.scroll(
            collection_name=RETRIEVAL_CFG.collection_name,
            scroll_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type)),
                qmodels.FieldCondition(key="page_number", match=qmodels.MatchValue(value=page_number)),
            ]),
            limit=1,
            with_payload=True,
        )
    finally:
        client.close()

    if not points:
        raise HTTPException(status_code=404, detail="Page not found.")

    try:
        image_path = resolve_image_path(points[0].payload["image_path"])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image path.")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk.")

    return FileResponse(image_path, media_type="image/png")


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)