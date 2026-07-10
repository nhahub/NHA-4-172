# ============================================================
# Phase 4: Vision LLM Integration — RAG Query Answering
# Retrieval (ColQwen2.5, per-query swap) + Generation (Qwen2.5-VL-7B, resident)
# Strict grounding prompt + citation check + neighbor-window context expansion
# + cooperative cancellation (stop mid-retrieval or mid-generation).
# ============================================================

import os
import gc
import re
import warnings
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Production Paths, Offline Safeguards, & Output Quiet Flags
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for _noisy_logger in ["transformers", "peft", "colpali_engine", "vidore_benchmark"]:
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

logger = logging.getLogger("phase4_generation")

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
)
from qwen_vl_utils import process_vision_info
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from colqwen_loader import load_colqwen25, MODEL_NAME as COLQWEN_MODEL_NAME


# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------
STRICT_GROUNDING_SYSTEM_PROMPT = """You are a strict document-grounded assistant. You will be shown several page images from a private document collection, followed by a user question.

STEP 1 — RELEVANCE CHECK (perform silently before writing anything):
For each page image shown, judge whether it contains content that explains, defines, or provides data relevant to the concepts in the user's question. A page that only contains a related keyword, a passing mention, or a table-of-contents entry listing a topic does NOT count as directly relevant — only pages with actual explanatory content count.
Do not require the question's exact wording to match the page's wording — judge based on meaning, not literal text matching (e.g. a hyphenated vs. unhyphenated term is the same concept). A question asking you to compare, contrast, or explain how something works can be answered by combining relevant explanatory content found separately across multiple pages — you do not need a single page that explicitly frames the answer as a comparison or full mechanism explanation.

STEP 2 — ANSWER OR REFUSE:
- If at least one page contains real explanatory content addressing the question's concepts — even partially, even if you must combine content from more than one page — write your answer using ONLY the facts, definitions, figures, and data visible on those specific pages.
- Refuse ONLY if the shown pages contain no substantive content related to the question's concepts at all. Do not refuse merely because answering requires synthesizing, comparing, or connecting information that appears in separate places across the pages — that is expected and required of you, not a reason to refuse.
- If you refuse, output exactly this sentence and nothing else:
"I am sorry, but the provided pages do not contain enough information to answer this question."
- Never use your own background/pretrained knowledge to fill a gap the pages don't cover.

STEP 3 — CITATIONS (mandatory whenever you answer; skip entirely only if you refused):
Any time you provide a substantive answer, you MUST end your response with a Reference section — never omit this. List ONLY the pages whose specific content you genuinely used to construct your answer: a fact, definition, number, or explanation drawn from that exact page — expressed in your own words — must actually appear in your answer. Do NOT cite a page just because it was shown to you, or because it contains a related keyword or a table-of-contents mention.

Use this exact format:

📄 Reference:
[Doc Type] [Doc ID] page [Page Number]

(Example:
📄 Reference:
book B1 page 12)

CRITICAL: If you output the refusal sentence from Step 2, do not print "Reference:" or any citation line at all."""


# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    model_name: str = COLQWEN_MODEL_NAME
    collection_name: str = "depi_page_images"
    default_top_k: int = 5          # number of retrieved images
    neighbor_window: int = 1        # how many pages before/after to fetch as context
    max_total_images: int = 11       # hard cap on images sent to the vision LLM


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 512
    max_pixels_per_image: int = 28 * 28 * 128
    system_prompt: str = STRICT_GROUNDING_SYSTEM_PROMPT


RETRIEVAL_CFG = RetrievalConfig()
GENERATION_CFG = GenerationConfig()
BYTES_PER_GB = 1024 ** 3

DATA_ROOT = Path("D:/Self Learning/DEPI/R4/DEPI Project/Data")


def resolve_image_path(image_path: str) -> Path:
    """Resolves a (possibly relative) stored image path against DATA_ROOT,
    regardless of process working directory, and refuses to resolve outside it."""
    path = Path(image_path)
    resolved = (path if path.is_absolute() else DATA_ROOT / path).resolve()
    root_resolved = DATA_ROOT.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Resolved image path escapes DATA_ROOT: {resolved}")
    return resolved


# ------------------------------------------------------------
# Cancellation
# ------------------------------------------------------------
class GenerationCancelled(Exception):
    """Raised when a cancel_event is set at any checkpoint in the pipeline —
    caught by the API layer and turned into a clean 'cancelled' response."""


class CancelStoppingCriteria(StoppingCriteria):
    """Checked by transformers between every generated token. Returning True
    stops generation early — this is what makes mid-generation cancellation
    actually work, not just cancellation between pipeline steps."""

    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


# ------------------------------------------------------------
# Hardware Memory Isolation Utilities
# ------------------------------------------------------------
def get_vram_usage_gb() -> float:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (total_bytes - free_bytes) / BYTES_PER_GB
    except Exception:
        return 0.0


def purge_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_model(model) -> None:
    del model
    purge_memory()
    logger.info(f"Model unloaded. VRAM now: {get_vram_usage_gb():.2f} GB")


# ------------------------------------------------------------
# Retrieval Pipeline
# ------------------------------------------------------------
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6334", prefer_grpc=True, timeout=90, check_compatibility=False)


def normalize_query_for_embedding(query: str) -> str:
    """Reduces ColQwen2.5's sensitivity to superficial hyphenation differences
    — confirmed via logs: 'Multi-modal RAG' vs 'Multimodal RAG' scored 14.23
    vs 15.79 on the identical top page, purely from hyphen tokenization.
    Only strips hyphens between letters. Known limitation: this also merges
    unrelated compound terms (e.g. "state-of-the-art" -> "stateoftheart") —
    an acceptable tradeoff for this corpus, worth revisiting if it causes
    new problems."""
    return re.sub(r'(?<=[A-Za-z])-(?=[A-Za-z])', '', query)


def encode_query(model, processor, query: str) -> list[list[float]]:
    batch = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)
    return embeddings[0].to(torch.float32).cpu().tolist()


def search_pages(client: QdrantClient, cfg: RetrievalConfig, query_vector: list[list[float]], top_k: int):
    results = client.query_points(
        collection_name=cfg.collection_name,
        query=query_vector,
        limit=top_k,
        # score_threshold=10.0,
        with_payload=True,
        search_params=qmodels.SearchParams(quantization=qmodels.QuantizationSearchParams(rescore=True)),
    )
    return results.points


def format_results(points) -> list[dict]:
    return [
        {"score": p.score, "doc_id": p.payload["doc_id"], "doc_type": p.payload["doc_type"],
         "page_number": p.payload["page_number"], "image_path": p.payload["image_path"]}
        for p in points
    ]


def retrieve(query: str, top_k: int, cfg: RetrievalConfig = RETRIEVAL_CFG) -> list[dict]:
    model, processor = load_colqwen25(cfg.model_name, logger=logger)
    client = get_qdrant_client()
    try:
        query_vector = encode_query(model, processor, normalize_query_for_embedding(query))
        points = search_pages(client, cfg, query_vector, top_k)
        results = format_results(points)
        
        logger.info("Retrieved: " + ", ".join(
            f"{r['doc_type']} {r['doc_id']} p{r['page_number']} ({r['score']:.2f})" for r in results
        ))
        
        return format_results(points)
    finally:
        client.close()
        unload_model(model)


# ------------------------------------------------------------
# Context Window Expansion Engine
# ------------------------------------------------------------
def fetch_neighbor_pages(client: QdrantClient, collection_name: str,
                          doc_id: str, center_page: int, window: int) -> list[dict]:
    page_range = list(range(max(1, center_page - window), center_page + window + 1))

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                qmodels.FieldCondition(key="page_number", match=qmodels.MatchAny(any=page_range)),
            ]
        ),
        limit=len(page_range),
        with_payload=True,
        with_vectors=False,
    )
    return [
        {"doc_id": p.payload["doc_id"], "doc_type": p.payload["doc_type"],
         "page_number": p.payload["page_number"], "image_path": p.payload["image_path"]}
        for p in points
    ]


def expand_with_neighbor_context(sources: list[dict], client: QdrantClient,
                                  collection_name: str, window: int, max_total_images: int) -> list[dict]:
    combined: dict[tuple, dict] = {}

    for source in sources:
        combined[(source["doc_id"], source["page_number"])] = {**source, "is_context": False}

    for source in sources:
        neighbors = fetch_neighbor_pages(client, collection_name, source["doc_id"], source["page_number"], window)
        for n in neighbors:
            key = (n["doc_id"], n["page_number"])
            if key not in combined:
                combined[key] = {**n, "score": None, "is_context": True}

    ordered = sorted(combined.values(), key=lambda s: (s["doc_id"], s["page_number"]))
    primary = [s for s in ordered if not s["is_context"]]
    context_only = [s for s in ordered if s["is_context"]]

    final = primary + context_only[: max(0, max_total_images - len(primary))]
    final.sort(key=lambda s: (s["doc_id"], s["page_number"]))
    return final


# ------------------------------------------------------------
# Text & Generation Core
# ------------------------------------------------------------
def load_vision_llm(cfg: GenerationConfig):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logger.info(f"Loading {cfg.model_name} (resident for session) | attn=sdpa")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name, quantization_config=quant_config, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(cfg.model_name, local_files_only=True)
    logger.info(f"Vision LLM loaded. VRAM: {get_vram_usage_gb():.2f} GB")
    return model, processor


def build_messages(query: str, expanded_sources: list[dict], cfg: GenerationConfig) -> list[dict]:
    content = []
    for s in expanded_sources:
        kind = "Primary source" if not s["is_context"] else "Surrounding context"
        source_tag = f"{kind}: {s['doc_type']} {s['doc_id']}, page {s['page_number']}"
        content.append({"type": "text", "text": source_tag})
        content.append({
            "type": "image",
            "image": Image.open(resolve_image_path(s["image_path"])).convert("RGB"),
            "max_pixels": cfg.max_pixels_per_image,
        })
    content.append({"type": "text", "text": f"User Question: {query}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]},
        {"role": "user", "content": content},
    ]


def generate_with_oom_fallback(model, processor, query: str, expanded_sources: list[dict],
                                cfg: GenerationConfig, cancel_event: threading.Event | None = None) -> str:
    """On OOM, drops the lowest-priority image and retries. On cancellation
    (checked before each attempt, and mid-generation via StoppingCriteria),
    raises GenerationCancelled instead of returning a partial/garbage answer."""
    candidates = sorted(expanded_sources, key=lambda s: s["is_context"])

    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled()

        inputs = generated_ids = trimmed = None
        try:
            messages = build_messages(query, candidates, cfg)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            stopping_criteria = None
            if cancel_event is not None:
                stopping_criteria = StoppingCriteriaList([CancelStoppingCriteria(cancel_event)])

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )

            if cancel_event is not None and cancel_event.is_set():
                # generate() stopped early due to the cancel signal — the
                # partial output is not a real answer, don't decode/return it.
                raise GenerationCancelled()

            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        except torch.cuda.OutOfMemoryError:
            purge_memory()
            if len(candidates) <= 1:
                raise RuntimeError("OOM even with a single page image — check VRAM availability.")
            logger.warning(f"OOM with {len(candidates)} images — dropping lowest-priority image")
            candidates = candidates[:-1]

        finally:
            del inputs, generated_ids, trimmed
            purge_memory()


def generate_streaming(model, processor, query: str, expanded_sources: list[dict],
                        cfg: GenerationConfig, cancel_event: threading.Event):
    """
    Returns (token_generator, result). Iterate token_generator fully to drive
    real generation; result is populated as a side effect once exhausted —
    check result["error"] / result["cancelled"], use result["full_text"] for
    citation parsing. Runs model.generate() on a background thread (required
    by TextIteratorStreamer) so the caller can consume tokens as produced.
    CancelStoppingCriteria is checked between every token — this is what
    makes Stop interrupt REAL compute mid-generation, not just an animation.

    Tradeoff vs. generate_with_oom_fallback: once tokens are already
    streaming to the user, "drop an image and retry" isn't coherent anymore.
    On OOM here, result["error"] is set and the caller reports a clean error.
    """
    result = {"full_text": "", "error": None, "cancelled": False}

    if cancel_event.is_set():
        result["cancelled"] = True
        def _empty_gen():
            return
            yield  # pragma: no cover
        return _empty_gen(), result

    ordered_sources = sorted(expanded_sources, key=lambda s: s["is_context"])
    messages = build_messages(query, ordered_sources, cfg)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping_criteria = StoppingCriteriaList([CancelStoppingCriteria(cancel_event)])

    generation_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=cfg.max_new_tokens,
        do_sample=False, stopping_criteria=stopping_criteria,
    )

    def _run_generate():
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as e:
            result["error"] = str(e)

    gen_thread = threading.Thread(target=_run_generate, daemon=True)
    gen_thread.start()

    def _consume():
        try:
            for chunk in streamer:
                result["full_text"] += chunk
                yield chunk
                if cancel_event.is_set():
                    break
        finally:
            gen_thread.join(timeout=30)
            purge_memory()
            if cancel_event.is_set():
                result["cancelled"] = True

    return _consume(), result


# ------------------------------------------------------------
# Citation Verification
# ------------------------------------------------------------
def check_citation_presence(answer: str, expanded_sources: list[dict]) -> dict:
    valid_keys = {(s["doc_id"], s["page_number"]) for s in expanded_sources}
    cited_raw = re.findall(r"([\w\.]+)\s+page\s+(\d+)", answer, re.IGNORECASE)
    cited = {(doc_id, int(page)) for doc_id, page in cited_raw}

    refuses_to_answer = "do not contain enough information" in answer.lower()
    fabricated = cited - valid_keys

    issues = []
    if not cited and not refuses_to_answer:
        issues.append("No parseable references found, and answer doesn't refuse.")
    if fabricated:
        issues.append(f"Cited page(s) not actually shown to the model: {fabricated}")

    return {
        "cited": cited,
        "fabricated": fabricated,
        "refuses_to_answer": refuses_to_answer,
        "issues": issues,
    }


# ------------------------------------------------------------
# Unified Orchestration Entrypoint
# ------------------------------------------------------------
@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    expanded_sources: list[dict] = field(default_factory=list)
    citation_check: dict = field(default_factory=dict)


def answer_question(query: str, vision_model, vision_processor,
                     retrieval_cfg: RetrievalConfig = RETRIEVAL_CFG,
                     generation_cfg: GenerationConfig = GENERATION_CFG,
                     cancel_event: threading.Event | None = None) -> RAGAnswer:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled()

    logger.info(f"Retrieving top-{retrieval_cfg.default_top_k} pages for: {query}")
    sources = retrieve(query, retrieval_cfg.default_top_k, retrieval_cfg)

    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled()

    if not sources:
        return RAGAnswer(answer="No relevant pages found in the corpus for this question.", sources=[])

    client = get_qdrant_client()
    try:
        expanded = expand_with_neighbor_context(
            sources, client, retrieval_cfg.collection_name,
            window=retrieval_cfg.neighbor_window, max_total_images=retrieval_cfg.max_total_images,
        )
    finally:
        client.close()

    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled()

    purge_memory()

    answer_text = generate_with_oom_fallback(
        vision_model, vision_processor, query, expanded, generation_cfg, cancel_event=cancel_event
    )

    citation_check = check_citation_presence(answer_text, expanded)
    if citation_check["issues"]:
        logger.warning(f"Grounding check flagged this answer: {citation_check['issues']}")

    return RAGAnswer(answer=answer_text, sources=sources, expanded_sources=expanded, citation_check=citation_check)


# ------------------------------------------------------------
# Interactive session (CLI) — no cancellation plumbing needed here;
# Ctrl+C already interrupts a CLI session naturally.
# ------------------------------------------------------------
def run_interactive_session():
    import time

    print("\n[System] Loading vision LLM (stays resident for this session)...")
    vision_model, vision_processor = load_vision_llm(GENERATION_CFG)

    print("\n" + "=" * 65)
    print("🤖 MULTIMODAL RAG — ASK QUESTIONS ABOUT YOUR DOCUMENT CORPUS")
    print(f"• Retrieval:  ColQwen2.5, top-{RETRIEVAL_CFG.default_top_k} + neighbor window ±{RETRIEVAL_CFG.neighbor_window}")
    print(f"• Generation: Qwen2.5-VL-7B (resident)")
    print("• Type 'exit' or 'quit' to end the session.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[System] Interrupted. Shutting down.")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("\n[System] Session ended.")
            break

        start_time = time.perf_counter()
        try:
            result = answer_question(query, vision_model, vision_processor)
        except Exception as e:
            print(f"\n❌ Error answering question: {e}\n")
            continue
        elapsed_seconds = time.perf_counter() - start_time

        print(f"\n🤖 Answer: {result.answer}")
        print(f"\n⏱️  Latency: {elapsed_seconds:.2f}s")
        if result.citation_check.get("issues"):
            print("⚠️  Grounding check:")
            for issue in result.citation_check["issues"]:
                print(f"   - {issue}")
        print()

        del result
        purge_memory()

    unload_model(vision_model)


if __name__ == "__main__":
    run_interactive_session()