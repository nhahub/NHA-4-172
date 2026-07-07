# ============================================================
# Phase 4: Vision LLM Integration — RAG Query Answering
# Retrieval (ColQwen2.5, per-query swap) + Generation (Qwen2.5-VL-7B, resident)
# Strict grounding prompt + citation check + neighbor-window context expansion.
# ============================================================

import os
import gc
import re
import warnings
import logging
from dataclasses import dataclass, field
import time

# Production Paths, Offline Safeguards, & Output Quiet Flags
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# STOP FRAGMENTATION:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Mute warnings and suppress background verbose layout tables
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Explicit library silencing (Kills FastPlaid lines and LoRA missing/unexpected key alerts)
logging.getLogger().setLevel(logging.ERROR) 
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("colpali_engine").setLevel(logging.ERROR)
logging.getLogger("vidore_benchmark").setLevel(logging.ERROR)

logger = logging.getLogger("phase4_generation")

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from colqwen_loader import load_colqwen25, MODEL_NAME as COLQWEN_MODEL_NAME


# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------
STRICT_GROUNDING_SYSTEM_PROMPT = """You are an elite, document-grounded research intelligence. You are analyzing raw visual page images from a document collection. Every image is tagged at its boundary with its metadata (e.g., "Primary source: book B9, page 450").

STRICT OPERATIONAL DIRECTIVES:

1. KNOWLEDGE GROUNDING:
   - Provide a comprehensive, accurate answer using ONLY factual details clearly visible inside the provided page images.
   - Do NOT use or extrapolate from background historical knowledge if the details are missing from the pages.

2. DYNAMIC CITATION GENERATION:
   - At the absolute end of every successful answer, you must provide a clean inventory of every document page that actively contributed factual information to your text.
   - EXCLUSION FILTER: Do NOT cite index pages, cover sheets, or Tables of Contents that merely list chapter headings. Only cite pages containing substantive content that directly shaped your answer.
   - List every contributing page clearly, one per line. Use this exact syntax structure:
   
   📄 Reference:
        [Doc Type] [Doc ID] page [Page Number]

   (Example with 1 page 'if there was only one page contributed in answer':
   📄 Reference:
        book B9 page 450)
        
   (Example with multiple distinct pages 'if there were more pages contributed in answer':
   📄 Reference:
        book B9 page 450
        paper 2212.03551 page 2
        book B10 page 229)

3. IRRELEVANT / OUT-OF-DOMAIN PROTOCOL:
   - If the images do not contain any information relevant to the question, or if you cannot visually locate the facts required to build an answer, you must decline to answer.
   - In this scenario, output this exact phrase and absolutely nothing else:
     "The provided pages do not contain enough information to answer this question."
   - CRITICAL: If you output the refusal phrase above, do NOT print the string "Reference:" or list any citations whatsoever."""


# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    model_name: str = COLQWEN_MODEL_NAME
    collection_name: str = "depi_page_images"
    default_top_k: int = 3          # number of retrieved images
    neighbor_window: int = 1        # how many pages before/after to fetch as context
    max_total_images: int = 9       # hard cap on images sent to the vision LLM


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 512
    max_pixels_per_image: int = 28 * 28 * 128  # resoultion of image 'increased -> VRAM usgae increased'
    vram_ceiling_gb: float = 6.0
    system_prompt: str = STRICT_GROUNDING_SYSTEM_PROMPT


RETRIEVAL_CFG = RetrievalConfig()
GENERATION_CFG = GenerationConfig()
BYTES_PER_GB = 1024 ** 3


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
    """Aggressively purges transient allocations from standard memory and hardware runtime caches."""
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
        score_threshold=13.0,
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
    """Self-contained entrypoint: manages ColQwen2.5 memory lifespans explicitly per-call."""
    model, processor = load_colqwen25(cfg.model_name, logger=logger)
    client = get_qdrant_client()
    try:
        query_vector = encode_query(model, processor, query)
        points = search_pages(client, cfg, query_vector, top_k)
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
    """Enriches primary source vectors with consecutive adjacent target pages."""
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
        # Build an explicit tag that the model can copy directly for its references
        kind = "Primary source" if not s["is_context"] else "Surrounding context"
        source_tag = f"{kind}: {s['doc_type']} {s['doc_id']}, page {s['page_number']}"
        
        content.append({"type": "text", "text": source_tag})
        content.append({
            "type": "image", 
            "image": Image.open(s["image_path"]).convert("RGB"),
            "max_pixels": cfg.max_pixels_per_image # Ensure this uses the lower VRAM budget
        })
        
    content.append({"type": "text", "text": f"User Question: {query}"})
    
    return [
        {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]},
        {"role": "user", "content": content},
    ]
    

def generate_with_oom_fallback(model, processor, query: str, expanded_sources: list[dict],
                                cfg: GenerationConfig) -> str:
    """Safely constructs tokens over inputs. Automatically drops low-priority adjacent contexts on OOM."""
    candidates = sorted(expanded_sources, key=lambda s: s["is_context"])  # primaries first, drop from the end

    while True:
        inputs = generated_ids = trimmed = None
        try:
            messages = build_messages(query, candidates, cfg)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                # generated_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    temperature=0.0
                )

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


# ------------------------------------------------------------
# Complete Double-Check Citation Pipeline
# ------------------------------------------------------------
def check_citation_presence(answer: str, expanded_sources: list[dict]) -> dict:
    """
    Parses the model's stated (doc_id, page) references and confirms each one
    actually corresponds to a page genuinely shown to it. The prior version
    only checked "does 'reference:' appear alongside any digit anywhere" —
    that passes for almost any answer mentioning a year, formula, or page
    number, so it could never meaningfully fail. This version can actually
    catch a fabricated or misattributed citation.
    """
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
    sources: list[dict] = field(default_factory=list)       # ORIGINAL retrieval ranking — for eval/citation purity
    expanded_sources: list[dict] = field(default_factory=list)  # what was actually shown to the model
    citation_check: dict = field(default_factory=dict)


def answer_question(query: str, vision_model, vision_processor,
                     retrieval_cfg: RetrievalConfig = RETRIEVAL_CFG,
                     generation_cfg: GenerationConfig = GENERATION_CFG) -> RAGAnswer:
    """Main query handler. Uses text-only isolation boundaries between processing tasks."""
    print("\n[System] Processing retrieval model & fetching context...")
    
    sources = retrieve(query, retrieval_cfg.default_top_k, retrieval_cfg)

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
        
    # MID-STEP VRAM PURGE: Clean up retriever memory before LLM starts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    answer_text = generate_with_oom_fallback(vision_model, vision_processor, query, expanded, generation_cfg)
    
    citation_check = check_citation_presence(answer_text, expanded)
    if citation_check["issues"]:
        logger.warning(f"Grounding check flagged this answer: {citation_check['issues']}")

    return RAGAnswer(answer=answer_text, sources=sources, expanded_sources=expanded, citation_check=citation_check)


# ------------------------------------------------------------
# Interactive session
# ------------------------------------------------------------
def run_interactive_session():
    print("\n[System] Loading vision LLM (stays resident for this session)...")
    vision_model, vision_processor = load_vision_llm(GENERATION_CFG)

    print("\n" + "=" * 65)
    print("🤖 MULTIMODAL RAG — ASK QUESTIONS ABOUT YOUR DOCUMENT CORPUS")
    print(f"• Retrieval:  ColQwen2.5, top-{RETRIEVAL_CFG.default_top_k} + neighbor window ±{RETRIEVAL_CFG.neighbor_window}")
    print(f"• Generation: Qwen2.5-VL-7B (resident, VRAM ceiling {GENERATION_CFG.vram_ceiling_gb}GB)")
    print(f"• Grounding:  Strict Validation Engine Integrated")
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

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        
        print(f"\n🤖 Answer: {result.answer}")
        
        print(f"\n⏱️  Latency: {elapsed_seconds:.2f}s")
        
        # [Commented Out Block - Removed from active console logs]
        # if result.expanded_sources:
        #     print("📄 Shown to model:")
        #     for s in result.expanded_sources:
        #         tag = "primary" if not s["is_context"] else "context"
        #         score_str = f"score={s['score']:.3f}" if s["score"] is not None else "score=n/a (neighbor)"
        #         print(f"   [{tag}] {s['doc_type']} | {s['doc_id']} | page {s['page_number']} | {score_str}")
                
        if result.citation_check.get("issues"):
            print("⚠️  Grounding check:")
            for issue in result.citation_check["issues"]:
                print(f"   - {issue}")
        
        print()

        # ==========================================================
        # ⚡ HARD VRAM FLUSH & MEMORY PURGE ZONE
        # ==========================================================
        del result  # Deletes tensor reference and structural string data pointers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # ==========================================================

    unload_model(vision_model)


if __name__ == "__main__":
    run_interactive_session()