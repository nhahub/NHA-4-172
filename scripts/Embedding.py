# from qdrant_client import QdrantClient
# client = QdrantClient(url="http://localhost:6333")
# client.delete_collection("depi_page_images")
# client.close()

# ============================================================
# Phase 2: ColQwen2.5 Embedding + Qdrant Multi-Vector Indexing
# Local device (RTX 3070 Laptop, 8GB VRAM) — all storage on D:
# ============================================================

import os

# MUST run before importing transformers / colpali_engine — redirects all
# HuggingFace downloads away from the constrained C: drive.
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from transformers import BitsAndBytesConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("embed_and_index")


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    batch_size: int = 8                  # proven-stable VRAM value for this hardware
    scan_batch_size: int = 500           # resume-check batch — decoupled from embed batch
    upload_batch_size: int = 12          # Qdrant upload batch — decoupled from embed batch
    vector_dim: int = 128                # ColQwen2.5 multi-vector dimensionality
    collection_name: str = "depi_page_images"
    cache_clear_interval: int = 100      # periodic defensive VRAM cleanup


PAGE_METADATA_FILE = Path("D:/Self Learning/DEPI/R4/DEPI Project/Data/extracted/Page Images/metadata/page_metadata.jsonl")
CFG = EmbeddingConfig()

# Point IDs must be reproducible so re-running the script skips already-indexed pages
ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def make_point_id(doc_id: str, page_number: int) -> str:
    """Deterministic UUID from (doc_id, page_number) — re-running never creates duplicates."""
    return str(uuid.uuid5(ID_NAMESPACE, f"{doc_id}_page_{page_number}"))


# ------------------------------------------------------------
# Model loading — includes the manual LoRA adapter remapping fix.
# ------------------------------------------------------------
from colqwen_loader import load_colqwen25, MODEL_NAME

def load_model_and_processor(cfg: EmbeddingConfig):
    return load_colqwen25(MODEL_NAME, logger=logger)


# ------------------------------------------------------------
# Qdrant connection — fresh client per call, always explicitly closed.
# gRPC (port 6334) chosen over REST for stability with large multi-vector
# payloads on Windows; timeout=90 gives background index maintenance room
# to breathe as the collection grows.
# ------------------------------------------------------------
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6334", prefer_grpc=True, timeout=90, check_compatibility=False)


def with_qdrant_client(fn):
    """Ensures every Qdrant call properly closes its connection afterward —
    prevents the connection-leak issue that caused WinError 10053 earlier."""
    def wrapper(*args, **kwargs):
        client = get_qdrant_client()
        try:
            return fn(client, *args, **kwargs)
        finally:
            client.close()
    return wrapper


def ensure_collection(client: QdrantClient, cfg: EmbeddingConfig) -> None:
    if client.collection_exists(cfg.collection_name):
        return

    logger.info(f"Creating multi-vector collection '{cfg.collection_name}' (int8 quantized, on-disk)")
    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config=qmodels.VectorParams(
            size=cfg.vector_dim,
            distance=qmodels.Distance.COSINE,
            multivector_config=qmodels.MultiVectorConfig(
                comparator=qmodels.MultiVectorComparator.MAX_SIM
            ),
            on_disk=True,
        ),
        quantization_config=qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            )
        ),
    )


@with_qdrant_client
def filter_unindexed(client: QdrantClient, cfg: EmbeddingConfig, candidate_ids: list[str]) -> set[str]:
    """Returns the subset of candidate_ids NOT already present in Qdrant — resume mechanism."""
    existing = client.retrieve(
        collection_name=cfg.collection_name,
        ids=candidate_ids,
        with_payload=False,
        with_vectors=False,
    )
    existing_ids = {str(point.id) for point in existing}
    return {pid for pid in candidate_ids if pid not in existing_ids}


def upsert_with_retry(cfg: EmbeddingConfig, points: list, max_retries: int = 5) -> None:
    """Fresh client per attempt, closed after every attempt — success or failure."""
    for attempt in range(1, max_retries + 1):
        client = get_qdrant_client()
        try:
            client.upsert(collection_name=cfg.collection_name, points=points)
            return
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Upsert failed after {max_retries} attempts: {e}")
                raise
            wait_time = min(2 ** attempt, 30)
            logger.warning(f"Upsert attempt {attempt} failed ({e}); retrying in {wait_time}s")
            time.sleep(wait_time)
        finally:
            client.close()


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------
def embed_image_batch(model, processor, images: list[Image.Image]) -> list[list[list[float]]]:
    """Returns one variable-length list of 128-dim vectors per image."""
    batch = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)  # shape: (batch, num_patches, 128)
    return [page_embedding.to(torch.float32).cpu().tolist() for page_embedding in embeddings]


def embed_with_oom_fallback(model, processor, records: list[dict]) -> list[list[list[float]]]:
    """Try the configured batch; on OOM (either PyTorch's clean exception or the
    raw async CUDA error PyTorch's async execution model can surface as a
    generic RuntimeError), retry one page at a time instead of failing the run."""
    images = [Image.open(r["image_path"]).convert("RGB") for r in records]

    try:
        return embed_image_batch(model, processor, images)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and "CUDA error" not in str(e):
            raise  # not an OOM — a real bug, let it surface normally

        logger.warning(f"OOM on batch of {len(images)} — retrying individually")
        torch.cuda.empty_cache()
        results = []
        for image in images:
            try:
                results.append(embed_image_batch(model, processor, [image])[0])
            except (torch.cuda.OutOfMemoryError, RuntimeError) as inner_e:
                if "out of memory" not in str(inner_e).lower() and "CUDA error" not in str(inner_e):
                    raise
                logger.error("OOM even at batch size 1 — skipping this page")
                torch.cuda.empty_cache()
                results.append(None)
        return results


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def load_page_records(metadata_file: Path) -> list[dict]:
    with open(metadata_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_embedding_pipeline(cfg: EmbeddingConfig) -> None:
    model, processor = load_model_and_processor(cfg)

    setup_client = get_qdrant_client()
    try:
        ensure_collection(setup_client, cfg)
    finally:
        setup_client.close()

    all_records = load_page_records(PAGE_METADATA_FILE)
    for record in all_records:
        record["point_id"] = make_point_id(record["doc_id"], record["page_number"])
    logger.info(f"Loaded {len(all_records)} page records")

    processed, skipped, failed = 0, 0, 0
    point_buffer = []
    embed_iteration = 0

    def flush_buffer():
        nonlocal point_buffer, processed
        if point_buffer:
            upsert_with_retry(cfg, point_buffer)
            processed += len(point_buffer)
            point_buffer = []

    # Outer loop: large scan batches, cheap existence checks (resume support).
    # Inner loop: small embed batches, VRAM-bound GPU work — only entered
    # when the outer check finds genuinely new work.
    scan_batches = list(batched(all_records, cfg.scan_batch_size))

    for scan_batch in tqdm(scan_batches, desc="Scanning + embedding"):
        candidate_ids = [r["point_id"] for r in scan_batch]
        remaining_ids = filter_unindexed(cfg, candidate_ids)

        pending_records = [r for r in scan_batch if r["point_id"] in remaining_ids]
        skipped += len(scan_batch) - len(pending_records)

        if not pending_records:
            continue

        for embed_batch in batched(pending_records, cfg.batch_size):
            embed_iteration += 1
            embeddings = embed_with_oom_fallback(model, processor, embed_batch)

            for record, embedding in zip(embed_batch, embeddings):
                if embedding is None:
                    failed += 1
                    continue
                point_buffer.append(
                    qmodels.PointStruct(
                        id=record["point_id"],
                        vector=embedding,
                        payload={
                            "doc_id": record["doc_id"],
                            "doc_type": record["doc_type"],
                            "page_number": record["page_number"],
                            "total_pages": record["total_pages"],
                            "image_path": record["image_path"],
                            "source_pdf": record["source_pdf"],
                        },
                    )
                )

            if len(point_buffer) >= cfg.upload_batch_size:
                flush_buffer()

            if embed_iteration % cfg.cache_clear_interval == 0:
                torch.cuda.empty_cache()

    flush_buffer()

    final_client = get_qdrant_client()
    try:
        logger.info(f"Done. Indexed: {processed} | Skipped: {skipped} | Failed: {failed}")
        logger.info(f"Collection point count: {final_client.count(cfg.collection_name).count}")
    finally:
        final_client.close()


# First open in terminal: ./qdrant.exe
# (PS D:\Self Learning\DEPI\R4\DEPI Project\Data\vector_store\Page Images\qdrant_server> ./qdrant.exe)

if __name__ == "__main__":
    run_embedding_pipeline(CFG)