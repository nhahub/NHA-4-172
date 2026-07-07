# ============================================================
# Phase 2 Validation: Collection Health + Embedding Integrity
# + Semantic Regression Test (catches the adapter-loading bug class)
# ============================================================

import os

os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient

from colqwen_loader import load_colqwen25, MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate_phase2")

PAGE_METADATA_FILE = Path("D:/Self Learning/DEPI/R4/DEPI Project/Data/extracted/Page Images/metadata/page_metadata.jsonl")
COLLECTION_NAME = "depi_page_images"
VECTOR_DIM = 128
SAMPLE_SIZE = 30


def get_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6334", prefer_grpc=True, timeout=90, check_compatibility=False)


# ------------------------------------------------------------
# Check 1: Collection-level health
# ------------------------------------------------------------
def check_collection_health(client: QdrantClient) -> dict:
    info = client.get_collection(COLLECTION_NAME)

    print(f"\n{'='*60}\nCOLLECTION HEALTH\n{'='*60}")
    print(f"Status:                 {info.status}")
    print(f"Points count:           {info.points_count}")
    print(f"Indexed vectors count:  {info.indexed_vectors_count}")
    print(f"Segments count:         {info.segments_count}")

    if str(info.status).lower() == "grey":
        print("\n⚠️  GREY status — optimizations paused. Click 'Trigger Optimizers'")
        print("    in the dashboard before proceeding.")

    return {"status": str(info.status), "points_count": info.points_count}


# ------------------------------------------------------------
# Check 2: Per-document completeness
# ------------------------------------------------------------
def load_expected_pages() -> dict[str, int]:
    expected = {}
    with open(PAGE_METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            expected[record["doc_id"]] = record["total_pages"]
    return expected


def scan_indexed_pages(client: QdrantClient) -> dict[str, set[int]]:
    indexed = defaultdict(set)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=["doc_id", "page_number"],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            indexed[point.payload["doc_id"]].add(point.payload["page_number"])
        if offset is None:
            break
    return indexed


def check_document_completeness(client: QdrantClient) -> dict:
    print(f"\n{'='*60}\nPER-DOCUMENT COMPLETENESS\n{'='*60}")

    expected = load_expected_pages()
    indexed = scan_indexed_pages(client)

    incomplete_docs = []
    for doc_id, expected_total in expected.items():
        indexed_pages = indexed.get(doc_id, set())
        if len(indexed_pages) != expected_total:
            missing = sorted(set(range(1, expected_total + 1)) - indexed_pages)
            incomplete_docs.append({"doc_id": doc_id, "expected": expected_total,
                                     "indexed": len(indexed_pages), "missing_pages": missing})

    total_docs = len(expected)
    print(f"Total documents:  {total_docs}")
    print(f"Fully indexed:    {total_docs - len(incomplete_docs)}")
    print(f"Incomplete:       {len(incomplete_docs)}")

    if incomplete_docs:
        print("\n⚠️  Documents with missing pages:")
        for doc in incomplete_docs[:20]:
            print(f"   - {doc['doc_id']}: {doc['indexed']}/{doc['expected']}, missing {doc['missing_pages']}")
    else:
        print("\n✅ Every document has all expected pages indexed.")

    return {"total_docs": total_docs, "incomplete_docs": incomplete_docs}


# ------------------------------------------------------------
# Check 3: Vector integrity — structural sample check
# ------------------------------------------------------------
def check_vector_integrity(client: QdrantClient) -> dict:
    print(f"\n{'='*60}\nVECTOR INTEGRITY (sample of {SAMPLE_SIZE})\n{'='*60}")

    sample_points, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=SAMPLE_SIZE, with_payload=True, with_vectors=False
    )
    sample_ids = [p.id for p in sample_points]
    detailed_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=sample_ids, with_payload=True, with_vectors=True
    )

    passed, failures = 0, []
    for point in detailed_points:
        issues = []
        vectors = point.vector
        if not vectors:
            issues.append("empty vector list")
        else:
            if any(len(v) != VECTOR_DIM for v in vectors):
                issues.append("wrong dimensionality")
            if all(all(x == 0.0 for x in v) for v in vectors):
                issues.append("all-zero vectors")
            if any(any(x != x for x in v) for v in vectors):
                issues.append("contains NaN")

        image_path = point.payload.get("image_path")
        if not image_path or not Path(image_path).exists():
            issues.append(f"image missing on disk: {image_path}")

        required_keys = {"doc_id", "doc_type", "page_number", "total_pages", "image_path", "source_pdf"}
        if missing_keys := required_keys - set(point.payload.keys()):
            issues.append(f"missing payload keys: {missing_keys}")

        if issues:
            failures.append({"id": point.id, "doc_id": point.payload.get("doc_id"), "issues": issues})
        else:
            passed += 1

    print(f"Passed:  {passed}/{len(detailed_points)}")
    print(f"Failed:  {len(failures)}/{len(detailed_points)}")
    if failures:
        print("\n⚠️  Issues found:")
        for f in failures:
            print(f"   - {f['doc_id']} (id={f['id']}): {', '.join(f['issues'])}")
    else:
        print("\n✅ All sampled points have valid vectors and complete payloads.")

    return {"passed": passed, "failed": len(failures)}


# ------------------------------------------------------------
# Check 4: SEMANTIC REGRESSION TEST — the check that was missing
# before, and would have caught the LoRA adapter bug immediately.
#
# Self-contained: generates two synthetic page images with clearly
# distinct written content, embeds them with the SAME loader used
# for the real corpus, and confirms each matching query scores
# meaningfully higher on its own image than on the other. This is
# a permanent canary against the exact failure mode this project
# already lost significant time to.
# ------------------------------------------------------------
def make_synthetic_probe_image(text: str) -> Image.Image:
    image = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(image)
    # Wrap crude but sufficient for OCR/semantic probing purposes
    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > 45:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 30), line, fill="black")
    return image


def check_semantic_discrimination(model, processor) -> dict:
    print(f"\n{'='*60}\nSEMANTIC REGRESSION TEST\n{'='*60}")

    probe_texts = [
        "The transformer attention mechanism computes queries keys and values "
        "to produce weighted contextual representations of each token.",
        "A variational autoencoder learns a probabilistic latent space using "
        "an encoder network and a decoder network trained together.",
    ]
    queries = [
        "What is the attention mechanism in transformers?",
        "What is a variational autoencoder?",
    ]

    images = [make_synthetic_probe_image(t) for t in probe_texts]

    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    print(scores)

    # Pass condition: each query's own-topic score exceeds its off-topic
    # score by a real margin — not just technically higher, since a tiny
    # margin is itself close to the broken-adapter symptom we saw earlier.
    diagonal_ok = scores[0][0] > scores[0][1] + 2.0 and scores[1][1] > scores[1][0] + 2.0

    if diagonal_ok:
        print("\n✅ Clear diagonal discrimination — adapter is loading correctly.")
    else:
        print("\n🚨 FAILED — scores do not show clear topic discrimination.")
        print("   This is the exact signature of the LoRA adapter silently failing")
        print("   to load. Do NOT trust this embedding model for production use.")
        print("   Re-verify colqwen_loader.py before proceeding to Phase 3.")

    return {"passed": diagonal_ok, "scores": scores.tolist()}


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    client = get_client()
    try:
        health = check_collection_health(client)
        completeness = check_document_completeness(client)
        integrity = check_vector_integrity(client)
    finally:
        client.close()

    model, processor = load_colqwen25(MODEL_NAME, logger=logger)
    semantic = check_semantic_discrimination(model, processor)
    del model
    torch.cuda.empty_cache()

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    all_good = (
        completeness["incomplete_docs"] == []
        and integrity["failed"] == 0
        and semantic["passed"]
    )
    print("🏆 Phase 2 validation passed — safe to proceed to Phase 3." if all_good
          else "⚠️  Issues found above — resolve before proceeding to Phase 3.")