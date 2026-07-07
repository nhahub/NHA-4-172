# ============================================================
# Shared ColQwen2.5 loader with the manual LoRA adapter fix.
# Imported by both Phase 2 (embedding) and Phase 3 (retrieval) to
# guarantee both use an identical embedding space — critical, since
# any drift between the two would silently reintroduce the exact
# bug this session spent hours diagnosing.
# ============================================================

import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import BitsAndBytesConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

MODEL_NAME = "vidore/colqwen2.5-v0.2"


def find_adapter_weights_file(model_path: Path) -> Path:
    candidates = list(model_path.glob("adapter_model.safetensors"))
    if not candidates:
        candidates = list(model_path.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No adapter weights file found in {model_path}")
    return candidates[0]


def remap_adapter_key(key: str) -> str:
    """
    Fixes two naming mismatches between the raw PEFT-saved adapter file and
    what the loaded model expects: (1) PEFT's triple-nested save prefix
    (base_model.model.model.layers) vs. the model's own naming, and
    (2) PEFT's stripped '.default' adapter-name segment.
    """
    key = re.sub(r"^base_model\.model\.model\.layers", "language_model.layers", key)
    key = re.sub(r"^model\.layers", "language_model.layers", key)
    key = re.sub(r"^base_model\.model\.custom_text_proj", "custom_text_proj", key)
    key = re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", key)
    return key


def load_colqwen25(model_name: str = MODEL_NAME, device: str = "cuda:0", logger=None):
    """Loads ColQwen2.5, 4-bit quantized, with the LoRA adapter correctly attached.
    Raises RuntimeError if any adapter key fails to attach — this must be loud,
    not silent, given the cost of discovering it late."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    model = ColQwen2_5.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device,
        attn_implementation=attn_impl,
    ).eval()

    snapshot_root = Path(os.environ["HF_HUB_CACHE"]) / f"models--{model_name.replace('/', '--')}" / "snapshots"
    hash_folder = next(f for f in snapshot_root.iterdir() if f.is_dir())

    adapter_weights_path = find_adapter_weights_file(hash_folder)
    raw_adapter_state_dict = load_safetensors(str(adapter_weights_path))
    remapped_state_dict = {remap_adapter_key(k): v for k, v in raw_adapter_state_dict.items()}

    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    failed_to_attach = set(remapped_state_dict.keys()) & set(unexpected)

    if failed_to_attach:
        raise RuntimeError(
            f"{len(failed_to_attach)} adapter keys failed to attach. "
            f"First few: {list(failed_to_attach)[:5]}"
        )

    if logger:
        logger.info(f"✅ All {len(remapped_state_dict)} adapter keys attached successfully (4-bit quantized).")

    # processor = ColQwen2_5_Processor.from_pretrained(model_name)
    processor = ColQwen2_5_Processor.from_pretrained(str(hash_folder))
    return model, processor