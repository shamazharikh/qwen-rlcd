"""Supported Qwen3.5 backbones and a loader that returns just the text model + LM head.

All three are the same hybrid design (3 Gated DeltaNet layers per gated full-attention layer, vocab 248,320),
so `fork.py`, the templates and the `<|read|>` token work unchanged. They differ in size and MLP type:

| alias     | repo                          | layers | hidden | MLP                     | bf16 weights |
|-----------|-------------------------------|--------|--------|-------------------------|--------------|
| `0.8b`    | Qwen/Qwen3.5-0.8B-Base        | 24     | 1024   | dense, tied embeddings  | ~1.7 GB      |
| `9b`      | Qwen/Qwen3.5-9B-Base          | 32     | 4096   | dense, untied           | ~19 GB       |
| `35b-a3b` | Qwen/Qwen3.5-35B-A3B-Base     | 40     | 2048   | MoE, 256 experts, top-8 | ~72 GB (FP8 ~37 GB) |

The MoE routes each token independently (no capacity limits), so branch rows stay isolated and the fork
stays exact. Memory, not code, is the constraint: 9B needs bf16 on a 24 GB GPU. 35B-A3B defaults to FP8
weights quantized at load (`quantization="fp8"`, ~37 GB, 128x128 blocks like Qwen's own FP8 release), which
fits one 48 GB L40S. Use `quantization=None` with ~80 GB of GPU memory, or several GPUs via `device_map="auto"`.
Qwen's pre-quantized `Qwen/Qwen3.5-35B-A3B-FP8` also loads as a plain repo id, but it is the instruct model.
"""

from __future__ import annotations

import sys

import torch

MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B-Base",
    "9b": "Qwen/Qwen3.5-9B-Base",
    "35b-a3b": "Qwen/Qwen3.5-35B-A3B-Base",
}
# Default dtype per model: fp32 where it fits (keeps the 0.8B results exact and comparable), else bf16.
DEFAULT_DTYPES = {"Qwen/Qwen3.5-0.8B-Base": torch.float32}
# Load-time quantization per model when the caller doesn't choose one.
DEFAULT_QUANTIZATION = {"Qwen/Qwen3.5-35B-A3B-Base": "fp8"}
# Kept in bf16 under FP8, mirroring Qwen's FP8 checkpoint: small or precision-sensitive projections (DeltaNet
# a/b gates and conv, MoE router and shared-expert gate), embeddings, LM head, and the unused vision tower.
FP8_SKIP = ["lm_head", "embed_tokens", "model.visual", "conv1d", "in_proj_a", "in_proj_b", "mlp.gate", "shared_expert_gate"]
_MODELING_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
)


def resolve(model: str) -> str:
    """Alias (`0.8b`, `9b`, `35b-a3b`) or any HF repo id -> repo id."""
    return MODELS.get(model.lower(), model)


def default_dtype(repo: str, device) -> torch.dtype:
    if repo in DEFAULT_DTYPES or str(device) == "cpu":
        return DEFAULT_DTYPES.get(repo, torch.float32)
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def prepare_kernels(device, dtype) -> None:
    """Pick DeltaNet kernels before transformers binds them at import (see README M0 findings)."""
    from system_one.kernels import disable_fla_tf32

    if str(device) == "cpu":
        # transformers binds the DeltaNet kernel at import and picks fla (CUDA-only) whenever installed.
        if not any(m in sys.modules for m in _MODELING_MODULES):
            sys.modules.setdefault("fla", None)
    elif dtype == torch.float32:
        disable_fla_tf32()  # fp32 should mean fp32: TF32 in fla moves reads by ~2e-2 on Ampere+


def quantization_config(quantization: str | None):
    if quantization in (None, "none"):
        return None
    if quantization == "fp8":  # block-wise FP8 weights, dynamic FP8 activations; needs sm >= 89 (Ada / Hopper)
        from transformers import FineGrainedFP8Config

        return FineGrainedFP8Config(weight_block_size=(128, 128), modules_to_not_convert=FP8_SKIP)
    raise ValueError(f"unknown quantization {quantization!r} (use 'fp8' or None)")


def load_text_model(
    model: str = "0.8b", dtype: torch.dtype | None = None, device=None, device_map=None, quantization: str | None = "default"
):
    """Load a Qwen3.5 checkpoint and return `(text_model, lm_head, tokenizer)`.

    Weights stream straight to `device` (no full CPU copy, which matters for 9B+ in small pods), and the
    vision tower stays on CPU since the text path never uses it. Pass `device_map="auto"` to shard a model
    that doesn't fit on one GPU. `quantization`: "fp8", None, or "default" (per `DEFAULT_QUANTIZATION`).
    """
    repo = resolve(model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or default_dtype(repo, device)
    prepare_kernels(device, dtype)
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if quantization == "default":
        quantization = DEFAULT_QUANTIZATION.get(repo) if str(device) != "cpu" else None
    if device_map is None:
        # On-the-fly FP8 refuses CPU entries in a device_map, so then the (bf16, <1 GB) vision tower joins the GPU.
        visual = str(device) if quantization not in (None, "none") else "cpu"
        device_map = {"model.language_model": str(device), "lm_head": str(device), "model.visual": visual}
    full = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=dtype, device_map=device_map, quantization_config=quantization_config(quantization)
    ).eval()
    return full.model.language_model, full.lm_head, AutoTokenizer.from_pretrained(repo)
