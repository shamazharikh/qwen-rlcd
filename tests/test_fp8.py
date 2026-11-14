"""FP8 load path (used by default for 35B-A3B): a tiny random A3B-shaped checkpoint saved to disk, then loaded
through `load_text_model(..., quantization="fp8")`. Needs a GPU with FP8 support (sm >= 89: Ada / Hopper)."""

import random

import pytest
import torch

from system_one.fork import fork_forward, sequential_reads
from system_one.models import load_text_model

FP8_OK = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
pytestmark = pytest.mark.skipif(not FP8_OK, reason="needs a CUDA GPU with FP8 (sm >= 89)")
if FP8_OK:
    pytest.importorskip("kernels", reason="install the [fp8] extra for FP8 matmul kernels")


@pytest.fixture(scope="module")
def tiny_moe_checkpoint(tmp_path_factory):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(0)
    text = dict(
        vocab_size=248320, hidden_size=256, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=64, linear_value_head_dim=64, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, shared_expert_intermediate_size=128,
    )  # fmt: skip  (dims are multiples of 128 so FP8 128x128 blocks tile exactly)
    vision = dict(depth=1, hidden_size=128, num_heads=2, intermediate_size=128, out_hidden_size=256)
    model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=text, vision_config=vision))
    path = tmp_path_factory.mktemp("tiny_a3b")
    model.to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base").save_pretrained(path)
    return str(path)


def _reads(text_model):
    random.seed(0)
    state = torch.tensor([[random.randrange(1, 1000) for _ in range(40)]], device="cuda")
    branches = [[random.randrange(1, 1000) for _ in range(n)] for n in (5, 12, 1, 30)]
    with torch.no_grad():
        return fork_forward(text_model, state, branches, 0).float(), sequential_reads(text_model, state, branches).float()


def test_fp8_load_quantizes_experts_and_fork_matches(tiny_moe_checkpoint):
    text_model, lm_head, _ = load_text_model(tiny_moe_checkpoint, torch.bfloat16, "cuda", quantization="fp8")
    dtypes = {n: p.dtype for n, p in text_model.named_parameters()}
    fp8 = {n for n, d in dtypes.items() if d == torch.float8_e4m3fn}
    assert any("experts" in n for n in fp8), "MoE experts should be FP8"
    assert any("linear_attn.in_proj_qkvz" in n or "in_proj_qkv" in n for n in fp8), "DeltaNet main projection should be FP8"
    for skipped in ("embed_tokens", "mlp.gate.", "shared_expert_gate", "in_proj_a", "in_proj_b", "conv1d"):
        assert not any(skipped in n for n in fp8), f"{skipped} should stay bf16"
    assert lm_head.weight.dtype == torch.bfloat16

    forked, reference = _reads(text_model)
    cos = torch.nn.functional.cosine_similarity(forked, reference, dim=-1).min().item()
    assert cos > 0.999, cos  # FP8 doesn't break branch isolation; same tolerance class as bf16

    bf16_model, _, _ = load_text_model(tiny_moe_checkpoint, torch.bfloat16, "cuda", quantization=None)
    bf16_reads, _ = _reads(bf16_model)
    agree = torch.nn.functional.cosine_similarity(forked, bf16_reads, dim=-1).min().item()
    print(f"\ntiny A3B FP8: fork vs sequential cos {cos:.6f}, FP8 vs bf16 cos {agree:.6f}, {len(fp8)} FP8 tensors")
    # Loose sanity bound only: random tiny routers sit near ties, so FP8 error flips expert choice (measured 0.93).
    # Real weights agree far better (0.8B FP8 vs bf16 reads: 0.981); judge A3B FP8 by task accuracy, not this.
    assert agree > 0.8, agree
