"""Kernel precision controls for the fla (flash-linear-attention) DeltaNet path."""
import os


def disable_fla_tf32() -> bool:
    """Make fla's Triton kernels compute fp32 dots in IEEE fp32 instead of TF32.

    On Ampere+ (sm >= 8) Triton defaults fp32 `tl.dot` to TF32, and fla 0.6 also hard-codes
    `input_precision='tf32'` in the gated-delta-rule `solve_tril` kernel. Together they put fp32
    fork-vs-sequential reads ~2e-2 apart on an L4 (2e-5 with IEEE, matching Turing and the torch path).
    Call before the first fla kernel compiles. Returns False if fla is not importable.
    """
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ["FLA_TRIL_PRECISION"] = "ieee"
    try:
        import triton.language as tl
        import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
    except ImportError:
        return False
    if hasattr(chunk_fwd, "SOLVE_TRIL_DOT_PRECISION"):
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    return True
