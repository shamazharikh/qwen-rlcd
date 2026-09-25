import sys

import torch

# transformers binds the DeltaNet kernel at import time and picks fla whenever it is installed, but
# fla is Triton/CUDA-only. Hide it without a GPU so the CPU path uses the torch fallback.
if not torch.cuda.is_available():
    sys.modules["fla"] = None
else:
    # On Ampere+ fla computes fp32 dots in TF32, which breaks the exactness bounds.
    from system_one.fork import force_ieee_fp32

    force_ieee_fp32()
