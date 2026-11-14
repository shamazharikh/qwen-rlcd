import sys

import torch

# transformers binds the DeltaNet kernel at import time and picks fla whenever it is installed, but
# fla is Triton/CUDA-only. Hide it without a GPU so the CPU path uses the torch fallback.
if not torch.cuda.is_available():
    sys.modules["fla"] = None
else:
    # Tests compare fp32 reads to tight tolerances; TF32 in fla's Triton kernels breaks that on Ampere+.
    from system_one.kernels import disable_fla_tf32

    disable_fla_tf32()
