import os
import sys

import torch

# transformers binds the DeltaNet kernel at import time and picks fla whenever it is installed, but
# fla is Triton/CUDA-only. Hide it without a GPU so the CPU path uses the torch fallback.
if not torch.cuda.is_available():
    sys.modules["fla"] = None

# On Ampere+ Triton (and so fla) computes fp32 dots in TF32, which moves reads by ~4e-5 and breaks the
# tight exactness bounds. fla only forces IEEE on pre-Ampere cards; force it everywhere for tests.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("FLA_TRIL_PRECISION", "ieee")
