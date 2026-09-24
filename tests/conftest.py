import sys

import torch

# transformers binds the DeltaNet kernel at import time and picks fla whenever it is installed, but
# fla is Triton/CUDA-only. Hide it without a GPU so the CPU path uses the torch fallback.
if not torch.cuda.is_available():
    sys.modules["fla"] = None
