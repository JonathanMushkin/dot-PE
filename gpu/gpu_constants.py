"""
GPU configuration constants.

COMPLEX_DTYPE and REAL_DTYPE set the default precision for GPU
computations throughout the package.
"""

import torch

# Use GPU if available; fall back to CPU for testing without a GPU
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# complex64 = two float32 -- matches numpy complex64
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
