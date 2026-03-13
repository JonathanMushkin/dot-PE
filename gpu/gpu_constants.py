"""
Auto-generated GPU tuning constants.
Device: NVIDIA L40S
"""

import torch

DEVICE = "cuda"  # NVIDIA L40S
VRAM_GB = 44.39
FP32_TFLOPS_APPROX = 91.6
BF16_TFLOPS_APPROX = 183.0

# Block / tile sizes tuned for L40S
BLOCK_SIZE = 256
TILE = 32

# Preferred dtype for complex arithmetic
# complex64 = two float32 → matches numpy complex64
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32
