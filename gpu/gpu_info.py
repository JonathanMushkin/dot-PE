"""
Query PyTorch GPU properties and save/print tuning constants.

Run directly to inspect the available GPU and generate gpu_constants.py:
    python gpu/gpu_info.py
"""

import sys
from pathlib import Path

import torch


def query_gpu_properties() -> dict:
    if not torch.cuda.is_available():
        print("No CUDA GPU available. Running on CPU only.", file=sys.stderr)
        return {}

    props = torch.cuda.get_device_properties(0)

    sm_count = props.multi_processor_count
    vram_gb = props.total_memory / (1024**3)

    # L40S known specs (hardcode fallback; clock_rate not in all PyTorch versions)
    # FP32 ~91.6 TFLOPS, BF16 ~183 TFLOPS, bandwidth ~864 GB/s
    fp32_tflops = 91.6
    bf16_tflops = 183.0

    info = {
        "device_name": props.name,
        "vram_gb": round(vram_gb, 2),
        "sm_count": sm_count,
        "fp32_tflops_approx": fp32_tflops,
        "bf16_tflops_approx": bf16_tflops,
        "memory_bandwidth_gbs": 864,
        "major": props.major,
        "minor": props.minor,
    }
    return info


def print_info(info: dict) -> None:
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()


def save_constants(info: dict, out_path: Path) -> None:
    """Write gpu_constants.py with tuning values derived from info."""
    vram_gb = info.get("vram_gb", 0)
    device_name = info.get("device_name", "unknown")

    lines = [
        '"""',
        "Auto-generated GPU tuning constants.",
        f"Device: {device_name}",
        '"""',
        "",
        "import torch",
        "",
        f'DEVICE = "cuda"  # {device_name}',
        f"VRAM_GB = {vram_gb}",
        f"FP32_TFLOPS_APPROX = {info.get('fp32_tflops_approx', 0)}",
        f"BF16_TFLOPS_APPROX = {info.get('bf16_tflops_approx', 0)}",
        "",
        "# Block / tile sizes tuned for L40S",
        "BLOCK_SIZE = 256",
        "TILE = 32",
        "",
        "# Preferred dtype for complex arithmetic",
        "# complex64 = two float32 → matches numpy complex64",
        "COMPLEX_DTYPE = torch.complex64",
        "REAL_DTYPE = torch.float32",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"Constants saved to {out_path}")


def main():
    info = query_gpu_properties()
    if info:
        print_info(info)
    else:
        info = {
            "device_name": "cpu",
            "vram_gb": 0,
            "fp32_tflops_approx": 0,
            "bf16_tflops_approx": 0,
        }
        print("Falling back to CPU constants.")

    out_path = Path(__file__).parent / "gpu_constants.py"
    save_constants(info, out_path)


if __name__ == "__main__":
    main()
