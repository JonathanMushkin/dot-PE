#!/usr/bin/env python3
"""
Simple GPU setup test script.

This script verifies that the GPU acceleration is working correctly.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.device_manager import initialize_device, get_device_manager


def test_gpu_setup():
    """Test basic GPU functionality."""
    print("Testing GPU setup...")

    # Initialize device
    initialize_device("cuda")
    device_manager = get_device_manager()

    print(f"Device info: {device_manager.get_device_info()}")

    # Test basic tensor operations
    print("\nTesting basic tensor operations...")

    # Create test data
    a = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)
    b = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)

    # Convert to tensors
    a_tensor = device_manager.to_tensor(a)
    b_tensor = device_manager.to_tensor(b)

    print(f"Tensor a device: {a_tensor.device}")
    print(f"Tensor b device: {b_tensor.device}")

    # Test matrix multiplication
    c_tensor = torch.matmul(a_tensor, b_tensor)
    print(f"Result tensor device: {c_tensor.device}")
    print(f"Result shape: {c_tensor.shape}")

    # Test einsum
    d_tensor = torch.einsum("ij,jk->ik", a_tensor, b_tensor)
    print(f"Einsum result device: {d_tensor.device}")

    # Test complex operations
    e_tensor = torch.exp(1j * a_tensor)
    print(f"Complex exp result device: {e_tensor.device}")

    # Convert back to numpy
    c_numpy = device_manager.to_numpy(c_tensor)
    print(f"Converted back to numpy, shape: {c_numpy.shape}")

    print("\nGPU setup test passed!")


def test_device_manager():
    """Test device manager functionality."""
    print("\nTesting device manager...")

    device_manager = get_device_manager()

    # Test different data types
    test_data = [
        np.random.randn(10, 10),
        np.random.randn(10, 10).astype(np.float32),
        np.random.randn(10, 10) + 1j * np.random.randn(10, 10),
        np.random.randn(10, 10) + 1j * np.random.randn(10, 10).astype(np.complex64),
    ]

    for i, data in enumerate(test_data):
        tensor = device_manager.to_tensor(data)
        print(f"Data {i + 1}: {data.dtype} -> {tensor.dtype}, device: {tensor.device}")

    print("Device manager test passed!")


def main():
    """Run all tests."""
    print("Starting GPU setup tests...")
    print("=" * 50)

    try:
        test_gpu_setup()
        test_device_manager()

        print("\n" + "=" * 50)
        print("All GPU tests passed!")
        print("GPU acceleration is ready to use.")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
