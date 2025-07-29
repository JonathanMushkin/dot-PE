"""
Simple test script for DeviceManager functionality without numpy.
"""

import torch
from pathlib import Path
import sys

# Add the dot_pe module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.device_manager import DeviceManager, get_device_manager


def test_device_manager():
    """Test basic DeviceManager functionality."""
    print("Testing DeviceManager...")

    # Test device manager creation
    device_manager = DeviceManager("auto")
    device_info = device_manager.get_device_info()

    print(f"Device info: {device_info}")
    print(f"Using GPU: {device_manager.is_gpu_available()}")

    # Test tensor conversion with torch tensors
    test_tensor = torch.randn(10, 10) + 1j * torch.randn(10, 10)
    converted_tensor = device_manager.to_tensor(test_tensor)

    print(f"Original tensor device: {test_tensor.device}")
    print(f"Converted tensor device: {converted_tensor.device}")
    print(f"Tensor dtype: {converted_tensor.dtype}")

    # Test numpy conversion
    numpy_array = device_manager.to_numpy(converted_tensor)

    print(f"Numpy array shape: {numpy_array.shape}")
    print(f"Numpy array dtype: {numpy_array.dtype}")

    # Test different data types
    test_float32 = torch.randn(5, 5, dtype=torch.float32)
    test_complex64 = torch.randn(5, 5, dtype=torch.complex64)

    tensor_float32 = device_manager.to_tensor(test_float32)
    tensor_complex64 = device_manager.to_tensor(test_complex64)

    assert tensor_float32.dtype == torch.float32
    assert tensor_complex64.dtype == torch.complex64
    print("✓ Data type preservation test passed")

    # Test global device manager
    global_dm = get_device_manager()
    assert global_dm.device_name == device_manager.device_name
    print("✓ Global device manager test passed")

    print("✓ All DeviceManager tests passed!")


def test_device_selection():
    """Test different device selection options."""
    print("\nTesting device selection...")

    # Test explicit CPU
    cpu_dm = DeviceManager("cpu")
    assert cpu_dm.device_name == "cpu"
    assert not cpu_dm.is_gpu_available()
    print("✓ CPU device selection test passed")

    # Test explicit CUDA (if available)
    if torch.cuda.is_available():
        cuda_dm = DeviceManager("cuda")
        assert cuda_dm.device_name == "cuda"
        assert cuda_dm.is_gpu_available()
        print("✓ CUDA device selection test passed")
    else:
        print("⚠ CUDA not available, skipping CUDA test")

    # Test auto selection
    auto_dm = DeviceManager("auto")
    if torch.cuda.is_available():
        assert auto_dm.device_name == "cuda"
    else:
        assert auto_dm.device_name == "cpu"
    print("✓ Auto device selection test passed")


def main():
    """Run all tests."""
    print("Starting DeviceManager tests...")
    print("=" * 40)

    try:
        test_device_manager()
        test_device_selection()

        print("\n" + "=" * 40)
        print("✓ All tests passed successfully!")

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
