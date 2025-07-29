"""
Test script for PyTorch conversion of single detector methods.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add the dot_pe module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.device_manager import DeviceManager
from dot_pe.single_detector import SingleDetectorProcessor


def create_test_data(n_det=3, n_int=50, n_t=16, n_phi=32, n_bins=100):
    """
    Create test data for single detector method.
    """
    np.random.seed(42)

    m_arr = np.array([2, 1, 3, 4])
    n_modes = len(m_arr)
    n_Modes = n_modes * (n_modes + 1) // 2
    n_pol = 2

    # Create test data using the same approach as test_pytorch_conversion.py
    dh_weights_dmpb = np.random.randn(
        n_det, n_modes, n_pol, n_bins
    ) + 1j * np.random.randn(n_det, n_modes, n_pol, n_bins)
    h_impb = np.random.randn(n_int, n_modes, n_pol, n_bins) + 1j * np.random.randn(
        n_int, n_modes, n_pol, n_bins
    )
    timeshift_dbt = np.random.randn(n_det, n_bins, n_t) + 1j * np.random.randn(
        n_det, n_bins, n_t
    )
    hh_weights_dmppb = np.random.randn(n_det, n_Modes, n_pol, n_pol, n_bins)
    asd_drift_d = np.random.randn(n_det)

    return {
        "dh_weights_dmpb": dh_weights_dmpb,
        "hh_weights_dmppb": hh_weights_dmppb,
        "h_impb": h_impb,
        "timeshift_dbt": timeshift_dbt,
        "asd_drift_d": asd_drift_d,
        "m_arr": m_arr,
        "n_phi": n_phi,
    }


def test_get_response_over_distance_and_lnlike():
    """Test PyTorch get_response_over_distance_and_lnlike produces same results."""
    print("Testing get_response_over_distance_and_lnlike...")

    # Create test data
    test_data = create_test_data()

    # Test the method
    print("Running get_response_over_distance_and_lnlike...")
    r_iotp, lnlike_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        test_data["dh_weights_dmpb"],
        test_data["hh_weights_dmppb"],
        test_data["h_impb"],
        test_data["timeshift_dbt"],
        test_data["asd_drift_d"],
        test_data["n_phi"],
        test_data["m_arr"],
    )

    print(f"r_iotp shape: {r_iotp.shape}")
    print(f"lnlike_iot shape: {lnlike_iot.shape}")
    print(f"r_iotp range: [{r_iotp.real.min():.6f}, {r_iotp.real.max():.6f}]")
    print(f"lnlike_iot range: [{lnlike_iot.min():.6f}, {lnlike_iot.max():.6f}]")
    print("✓ get_response_over_distance_and_lnlike test passed")


def benchmark_performance():
    """Benchmark performance of PyTorch operations."""
    print("\nBenchmarking performance...")

    # Create larger test data for benchmarking
    test_data = create_test_data(n_int=200, n_t=32, n_phi=64)

    # Benchmark get_response_over_distance_and_lnlike
    print("Benchmarking get_response_over_distance_and_lnlike...")

    import time

    start_time = time.time()
    r_iotp, lnlike_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        test_data["dh_weights_dmpb"],
        test_data["hh_weights_dmppb"],
        test_data["h_impb"],
        test_data["timeshift_dbt"],
        test_data["asd_drift_d"],
        test_data["n_phi"],
        test_data["m_arr"],
    )
    pytorch_time = time.time() - start_time

    print(f"PyTorch time: {pytorch_time:.4f}s")
    print(f"r_iotp shape: {r_iotp.shape}")
    print(f"lnlike_iot shape: {lnlike_iot.shape}")
    print("✓ Benchmark test passed")


def main():
    """Run all tests."""
    print("Starting single detector PyTorch conversion tests...")
    print("=" * 60)

    try:
        test_get_response_over_distance_and_lnlike()
        benchmark_performance()

        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
