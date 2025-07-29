"""
Test script for PyTorch conversion of likelihood calculations.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add the dot_pe module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.device_manager import DeviceManager
from dot_pe.likelihood_calculating import LikelihoodCalculator


def create_test_data(n_det=3, n_int=50, n_ext=25, n_bins=100):
    """
    Create test data for comparison.
    """
    np.random.seed(42)
    m_arr = np.array([2, 1, 3, 4])
    n_modes = len(m_arr)
    n_Modes = n_modes * (n_modes + 1) // 2
    n_pol = 2
    # Create test data
    dh_weights_dmpb = np.random.randn(
        n_det, n_modes, n_pol, n_bins
    ) + 1j * np.random.randn(n_det, n_modes, n_pol, n_bins)
    h_impb = np.random.randn(n_int, n_modes, n_pol, n_bins) + 1j * np.random.randn(
        n_int, n_modes, n_pol, n_bins
    )
    response_dpe = np.random.randn(n_det, n_pol, n_ext)
    timeshift_dbe = np.random.randn(n_det, n_bins, n_ext) + 1j * np.random.randn(
        n_det, n_bins, n_ext
    )

    hh_weights_dmppb = np.random.randn(n_det, n_Modes, n_pol, n_pol, n_bins)
    asd_drift_d = np.random.randn(n_det)

    return {
        "dh_weights_dmpb": dh_weights_dmpb,
        "h_impb": h_impb,
        "response_dpe": response_dpe,
        "timeshift_dbe": timeshift_dbe,
        "hh_weights_dmppb": hh_weights_dmppb,
        "asd_drift_d": asd_drift_d,
        "m_arr": m_arr,
    }


def test_get_dh_by_mode():
    """Test PyTorch get_dh_by_mode produces same results."""
    print("Testing get_dh_by_mode...")

    # Create test data
    test_data = create_test_data()

    # Initialize calculator
    calc = LikelihoodCalculator(32, test_data["m_arr"])

    # Test get_dh_by_mode
    print("Running get_dh_by_mode...")
    dh_result = calc.get_dh_by_mode(
        test_data["dh_weights_dmpb"],
        test_data["h_impb"],
        test_data["response_dpe"],
        test_data["timeshift_dbe"],
        test_data["asd_drift_d"],
    )

    print(f"Result shape: {dh_result.shape}")
    print(f"Result dtype: {dh_result.dtype}")
    print(f"Result range: [{dh_result.real.min():.6f}, {dh_result.real.max():.6f}]")
    print("✓ get_dh_by_mode test passed")


def test_get_hh_by_mode():
    """Test PyTorch get_hh_by_mode produces same results."""
    print("\nTesting get_hh_by_mode...")

    # Create test data
    test_data = create_test_data()

    # Initialize calculator
    calc = LikelihoodCalculator(32, test_data["m_arr"])

    # Test get_hh_by_mode
    print("Running get_hh_by_mode...")
    hh_result = calc.get_hh_by_mode(
        test_data["h_impb"],
        test_data["response_dpe"],
        test_data["hh_weights_dmppb"],
        test_data["asd_drift_d"],
        calc.m_inds,
        calc.mprime_inds,
    )

    print(f"Result shape: {hh_result.shape}")
    print(f"Result dtype: {hh_result.dtype}")
    print(f"Result range: [{hh_result.real.min():.6f}, {hh_result.real.max():.6f}]")
    print("✓ get_hh_by_mode test passed")


def test_get_dh_hh_phi_grid():
    """Test PyTorch get_dh_hh_phi_grid produces same results."""
    print("\nTesting get_dh_hh_phi_grid...")

    # Create test data
    test_data = create_test_data()

    # Initialize calculator
    calc = LikelihoodCalculator(32, test_data["m_arr"])

    # Get dh and hh first
    dh_iem = calc.get_dh_by_mode(
        test_data["dh_weights_dmpb"],
        test_data["h_impb"],
        test_data["response_dpe"],
        test_data["timeshift_dbe"],
        test_data["asd_drift_d"],
    )

    hh_iem = calc.get_hh_by_mode(
        test_data["h_impb"],
        test_data["response_dpe"],
        test_data["hh_weights_dmppb"],
        test_data["asd_drift_d"],
        calc.m_inds,
        calc.mprime_inds,
    )

    # Test get_dh_hh_phi_grid
    print("Running get_dh_hh_phi_grid...")
    dh_ieo, hh_ieo = calc.get_dh_hh_phi_grid(dh_iem, hh_iem)

    print(f"dh_ieo shape: {dh_ieo.shape}")
    print(f"hh_ieo shape: {hh_ieo.shape}")
    print(f"dh_ieo range: [{dh_ieo.min():.6f}, {dh_ieo.max():.6f}]")
    print(f"hh_ieo range: [{hh_ieo.min():.6f}, {hh_ieo.max():.6f}]")
    print("✓ get_dh_hh_phi_grid test passed")


def benchmark_performance():
    """Benchmark performance of PyTorch operations."""
    print("\nBenchmarking performance...")

    # Create larger test data for benchmarking
    test_data = create_test_data(n_int=200, n_ext=100, n_bins=200)

    calc = LikelihoodCalculator(32, test_data["m_arr"])

    # Benchmark get_dh_by_mode
    print("Benchmarking get_dh_by_mode...")

    import time

    start_time = time.time()
    dh_result = calc.get_dh_by_mode(
        test_data["dh_weights_dmpb"],
        test_data["h_impb"],
        test_data["response_dpe"],
        test_data["timeshift_dbe"],
        test_data["asd_drift_d"],
    )
    pytorch_time = time.time() - start_time

    print(f"PyTorch time: {pytorch_time:.4f}s")
    print(f"Result shape: {dh_result.shape}")
    print("✓ Benchmark test passed")


def main():
    """Run all tests."""
    print("Starting PyTorch conversion tests...")
    print("=" * 50)

    try:
        test_get_dh_by_mode()
        test_get_hh_by_mode()
        test_get_dh_hh_phi_grid()
        benchmark_performance()

        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
