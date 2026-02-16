"""
Test script for multicore HPC parallelization.

Tests run_hpc() with different numbers of cores (4 and 8) using parameters
from notebooks/03_run_inference/03_run_inference.ipynb.

This script:
1. Sets up the test environment with proper core control
2. Runs inference with 4 cores
3. Runs inference with 8 cores
4. Compares timing and verifies outputs
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dot_pe.multicore import run_hpc, HPCConfig
from dot_pe import utils


def setup_test_environment():
    """Set up test environment and return paths."""
    # Use artifacts from notebook
    artifacts_dir = project_root / "notebooks" / "03_run_inference" / "artifacts"
    
    if not artifacts_dir.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found: {artifacts_dir}\n"
            "Please run notebooks/03_run_inference/03_run_inference.ipynb first to generate test data."
        )
    
    event_data_path = artifacts_dir / "tutorial_inference_event.npz"
    if not event_data_path.exists():
        raise FileNotFoundError(
            f"Event data not found: {event_data_path}\n"
            "Please run the notebook to generate test data."
        )
    
    bank_names = ["bank_15_20", "bank_20_30"]
    bank_folders = [artifacts_dir / b for b in bank_names]
    
    for bank_folder in bank_folders:
        if not bank_folder.exists():
            raise FileNotFoundError(
                f"Bank folder not found: {bank_folder}\n"
                "Please run the notebook to generate test data."
            )
    
    return artifacts_dir, event_data_path, bank_folders


def run_test_with_cores(n_procs: int, artifacts_dir: Path, event_data_path: Path, 
                        bank_folders: list, output_suffix: str = ""):
    """
    Run inference test with specified number of cores.
    
    Parameters
    ----------
    n_procs : int
        Number of worker processes
    artifacts_dir : Path
        Directory containing artifacts
    event_data_path : Path
        Path to event data file
    bank_folders : list
        List of bank folder paths
    output_suffix : str
        Suffix to add to output directory name
    
    Returns
    -------
    dict
        Results dictionary with timing and rundir
    """
    print(f"\n{'='*70}")
    print(f"Testing with {n_procs} cores")
    print(f"{'='*70}")
    
    # Parameters from notebook
    n_int = 2**15
    n_ext = 1024
    n_phi = 50
    n_t = 128
    blocksize = 2048
    single_detector_blocksize = 2048
    seed_ext = sum([int(str(i) * i) for i in range(1, 10)])
    mchirp_guess = 20.0  # chirp_mass from notebook
    
    # Create output directory
    rundir_base = artifacts_dir / f"run_multicore_test{output_suffix}"
    rundir_base.mkdir(parents=True, exist_ok=True)
    
    # Create HPC config
    hpc_config = HPCConfig(
        n_procs=n_procs,
        i_batch=single_detector_blocksize,  # Use same as single_detector_blocksize
        batches_per_task=1,  # Start with 1 batch per task
    )
    
    print(f"HPC Config: n_procs={hpc_config.n_procs}, "
          f"i_batch={hpc_config.i_batch}, "
          f"batches_per_task={hpc_config.batches_per_task}")
    print(f"Output directory: {rundir_base}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run inference
        final_rundir = run_hpc(
            event=str(event_data_path),
            bank_folder=bank_folders,
            n_int=n_int,
            n_ext=n_ext,
            n_phi=n_phi,
            n_t=n_t,
            blocksize=blocksize,
            single_detector_blocksize=single_detector_blocksize,
            seed=seed_ext,
            event_dir=str(rundir_base),
            mchirp_guess=mchirp_guess,
            preselected_indices=None,
            max_incoherent_lnlike_drop=20,
            max_bestfit_lnlike_diff=20,
            draw_subset=True,
            hpc_config=hpc_config,
        )
        
        elapsed_time = time.time() - start_time
        
        # Load and verify results
        summary_path = final_rundir / "summary_results.json"
        if summary_path.exists():
            summary = utils.read_json(summary_path)
            print(f"\nResults:")
            print(f"  n_effective: {summary.get('n_effective', 'N/A'):.2f}")
            print(f"  n_effective_i: {summary.get('n_effective_i', 'N/A'):.2f}")
            print(f"  n_effective_e: {summary.get('n_effective_e', 'N/A'):.2f}")
            print(f"  ln_evidence: {summary.get('ln_evidence', 'N/A'):.2f}")
        else:
            print(f"\nWarning: Summary file not found at {summary_path}")
            summary = {}
        
        # Check samples file
        samples_path = final_rundir / "samples.feather"
        if samples_path.exists():
            samples = pd.read_feather(samples_path)
            print(f"  Final samples shape: {samples.shape}")
        else:
            print(f"\nWarning: Samples file not found at {samples_path}")
            samples = None
        
        print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return {
            "n_procs": n_procs,
            "elapsed_time": elapsed_time,
            "rundir": final_rundir,
            "summary": summary,
            "samples_shape": samples.shape if samples is not None else None,
            "success": True,
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nERROR: Test failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "n_procs": n_procs,
            "elapsed_time": elapsed_time,
            "rundir": None,
            "summary": {},
            "samples_shape": None,
            "success": False,
            "error": str(e),
        }


def main():
    """Main test function."""
    print("="*70)
    print("Multicore HPC Scaling Test")
    print("="*70)
    
    # Check available cores
    import multiprocessing as mp
    available_cores = mp.cpu_count()
    print(f"\nAvailable CPU cores: {available_cores}")
    
    # Test configurations
    test_configs = [4, 8]
    
    # Filter to only test with available cores
    test_configs = [n for n in test_configs if n <= available_cores]
    
    if not test_configs:
        print(f"\nWarning: Requested core counts ({[4, 8]}) exceed available cores ({available_cores})")
        print(f"Testing with available cores instead: {available_cores}")
        test_configs = [min(4, available_cores), min(8, available_cores)]
        test_configs = list(set(test_configs))  # Remove duplicates
    
    print(f"Will test with: {test_configs} cores")
    
    # Setup environment
    try:
        artifacts_dir, event_data_path, bank_folders = setup_test_environment()
        print(f"\nTest environment setup complete:")
        print(f"  Artifacts directory: {artifacts_dir}")
        print(f"  Event data: {event_data_path}")
        print(f"  Bank folders: {bank_folders}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Run tests
    results = []
    for n_procs in test_configs:
        result = run_test_with_cores(
            n_procs=n_procs,
            artifacts_dir=artifacts_dir,
            event_data_path=event_data_path,
            bank_folders=bank_folders,
            output_suffix=f"_nprocs_{n_procs}",
        )
        results.append(result)
        
        # Brief pause between tests
        if n_procs != test_configs[-1]:
            print("\nWaiting 2 seconds before next test...")
            time.sleep(2)
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for result in results:
        if result["success"]:
            print(f"\n{result['n_procs']} cores:")
            print(f"  Time: {result['elapsed_time']:.2f} seconds ({result['elapsed_time']/60:.2f} minutes)")
            print(f"  Rundir: {result['rundir']}")
            if result['samples_shape']:
                print(f"  Samples shape: {result['samples_shape']}")
        else:
            print(f"\n{result['n_procs']} cores: FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Compare timings
    successful_results = [r for r in results if r["success"]]
    if len(successful_results) >= 2:
        times = [r["elapsed_time"] for r in successful_results]
        procs = [r["n_procs"] for r in successful_results]
        
        print("\n" + "="*70)
        print("Scaling Analysis")
        print("="*70)
        
        for i in range(len(successful_results)):
            for j in range(i+1, len(successful_results)):
                r1, r2 = successful_results[i], successful_results[j]
                speedup = r1["elapsed_time"] / r2["elapsed_time"]
                efficiency = speedup * (r1["n_procs"] / r2["n_procs"])
                
                print(f"\n{r1['n_procs']} cores → {r2['n_procs']} cores:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Efficiency: {efficiency:.2%}")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
