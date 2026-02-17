"""
Test script for multicore parallelization.

Tests run_multicore() with different numbers of cores (4 and 8) using parameters
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
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dot_pe.multicore import run_multicore, MulticoreConfig
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


# Path to existing marg_info.pkl: load from file instead of computing (~3 min saved per run).
# Create once by running without this file present, or use e.g. run_multicore_test_nprocs_4/run_3/marg_info.pkl.
SHARED_MARG_INFO_REL = "run_multicore_test_nprocs_4/run_3/marg_info.pkl"

# Core counts to test (filtered by available_cores in main).
CORE_COUNTS_TO_TEST = [1, 4]


def run_test_with_cores(
    n_procs: int,
    artifacts_dir: Path,
    event_data_path: Path,
    bank_folders: list,
    output_suffix: str = "",
    marg_info_path: Optional[Path] = None,
):
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
    marg_info_path : Path, optional
        If set, load MarginalizationInfo from this file (skip ~3 min extrinsic
        computation). Use a previous run's run_X/marg_info.pkl for scaling tests.

    Returns
    -------
    dict
        Results dictionary with timing and rundir
    """
    print(f"\n{'=' * 70}")
    print(f"Testing with {n_procs} cores")
    print(f"{'=' * 70}")

    # Parameters from notebook
    n_int = 2**15
    n_ext = 1024
    n_phi = 50
    n_t = 128
    blocksize = 2048
    e_blocksize = 64
    single_detector_blocksize = 2048
    seed_ext = sum([int(str(i) * i) for i in range(1, 10)])
    mchirp_guess = 20.0  # chirp_mass from notebook

    # Create output directory
    rundir_base = artifacts_dir / f"run_multicore_test{output_suffix}"
    rundir_base.mkdir(parents=True, exist_ok=True)

    # Create multicore config
    multicore_config = MulticoreConfig(
        n_procs=n_procs,
        i_batch=single_detector_blocksize,  # Use same as single_detector_blocksize
        batches_per_task=1,  # Start with 1 batch per task
    )

    print(
        f"Multicore Config: n_procs={multicore_config.n_procs}, "
        f"i_batch={multicore_config.i_batch}, "
        f"batches_per_task={multicore_config.batches_per_task}"
    )
    print(f"Output directory: {rundir_base}")
    if marg_info_path is not None:
        print(f"Loading marg_info from: {marg_info_path}")

    # Record start time
    start_time = time.time()

    try:
        # Run inference
        final_rundir = run_multicore(
            event=str(event_data_path),
            bank_folder=bank_folders,
            n_int=n_int,
            n_ext=n_ext,
            n_phi=n_phi,
            n_t=n_t,
            blocksize=blocksize,
            e_blocksize=e_blocksize,
            single_detector_blocksize=single_detector_blocksize,
            seed=seed_ext,
            event_dir=str(rundir_base),
            mchirp_guess=mchirp_guess,
            preselected_indices=None,
            max_incoherent_lnlike_drop=20,
            max_bestfit_lnlike_diff=20,
            draw_subset=True,
            multicore_config=multicore_config,
            marg_info_path=str(marg_info_path) if marg_info_path is not None else None,
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

        print(
            f"\nTotal elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)"
        )

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


def compare_rundirs_numerically(
    rundir_multicore: Path,
    rundir_seq: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """
    Compare key outputs from a multicore run vs a sequential run for numerical equivalence.

    Loads summary_results.json and samples.feather from both rundirs and checks
    ln_evidence, n_effective, and sample column agreement within tolerance.
    Use after running run_multicore() and inference.run() on the same inputs.

    Returns
    -------
    dict
        Keys: "match" (bool), "summary_diff" (dict), "samples_match" (bool), "message" (str).
    """
    out = {"match": True, "summary_diff": {}, "samples_match": True, "message": ""}
    try:
        sh = utils.read_json(rundir_multicore / "summary_results.json")
        ss = utils.read_json(rundir_seq / "summary_results.json")
    except Exception as e:
        out["match"] = False
        out["message"] = f"Cannot load summary: {e}"
        return out

    for key in ("ln_evidence", "n_effective", "n_effective_i", "n_effective_e"):
        if key not in sh or key not in ss:
            continue
        a, b = float(sh[key]), float(ss[key])
        diff = abs(a - b)
        out["summary_diff"][key] = {"multicore": a, "seq": b, "diff": diff}
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            out["match"] = False

    try:
        samples_multicore = pd.read_feather(rundir_multicore / "samples.feather")
        samples_seq = pd.read_feather(rundir_seq / "samples.feather")
    except Exception as e:
        out["message"] = f"Cannot load samples: {e}"
        return out

    if set(samples_multicore.columns) != set(samples_seq.columns):
        out["match"] = False
        out["samples_match"] = False
        out["message"] = "Sample columns differ"
        return out
    if len(samples_multicore) != len(samples_seq):
        out["match"] = False
        out["samples_match"] = False
        out["message"] = (
            f"Sample length differ: {len(samples_multicore)} vs {len(samples_seq)}"
        )
        return out

    for col in samples_multicore.columns:
        if not np.issubdtype(samples_multicore[col].dtype, np.number):
            continue
        if not np.allclose(
            samples_multicore[col].values, samples_seq[col].values, rtol=rtol, atol=atol
        ):
            out["match"] = False
            out["samples_match"] = False
            out["message"] = out["message"] or f"Column {col} differs"
    return out


def main():
    """Main test function."""
    print("=" * 70)
    print("Multicore Scaling Test")
    print("=" * 70)

    # Check available cores
    import multiprocessing as mp

    available_cores = mp.cpu_count()
    print(f"\nAvailable CPU cores: {available_cores}")

    requested = CORE_COUNTS_TO_TEST
    test_configs = [n for n in requested if n <= available_cores]

    if not test_configs:
        print(
            f"\nWarning: Requested core counts ({requested}) exceed available cores ({available_cores})"
        )
        print(f"Testing with available cores instead: {available_cores}")
        test_configs = [min(c, available_cores) for c in requested]
        test_configs = list(set(test_configs))

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

    # Use shared marg_info so every run loads from file (no ~3 min extrinsic computation per run)
    shared_marg = artifacts_dir / SHARED_MARG_INFO_REL
    marg_info_path = shared_marg if shared_marg.exists() else None
    if marg_info_path is None:
        print(
            f"\nNote: {SHARED_MARG_INFO_REL} not found; first run will compute marg_info (~3 min)."
        )
    results = []
    for n_procs in test_configs:
        result = run_test_with_cores(
            n_procs=n_procs,
            artifacts_dir=artifacts_dir,
            event_data_path=event_data_path,
            bank_folders=bank_folders,
            output_suffix=f"_nprocs_{n_procs}",
            marg_info_path=marg_info_path,
        )
        results.append(result)
        if n_procs != test_configs[-1]:
            print("\nWaiting 2 seconds before next test...")
            time.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for result in results:
        if result["success"]:
            print(f"\n{result['n_procs']} cores:")
            print(
                f"  Time: {result['elapsed_time']:.2f} seconds ({result['elapsed_time'] / 60:.2f} minutes)"
            )
            print(f"  Rundir: {result['rundir']}")
            if result["samples_shape"]:
                print(f"  Samples shape: {result['samples_shape']}")
        else:
            print(f"\n{result['n_procs']} cores: FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")

    # Compare timings
    successful_results = [r for r in results if r["success"]]
    if len(successful_results) >= 2:
        times = [r["elapsed_time"] for r in successful_results]
        procs = [r["n_procs"] for r in successful_results]

        print("\n" + "=" * 70)
        print("Scaling Analysis")
        print("=" * 70)

        for i in range(len(successful_results)):
            for j in range(i + 1, len(successful_results)):
                r1, r2 = successful_results[i], successful_results[j]
                speedup = r1["elapsed_time"] / r2["elapsed_time"]
                efficiency = speedup * (r1["n_procs"] / r2["n_procs"])

                print(f"\n{r1['n_procs']} cores → {r2['n_procs']} cores:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Efficiency: {efficiency:.2%}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
