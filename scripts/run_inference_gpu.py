#!/usr/bin/env python3
"""
GPU-enabled inference script for dot-pe.

This script mimics the workflow in notebooks/run_inference.ipynb
but runs on GPU with fixed n_int and n_ext values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogwheel import data, gw_utils, gw_plotting, utils
from dot_pe import inference
from dot_pe.device_manager import initialize_device


def main(device="cpu"):
    """Run GPU-enabled inference with fixed parameters.
    Parameters
    ----------
    device : str
        Device to run inference on. Either "cpu" or "cuda".
    """

    # Initialize GPU device
    print("Initializing GPU device...")
    initialize_device(device)
    print("Device initialized successfully!")

    # Generate synthetic data for inference
    print("Generating synthetic event data...")
    event_data_kwargs = {
        "detector_names": "HLV",
        "duration": 120.0,
        "asd_funcs": ["asd_H_O3", "asd_L_O3", "asd_V_O3"],
        "tgps": 0.0,
        "fmax": 1600.0,
    }

    eventname = "test_event_gpu"

    event_data = data.EventData.gaussian_noise(
        eventname=eventname, **event_data_kwargs, seed=20250311
    )

    # Set injection parameters
    mchirp = 75
    q = 1 / 2

    m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp, gw_utils.q_to_eta(q))
    injection_par_dic = dict(
        m1=m1,
        m2=m2,
        ra=0.5,
        dec=0.5,
        iota=np.pi * 1 / 3,
        psi=1.0,
        phi_ref=12.0,
        s1z=0.6,
        s2z=0.6,
        s1x_n=0.1,
        s1y_n=0.2,
        s2x_n=0.3,
        s2y_n=-0.2,
        l1=0.0,
        l2=0.0,
        tgps=0.0,
        f_ref=50.0,
        d_luminosity=5e3,
        t_geocenter=0.0,
    )

    event_data.inject_signal(injection_par_dic, "IMRPhenomXODE")

    print(
        f"Signal-to-noise ratio: {event_data.injection['d_h'] - event_data.injection['h_h'] / 2}"
    )
    print(
        f"Total SNR: {sum(event_data.injection['d_h'] - event_data.injection['h_h'] / 2)}"
    )

    # Set bank folder and parameters
    bank_folder = Path("test_bank")
    event_dir = event_data.eventname

    # Check if bank exists
    if not bank_folder.exists():
        print(f"Error: Bank folder {bank_folder} does not exist!")
        print("Please create the bank first using create_sample_bank.ipynb")
        return

    # Read bank size
    try:
        bank_df = pd.read_feather(bank_folder / "intrinsic_sample_bank.feather")
        bank_size = len(bank_df)
        print(f"Bank size: {bank_size}")
    except FileNotFoundError:
        print(f"Error: Could not find intrinsic_sample_bank.feather in {bank_folder}")
        return

    # Fixed parameters for GPU testing
    n_int = min(1024, bank_size)  # Use smaller subset for testing
    n_ext = 256  # Fixed extrinsic sample size
    n_phi = 64
    n_phi_incoherent = 32
    n_t = 64
    blocksize = min(n_int, 512)  # Smaller blocks for testing
    single_detector_blocksize = min(n_int, 512)

    print(f"Running inference with:")
    print(f"  n_int: {n_int}")
    print(f"  n_ext: {n_ext}")
    print(f"  n_phi: {n_phi}")
    print(f"  blocksize: {blocksize}")

    # Run inference
    print("Starting inference...")
    try:
        rundir = inference.run(
            event_dir=event_dir,
            event=event_data,
            bank_folder=bank_folder,
            n_int=n_int,
            n_ext=n_ext,
            n_phi=n_phi,
            n_phi_incoherent=n_phi_incoherent,
            n_t=n_t,
            i_int_start=0,
            blocksize=blocksize,
            single_detector_blocksize=single_detector_blocksize,
            seed=42,
            size_limit=10**5,  # Smaller limit for testing
            draw_subset=False,
            n_draws=None,
        )

        print(f"Inference completed! Results saved to: {rundir}")

        # Read and display summary results
        summary_results = utils.read_json(rundir / "summary_results.json")
        print("\nSummary Results:")
        for k, v in summary_results.items():
            print(f"  {k}: {v}")

        # Load samples and display basic info
        samples_path = rundir / "samples.feather"
        if samples_path.exists():
            samples = pd.read_feather(samples_path)
            print(f"\nSamples shape: {samples.shape}")
            print(f"Sample columns: {list(samples.columns)}")

            # Display some basic statistics
            if "lnl" in samples.columns:
                print(
                    f"Log-likelihood range: {samples['lnl'].min():.2f} to {samples['lnl'].max():.2f}"
                )
            if "mchirp" in samples.columns:
                print(
                    f"Chirp mass range: {samples['mchirp'].min():.2f} to {samples['mchirp'].max():.2f}"
                )
        else:
            print("Warning: samples.feather not found!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\nGPU inference test completed successfully!")


if __name__ == "__main__":
    main()
