#!/usr/bin/env python3
"""
Script to run parameter estimation on GW231123 using the created waveform bank.
"""

import sys
from pathlib import Path
import warnings

# Suppress SWIG LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Add dot-pe to path
dot_pe_path = "/home/projects/barakz/jonatahm/dot-pe"
sys.path.append(dot_pe_path)

from cogwheel.data import EventData
from dot_pe import inference, utils


def run_pe():
    """Run parameter estimation on GW231123."""

    # Event and bank setup
    eventname = "GW231123_135430_inpainted"
    event_data = EventData.from_npz(filename=eventname + ".npz")
    bank_folder = "GW231123_bank_5"
    bank_folder = Path(bank_folder)
    event_dir = event_data.eventname
    bank_size = utils.read_json(bank_folder / "bank_config.json")["bank_size"]

    print(f"Running PE for event: {eventname}")
    print(f"Using bank: {bank_folder}")
    print(f"Bank size: {bank_size}")
    print(f"Event directory: {event_dir}")

    # Run inference
    inference.run(
        event_dir=event_dir,
        event=event_data,
        bank_folder=bank_folder,
        n_int=bank_size,
        n_ext=2**11,
        n_phi=100,
        n_t=128,
        i_int_start=0,
        blocksize=2**12,
        single_detector_blocksize=2**12,
        seed=42,
        size_limit=10**6,
        draw_subset=True,
        n_draws=None,
        mchirp_guess=150.0,
        max_incoherent_lnlike_drop=25.0,
        max_bestfit_lnlike_diff=30.0,
        load_inds=False,
    )

    print("PE analysis completed successfully")


if __name__ == "__main__":
    try:
        run_pe()
    except Exception as e:
        print(f"Error running PE: {e}")
        sys.exit(1)
