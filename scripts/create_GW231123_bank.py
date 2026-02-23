#!/usr/bin/env python3
"""
Script to create waveform banks for GW231123 analysis.
Can be run from terminal and used with LSF job submission.
"""

import argparse
import sys
from pathlib import Path
import warnings

# Suppress SWIG LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Add dot-pe to path
dot_pe_path = "/home/projects/barakz/jonatahm/dot-pe"
sys.path.append(dot_pe_path)

from dot_pe import sample_banks, config
from dot_pe.utils import validate_q_bounds


def create_bank(
    bank_dir,
    bank_size,
    n_pool,
    blocksize,
    mchirp_min=100,
    mchirp_max=200,
    q_min=0.2,
    q_max=1.0,
    f_ref=30.0,
    inc_faceon_factor=1,
    use_limited_fbin=False,
    resume=True,
):
    """
    Create a waveform bank for GW231123 analysis.

    Parameters:
    -----------
    bank_dir : str
        Directory to save the bank
    bank_size : int
        Number of samples in the bank (power of 2)
    n_pool : int
        Number of parallel processes
    blocksize : int
        Block size for waveform generation (power of 2)
    mchirp_min : float
        Minimum chirp mass
    mchirp_max : float
        Maximum chirp mass
    q_min : float
        Minimum mass ratio
    q_max : float
        Maximum mass ratio (default: 1.0)
    f_ref : float
        Reference frequency
    inc_faceon_factor : float
        Face-on inclination factor
    use_limited_fbin : bool
        Whether to limit frequency bins to < 250 Hz
    resume : bool
        Whether to resume from existing files with auto-detection (default: True)
    """

    # Set up frequency bins
    fbin = config.DEFAULT_FBIN
    if use_limited_fbin:
        fbin = fbin[fbin < 250]  # merger frequency is around 50 Hz

    print(f"Creating bank with {bank_size} samples in {bank_dir}")
    print(f"Using {n_pool} processes with blocksize {blocksize}")
    print(f"Chirp mass range: {mchirp_min} - {mchirp_max}")

    bank_main = sample_banks.main
    bank_main(
        bank_size=bank_size,
        q_min=q_min,
        m_min=mchirp_min,
        m_max=mchirp_max,
        q_max=q_max,
        inc_faceon_factor=inc_faceon_factor,
        f_ref=f_ref,
        fbin=fbin,
        n_pool=n_pool,
        blocksize=blocksize,
        bank_dir=bank_dir,
        resume=resume,
    )

    print(f"Bank creation completed: {bank_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create waveform bank for GW231123")

    # Required arguments
    parser.add_argument("bank_dir", help="Directory to save the bank")
    parser.add_argument(
        "--bank-size",
        type=int,
        default=2**20,
        help="Bank size (default: 2^20 = 1048576)",
    )

    # Optional arguments
    parser.add_argument(
        "--n-pool",
        type=int,
        default=6,
        help="Number of parallel processes (default: 6)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=2**13,
        help="Block size for waveform generation (default: 2^13 = 8192)",
    )
    parser.add_argument(
        "--mchirp-min",
        type=float,
        default=100,
        help="Minimum chirp mass (default: 100)",
    )
    parser.add_argument(
        "--mchirp-max",
        type=float,
        default=200,
        help="Maximum chirp mass (default: 200)",
    )
    parser.add_argument(
        "--q-min", type=float, default=0.2, help="Minimum mass ratio (default: 0.2)"
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.0,
        help="Maximum mass ratio, must satisfy q_min < q_max <= 1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--f-ref", type=float, default=30.0, help="Reference frequency (default: 30.0)"
    )
    parser.add_argument(
        "--limited-fbin", action="store_true", help="Limit frequency bins to < 250 Hz"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing files with auto-detection (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from scratch, overwrite existing files",
    )

    args = parser.parse_args()
    validate_q_bounds(args.q_min, args.q_max)

    try:
        create_bank(
            bank_dir=args.bank_dir,
            bank_size=args.bank_size,
            n_pool=args.n_pool,
            blocksize=args.blocksize,
            mchirp_min=args.mchirp_min,
            mchirp_max=args.mchirp_max,
            q_min=args.q_min,
            q_max=args.q_max,
            f_ref=args.f_ref,
            use_limited_fbin=args.limited_fbin,
            resume=args.resume,
        )
    except Exception as e:
        print(f"Error creating bank: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
