#!/usr/bin/env python3
"""
Test script to verify bank creation using the flexible sample_banks_flexible module.
Uses the same parameters as notebooks/create_sample_bank.ipynb
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import dot_pe
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe import sample_banks_flexible, config


def test_bank_creation():
    """Test bank creation with the same parameters as the notebook."""

    # Parameters from notebooks/create_sample_bank.ipynb
    bank_size = 2**12  # 4096 samples
    q_min = 1 / 6
    m_min = 50
    m_max = 100
    inc_faceon_factor = 1.0  # isotropic (no face-on bias)
    f_ref = 50.0
    fbin = config.DEFAULT_FBIN
    n_pool = 4
    blocksize = 1024
    approximant = "IMRPhenomXODE"
    bank_dir = Path("test_flexible_bank")
    seed = 42

    print("Testing flexible bank creation...")
    print(f"Parameters:")
    print(f"  bank_size: {bank_size}")
    print(f"  q_min: {q_min}")
    print(f"  m_min: {m_min}, m_max: {m_max}")
    print(f"  inc_faceon_factor: {inc_faceon_factor}")
    print(f"  f_ref: {f_ref}")
    print(f"  bank_dir: {bank_dir}")

    try:
        # Create the bank using the flexible module
        sample_banks_flexible.main(
            bank_size=bank_size,
            q_min=q_min,
            m_min=m_min,
            m_max=m_max,
            inc_faceon_factor=inc_faceon_factor,
            f_ref=f_ref,
            fbin=fbin,
            n_pool=n_pool,
            blocksize=blocksize,
            approximant=approximant,
            bank_dir=bank_dir,
            seed=seed,
        )

        print(f"\n Bank creation successful!")
        print(f"Bank files created in: {bank_dir}")

        # Check that the expected files were created
        expected_files = [
            "intrinsic_sample_bank.feather",
            "bank_config.json",
            "waveforms/",
        ]

        for file_path in expected_files:
            full_path = bank_dir / file_path
            if full_path.exists():
                print(f"  {file_path}")
            else:
                print(f"  {file_path} - NOT FOUND")

        # Print some basic info about the samples
        import pandas as pd

        samples_path = bank_dir / "intrinsic_sample_bank.feather"
        if samples_path.exists():
            samples = pd.read_feather(samples_path)
            print(f"\nSample bank info:")
            print(f"  Number of samples: {len(samples)}")
            print(f"  Columns: {list(samples.columns)}")
            print(
                f"  Weight range: {samples['log_prior_weights'].min():.3f} to {samples['log_prior_weights'].max():.3f}"
            )

            # Print some statistics
            print(f"\nSample statistics:")
            print(
                f"  m1 range: {samples['m1'].min():.1f} - {samples['m1'].max():.1f} M☉"
            )
            print(
                f"  m2 range: {samples['m2'].min():.1f} - {samples['m2'].max():.1f} M☉"
            )
            print(
                f"  iota range: {samples['iota'].min():.2f} - {samples['iota'].max():.2f} rad"
            )

    except Exception as e:
        print(f"\n Bank creation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_bank_creation()
    if success:
        print("\n All tests passed!")
    else:
        print("\n Tests failed!")
        sys.exit(1)
