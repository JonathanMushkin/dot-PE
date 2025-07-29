#!/usr/bin/env python3
"""
Create a small sample bank for GPU inference testing.

This script creates a smaller bank than the notebook version,
optimized for testing GPU inference with run_inference_gpu.py.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogwheel import gw_plotting, gw_utils, utils
from dot_pe import sample_banks, config


def main():
    """Create a small sample bank for GPU testing."""
    
    print("Creating small sample bank for GPU inference testing...")
    
    bank_size = 2**16  
    q_min = 1 / 6
    m_min = 50
    m_max = 100
    inc_faceon_factor = 1.0
    f_ref = 50.0
    fbin = config.DEFAULT_FBIN
    n_pool = 8
    blocksize = 4096
    approximant = "IMRPhenomXODE"
    bank_dir = Path("test_bank")
    
    print(f"Bank parameters:")
    print(f"  Bank size: {bank_size}")
    print(f"  Mass range: {m_min} - {m_max} M☉")
    print(f"  q_min: {q_min}")
    print(f"  f_ref: {f_ref} Hz")
    print(f"  Approximant: {approximant}")
    print(f"  Bank directory: {bank_dir}")
    
    # Create the bank
    print("\nGenerating bank samples...")
    sample_banks.main(
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
    )
    
    
    print(f"\nSmall bank ready for GPU inference testing!")
    print(f"Run: python scripts/run_inference_gpu.py")


if __name__ == "__main__":
    main() 