#!/usr/bin/env python3
"""
Script to generate bank_mchirp_{x} banks with x=3,5,10,20,50,100
and their dense counterparts for reproducing arXiv:2507.16022 experiments.

This script creates banks with:
- Regular banks: 2^16 samples (65536)
- Dense banks: 2^18 samples (262144)
"""

import sys
from pathlib import Path
import logging

# Add dot-PE to path (now we're in the dot-PE directory)
sys.path.insert(0, "..")
from dot_pe import sample_banks, config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Generate all required banks for the paper reproduction."""

    # Bank configurations: mchirp_range -> (min_mchirp, max_mchirp)
    mchirp_ranges = {
        3: (3.0, 3.2),
        5: (5.0, 6.0),
        10: (10.0, 12.0),
        20: (20.0, 30.0),
        50: (50.0, 60.0),
        100: (100.0, 300.0),
    }
    q_min = 1 / 5

    # Bank sizes
    regular_bank_size = 2**16  # 65536
    dense_bank_size = 2**18  # 262144

    # Common parameters from existing configs
    common_params = {
        "q_min": 1 / 5,
        "f_ref": 50.0,
        "fbin": config.DEFAULT_FBIN,
        "inc_faceon_factor": 1.0,
        "approximant": "IMRPhenomXODE",
        "blocksize": 4096,
        "n_pool": 4,
        "seed": None,
    }

    base_dir = Path.cwd()

    for mchirp_val, (min_mchirp, max_mchirp) in mchirp_ranges.items():
        logger.info(f"Creating banks for mchirp range {mchirp_val}")

        # # Create regular bank
        # regular_bank_dir = base_dir / f"bank_mchirp_{mchirp_val}"
        # logger.info(f"Creating regular bank: {regular_bank_dir}")

        # try:
        #     sample_banks.main(
        #         bank_size=regular_bank_size,
        #         m_min=min_mchirp,
        #         m_max=max_mchirp,
        #         bank_dir=regular_bank_dir,
        #         **common_params,
        #     )
        #     logger.info(
        #         f"Successfully created regular bank: {regular_bank_dir}"
        #     )
        # except Exception as e:
        #     logger.error(
        #         f"Failed to create regular bank {regular_bank_dir}: {e}"
        #     )
        #     continue

        # Create dense bank
        dense_bank_dir = base_dir / f"bank_mchirp_dense_{mchirp_val}"
        logger.info(f"Creating dense bank: {dense_bank_dir}")

        try:
            sample_banks.main(
                bank_size=dense_bank_size,
                m_min=min_mchirp,
                m_max=max_mchirp,
                bank_dir=dense_bank_dir,
                **common_params,
            )
            logger.info(f"Successfully created dense bank: {dense_bank_dir}")
        except Exception as e:
            logger.error(f"Failed to create dense bank {dense_bank_dir}: {e}")
            continue

    logger.info("Bank generation completed!")


if __name__ == "__main__":
    main()
