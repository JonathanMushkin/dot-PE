#!/usr/bin/env python3
"""
Test script for scattered missing files resume functionality
"""

import sys
import shutil
import random
from pathlib import Path
import glob

# Add the current directory to the path so we can import dot_pe
sys.path.insert(0, str(Path.cwd()))
from dot_pe import sample_banks, config


def test_scattered_resume():
    """Test the i_list functionality with scattered missing files."""

    print("=" * 60)
    print("TESTING SCATTERED MISSING FILES RESUME")
    print("=" * 60)

    # Setup test parameters
    bank_dir = Path("test_scattered_bank")
    bank_size = 100  # Small for quick testing
    blocksize = 4

    # Clean up any existing test directory
    if bank_dir.exists():
        print(f"Removing existing test directory: {bank_dir}")
        shutil.rmtree(bank_dir)

    print(
        f"\nStep 1: Creating initial bank with {bank_size} samples, blocksize={blocksize}"
    )
    print("-" * 50)

    # Create the initial bank
    sample_banks.main(
        bank_size=bank_size,
        q_min=1 / 6,
        m_min=30,
        m_max=80,
        inc_faceon_factor=1.0,
        f_ref=50.0,
        fbin=config.DEFAULT_FBIN,
        n_pool=1,
        blocksize=blocksize,
        approximant="IMRPhenomXODE",
        bank_dir=str(bank_dir),
        resume=True,
    )

    # Check what files were created
    waveform_dir = bank_dir / "waveforms"
    expected_blocks = (bank_size + blocksize - 1) // blocksize

    print(f"\nStep 2: Randomly deleting scattered files to simulate your failure mode")
    print("-" * 50)

    # Delete random scattered files (not contiguous) to simulate your scenario
    all_blocks = list(range(expected_blocks))
    random.seed(42)  # For reproducible test
    blocks_to_delete = random.sample(all_blocks, k=expected_blocks // 3)  # Delete ~1/3
    blocks_to_delete.sort()

    deleted_files = []
    for block_idx in blocks_to_delete:
        amp_file = waveform_dir / f"amplitudes_block_{block_idx}.npy"
        phase_file = waveform_dir / f"phase_block_{block_idx}.npy"

        if amp_file.exists():
            amp_file.unlink()
            deleted_files.append(f"amplitudes_block_{block_idx}.npy")
        if phase_file.exists():
            phase_file.unlink()
            deleted_files.append(f"phase_block_{block_idx}.npy")

    print(f"Deleted {len(deleted_files)} files from blocks: {blocks_to_delete}")

    # Count remaining files
    remaining_amp = len(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    remaining_phase = len(glob.glob(str(waveform_dir / "phase_block_*.npy")))
    print(f"Remaining: {remaining_amp} amplitude files, {remaining_phase} phase files")
    print(
        f"Missing: {expected_blocks - remaining_amp} amplitude files, {expected_blocks - remaining_phase} phase files"
    )

    print(f"\nStep 3: Testing auto-detection of scattered missing files")
    print("-" * 50)

    # Test auto-detection (should find the scattered missing blocks)
    sample_banks.main(
        bank_size=bank_size,
        q_min=1 / 6,
        m_min=30,
        m_max=80,
        inc_faceon_factor=1.0,
        f_ref=50.0,
        fbin=config.DEFAULT_FBIN,
        n_pool=1,
        blocksize=blocksize,
        approximant="IMRPhenomXODE",
        bank_dir=str(bank_dir),
        resume=True,
    )

    # Check final state
    final_amp = len(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    final_phase = len(glob.glob(str(waveform_dir / "phase_block_*.npy")))

    print(f"\nStep 4: Verification")
    print("-" * 50)
    print(f"Final state:")
    print(f"  Amplitude files: {final_amp}")
    print(f"  Phase files: {final_phase}")
    print(f"  Expected: {expected_blocks}")

    success_auto = final_amp == expected_blocks and final_phase == expected_blocks

    if success_auto:
        print("SUCCESS: Auto-detection found and fixed all scattered missing files!")
    else:
        print("FAILURE: Auto-detection did not fix all missing files!")

    print(f"\nStep 5: Testing manual i_list specification")
    print("-" * 50)

    # Delete a few more files and test manual i_list
    test_blocks = [5, 15, 20]  # Manually specify these blocks
    for block_idx in test_blocks:
        amp_file = waveform_dir / f"amplitudes_block_{block_idx}.npy"
        phase_file = waveform_dir / f"phase_block_{block_idx}.npy"
        if amp_file.exists():
            amp_file.unlink()
        if phase_file.exists():
            phase_file.unlink()

    print(f"Deleted blocks {test_blocks} for manual i_list test")

    # Test manual i_list specification
    sample_banks.main(
        bank_size=bank_size,
        q_min=1 / 6,
        m_min=30,
        m_max=80,
        inc_faceon_factor=1.0,
        f_ref=50.0,
        fbin=config.DEFAULT_FBIN,
        n_pool=1,
        blocksize=blocksize,
        approximant="IMRPhenomXODE",
        bank_dir=str(bank_dir),
        i_list=test_blocks,  # Manually specify which blocks to generate
        resume=True,
    )

    # Verify the manual blocks were regenerated
    success_manual = True
    for block_idx in test_blocks:
        amp_file = waveform_dir / f"amplitudes_block_{block_idx}.npy"
        phase_file = waveform_dir / f"phase_block_{block_idx}.npy"
        if not (amp_file.exists() and phase_file.exists()):
            success_manual = False
            break

    if success_manual:
        print("SUCCESS: Manual i_list specification worked correctly!")
    else:
        print("FAILURE: Manual i_list specification failed!")

    print(f"\n" + "=" * 60)
    print("TEST COMPLETE")
    print(f"Auto-detection test: {'PASSED' if success_auto else 'FAILED'}")
    print(f"Manual i_list test: {'PASSED' if success_manual else 'FAILED'}")
    print("=" * 60)

    return success_auto and success_manual


if __name__ == "__main__":
    test_scattered_resume()
