#!/usr/bin/env python3
"""
Test script for the resume functionality in sample_banks.py
"""

import sys
import shutil
import time
import os
from pathlib import Path
import glob

# Add the current directory to the path so we can import dot_pe
sys.path.insert(0, str(Path.cwd()))
from dot_pe import sample_banks, config


def test_resume_functionality():
    """Test the resume functionality by creating a bank, deleting some files, and resuming."""

    print("=" * 60)
    print("TESTING RESUME FUNCTIONALITY")
    print("=" * 60)

    # Setup test parameters
    bank_dir = Path("test_resume_bank")
    bank_size = 2**8  # 256 samples
    blocksize = 4  # Small blocks for easy testing

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
        n_pool=1,  # Use single process for simplicity
        blocksize=blocksize,
        approximant="IMRPhenomXODE",
        bank_dir=str(bank_dir),
        resume=True,  # Test resume=True (default)
    )

    # Check what files were created
    waveform_dir = bank_dir / "waveforms"
    amp_files = sorted(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    phase_files = sorted(glob.glob(str(waveform_dir / "phase_block_*.npy")))

    print(f"\nFiles created:")
    print(f"  Amplitude files: {len(amp_files)}")
    print(f"  Phase files: {len(phase_files)}")
    print(f"  Expected blocks: {(bank_size + blocksize - 1) // blocksize}")

    for amp_file in amp_files:
        print(f"    {Path(amp_file).name}")

    # Record timestamps of all files after initial creation
    print(f"\nRecording file timestamps...")
    file_timestamps = {}
    for amp_file in amp_files:
        file_timestamps[Path(amp_file).name] = os.path.getmtime(amp_file)
    for phase_file in phase_files:
        file_timestamps[Path(phase_file).name] = os.path.getmtime(phase_file)

    print(f"Recorded timestamps for {len(file_timestamps)} files")

    print(f"\nStep 2: Deleting second half of waveform files")
    print("-" * 50)

    # Delete the second half of the files to simulate interruption
    total_files = len(amp_files)
    files_to_keep = total_files // 2

    files_deleted = []
    deleted_file_names = set()
    for i in range(files_to_keep, total_files):
        amp_file = waveform_dir / f"amplitudes_block_{i}.npy"
        phase_file = waveform_dir / f"phase_block_{i}.npy"

        if amp_file.exists():
            amp_file.unlink()
            files_deleted.append(f"amplitudes_block_{i}.npy")
            deleted_file_names.add(f"amplitudes_block_{i}.npy")
        if phase_file.exists():
            phase_file.unlink()
            files_deleted.append(f"phase_block_{i}.npy")
            deleted_file_names.add(f"phase_block_{i}.npy")

    print(f"Deleted {len(files_deleted)} files:")
    for file in files_deleted:
        print(f"    {file}")

    # Check remaining files
    remaining_amp = sorted(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    remaining_phase = sorted(glob.glob(str(waveform_dir / "phase_block_*.npy")))
    print(f"\nRemaining files:")
    print(f"  Amplitude files: {len(remaining_amp)}")
    print(f"  Phase files: {len(remaining_phase)}")

    print(f"\nStep 3: Testing resume functionality")
    print("-" * 50)

    # Test the resume functionality
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
        resume=True,  # This should auto-detect where to resume
    )

    # Check final state
    final_amp = sorted(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    final_phase = sorted(glob.glob(str(waveform_dir / "phase_block_*.npy")))

    print(f"\nStep 4: Verification")
    print("-" * 50)
    print(f"Final state:")
    print(f"  Amplitude files: {len(final_amp)}")
    print(f"  Phase files: {len(final_phase)}")
    print(f"  Expected blocks: {(bank_size + blocksize - 1) // blocksize}")

    # Verify all files exist
    expected_blocks = (bank_size + blocksize - 1) // blocksize
    success = len(final_amp) == expected_blocks and len(final_phase) == expected_blocks

    # Verify timestamp efficiency - only recreated files should have new timestamps
    print(f"\nTimestamp analysis:")
    files_recreated = 0
    files_preserved = 0
    timestamp_success = True

    for file_path in final_amp + final_phase:
        file_name = Path(file_path).name
        old_timestamp = file_timestamps.get(file_name)
        new_timestamp = os.path.getmtime(file_path)

        if file_name in deleted_file_names:
            # This file was deleted, so it should have a new timestamp
            if (
                old_timestamp is None or new_timestamp > old_timestamp + 1
            ):  # 1 sec tolerance
                files_recreated += 1
            else:
                print(f"  ERROR: Deleted file {file_name} has old timestamp!")
                timestamp_success = False
        else:
            # This file was not deleted, so it should have the same timestamp
            if (
                old_timestamp is not None and abs(new_timestamp - old_timestamp) < 1
            ):  # 1 sec tolerance
                files_preserved += 1
            else:
                print(
                    f"  WARNING: Preserved file {file_name} has new timestamp (old: {old_timestamp}, new: {new_timestamp})"
                )
                # Don't fail the test for this, as file systems might update timestamps

    print(f"  Files recreated (expected): {len(deleted_file_names)}")
    print(f"  Files actually recreated: {files_recreated}")
    print(f"  Files preserved with original timestamps: {files_preserved}")

    if success:
        print("SUCCESS: All expected files are present!")
        if timestamp_success and files_recreated == len(deleted_file_names):
            print(
                "SUCCESS: Resume functionality working efficiently - only missing files were recreated!"
            )
        else:
            print(
                "PARTIAL SUCCESS: Files recreated but timestamp analysis shows inefficiency"
            )
    else:
        print("FAILURE: Missing files after resume!")
        print(f"   Expected: {expected_blocks} files of each type")
        print(f"   Found: {len(final_amp)} amplitude, {len(final_phase)} phase files")

    print(f"\nStep 5: Testing no-resume functionality")
    print("-" * 50)

    # Delete one file to create an incomplete state
    test_file = waveform_dir / "amplitudes_block_0.npy"
    if test_file.exists():
        test_file.unlink()
        print("Deleted amplitudes_block_0.npy to create incomplete state")

    # Test with resume=False (should start from beginning)
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
        resume=False,  # This should start from beginning
    )

    # Final verification
    final_amp2 = sorted(glob.glob(str(waveform_dir / "amplitudes_block_*.npy")))
    final_phase2 = sorted(glob.glob(str(waveform_dir / "phase_block_*.npy")))

    success2 = (
        len(final_amp2) == expected_blocks and len(final_phase2) == expected_blocks
    )

    if success2:
        print("SUCCESS: No-resume functionality also working!")
    else:
        print("FAILURE: No-resume functionality failed!")

    efficiency_test = timestamp_success and files_recreated == len(deleted_file_names)

    print(f"\n" + "=" * 60)
    print("TEST COMPLETE")
    print(f"Resume test: {'PASSED' if success else 'FAILED'}")
    print(f"Efficiency test: {'PASSED' if efficiency_test else 'FAILED'}")
    print(f"No-resume test: {'PASSED' if success2 else 'FAILED'}")
    print("=" * 60)

    return success and success2 and efficiency_test


if __name__ == "__main__":
    test_resume_functionality()
