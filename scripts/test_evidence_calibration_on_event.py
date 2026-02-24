#!/usr/bin/env python3
"""Test evidence calibration on an event (creates noise and runs compute_evidence_on_noise)."""

import argparse
import sys
from pathlib import Path

from dot_pe.evidence_calibration import compute_evidence_on_noise


def main():
    parser = argparse.ArgumentParser(
        description="Test evidence calibration: create noise and compute ln_evidence."
    )
    parser.add_argument("rundir", type=Path, help="Path to completed inference rundir")
    parser.add_argument(
        "--event",
        type=Path,
        default=None,
        help="Path to original event npz (when run_kwargs resolution fails)",
    )
    parser.add_argument(
        "--noise",
        type=Path,
        default=None,
        help="Path to pre-saved noise npz (skip creation, use this instead)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for noise",
    )
    parser.add_argument(
        "--no-event-psd",
        action="store_true",
        help="Use cogwheel ASD models instead of event PSD (when creating noise)",
    )
    parser.add_argument(
        "--use-selected-inds",
        action="store_true",
        help="Load selection from intrinsic_samples.npz (default: use entire bank)",
    )
    args = parser.parse_args()

    ln_evidence, ln_evidence_discarded = compute_evidence_on_noise(
        args.rundir,
        event_data_noise=args.noise,
        event_path=args.event,
        use_event_psd=not args.no_event_psd,
        use_selected_inds=args.use_selected_inds,
        seed=args.seed,
    )
    print(f"ln_evidence = {ln_evidence:.6f}")
    print(f"ln_evidence_discarded = {ln_evidence_discarded:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
