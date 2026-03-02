#!/usr/bin/env python3
"""Idempotent test data setup: creates a mock event + small/large banks once.

Usage:
    python test_data/setup.py [--n-pool N] [--base-dir DIR]

Outputs (inside base-dir, default: test_data/):
    event/tutorial_event.npz  — gaussian noise + IMRPhenomXPHM injection
    bank_small/               — 4 096 intrinsic samples + waveforms
    bank_large/               — 262 144 (2^18) intrinsic samples + waveforms

Re-running is safe: each artifact is skipped if it already exists.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cogwheel import data as cogdata, gw_utils, prior_ratio
from cogwheel.gw_prior import IntrinsicIASPrior
from dot_pe import config, waveform_banks
from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior

# ── fixed parameters (matching notebooks/03_run_inference) ────────────────────
CHIRP_MASS, Q = 20.0, 0.7
MCHIRP_MIN, MCHIRP_MAX, Q_MIN, F_REF = 15, 30, 0.2, 50.0
APPROXIMANT = "IMRPhenomXPHM"
EVENTNAME = "tutorial_event"

BANK_COLUMNS = [
    "m1", "m2", "s1z", "s1x_n", "s1y_n",
    "s2z", "s2x_n", "s2y_n", "iota", "log_prior_weights",
]

BANKS = {
    "bank_small": {"bank_size": 2**12, "seed": 777},
    "bank_large": {"bank_size": 2**18, "seed": 778},
}


def _make_event(base: Path) -> None:
    out = base / "event" / f"{EVENTNAME}.npz"
    if out.exists():
        print(f"[skip]   event  → {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[create] event  → {out}")

    m1, m2 = gw_utils.mchirpeta_to_m1m2(CHIRP_MASS, gw_utils.q_to_eta(Q))
    injection = dict(
        m1=m1, m2=m2, ra=0.5, dec=0.5, iota=np.pi / 3, psi=1.0, phi_ref=12.0,
        s1z=0.3, s2z=0.3, s1x_n=0.1, s1y_n=0.2, s2x_n=0.3, s2y_n=-0.2,
        l1=0.0, l2=0.0, tgps=0.0, f_ref=F_REF, d_luminosity=2000.0, t_geocenter=0.0,
    )
    ev = cogdata.EventData.gaussian_noise(
        eventname=EVENTNAME, detector_names="HLV", duration=120.0,
        asd_funcs=["asd_H_O3", "asd_L_O3", "asd_V_O3"],
        tgps=0.0, fmax=1600.0, seed=20223001,
    )
    ev.inject_signal(injection, APPROXIMANT)
    ev.to_npz(filename=str(out), overwrite=True)
    print(f"         done")


def _make_bank(bank_dir: Path, bank_size: int, seed: int, n_pool: int) -> None:
    config_path = bank_dir / "bank_config.json"
    wav_dir = bank_dir / "waveforms"

    if config_path.exists() and wav_dir.exists() and any(wav_dir.iterdir()):
        print(f"[skip]   {bank_dir.name}")
        return

    bank_dir.mkdir(parents=True, exist_ok=True)
    print(f"[create] {bank_dir.name}  ({bank_size:,} samples)")

    # Build prior objects
    powerlaw_prior = PowerLawIntrinsicIASPrior(
        mchirp_range=(MCHIRP_MIN, MCHIRP_MAX), q_min=Q_MIN, f_ref=F_REF,
    )
    ias_prior = IntrinsicIASPrior(
        mchirp_range=(MCHIRP_MIN, MCHIRP_MAX), q_min=Q_MIN, f_ref=F_REF,
    )
    pr = prior_ratio.PriorRatio(ias_prior, powerlaw_prior)
    prior_ratio._remove_matching_items(pr._numerator_subpriors, pr._denominator_subpriors)

    # Draw samples
    print(f"         generating samples (seed={seed})...")
    samples = powerlaw_prior.generate_random_samples(bank_size, seed=seed, return_lnz=False)

    print(f"         computing prior weights ({bank_size:,} rows)...")
    samples["log_prior_weights"] = samples.apply(
        lambda row: pr.ln_prior_ratio(**row.to_dict()), axis=1
    )

    samples_path = bank_dir / "intrinsic_sample_bank.feather"
    samples[BANK_COLUMNS].to_feather(samples_path)

    bank_cfg = {
        "bank_size": bank_size,
        "mchirp_min": MCHIRP_MIN,
        "mchirp_max": MCHIRP_MAX,
        "q_min": Q_MIN,
        "f_ref": F_REF,
        "fbin": config.DEFAULT_FBIN.tolist(),
        "approximant": APPROXIMANT,
        "m_arr": [2, 1, 3, 4],
        "seed": seed,
    }
    with open(config_path, "w") as f:
        json.dump(bank_cfg, f, indent=4)

    # Generate waveforms
    print(f"         generating waveforms (n_pool={n_pool})...")
    waveform_banks.create_waveform_bank_from_samples(
        samples_path=samples_path,
        bank_config_path=config_path,
        waveform_dir=wav_dir,
        n_pool=n_pool,
        blocksize=4096,
        approximant=APPROXIMANT,
    )
    print(f"         done")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-pool", type=int, default=4,
        help="parallel workers for waveform generation (default: 4)",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).parent,
        help="output root directory (default: test_data/)",
    )
    args = parser.parse_args()

    base = args.base_dir.resolve()
    print(f"Test data root: {base}\n")

    _make_event(base)
    print()
    for name, cfg in BANKS.items():
        _make_bank(base / name, cfg["bank_size"], cfg["seed"], args.n_pool)
        print()

    print("All done. Artifact paths:")
    print(f"  event      : {base}/event/{EVENTNAME}.npz")
    for name in BANKS:
        print(f"  {name:<12}: {base}/{name}/")


if __name__ == "__main__":
    main()
