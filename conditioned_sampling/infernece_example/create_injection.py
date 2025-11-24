#!/usr/bin/env python3
"""
Create a synthetic injection using IMRPhenomXPHM.

Example:
    python create_injection.py \
        --event-name toy_event \
        --detectors HLV \
        --duration 120 \
        --seed 20250311 \
        --mchirp 30 \
        --q 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cogwheel import data, gw_utils


def build_event_data(detectors: str, duration: float, seed: int) -> data.EventData:
    det_list = list(detectors)
    asd_lookup = {"H": "asd_H_O3", "L": "asd_L_O3", "V": "asd_V_O3"}
    asd_funcs = [asd_lookup[d] for d in det_list if d in asd_lookup]
    if not asd_funcs:
        raise ValueError(f"No ASD functions found for detectors '{detectors}'.")
    event_data_kwargs = {
        "detector_names": detectors,
        "duration": duration,
        "asd_funcs": asd_funcs,
        "tgps": 0.0,
        "fmax": 1600.0,
    }
    return data.EventData.gaussian_noise(
        eventname="placeholder", **event_data_kwargs, seed=seed
    )


def create_injection_params(args: argparse.Namespace) -> dict:
    m1, m2 = gw_utils.mchirpeta_to_m1m2(args.mchirp, gw_utils.q_to_eta(args.q))
    return dict(
        m1=m1,
        m2=m2,
        ra=args.ra,
        dec=args.dec,
        iota=args.iota,
        psi=args.psi,
        phi_ref=args.phi_ref,
        s1z=args.s1z,
        s2z=args.s2z,
        s1x_n=args.s1x_n,
        s1y_n=args.s1y_n,
        s2x_n=args.s2x_n,
        s2y_n=args.s2y_n,
        l1=args.l1,
        l2=args.l2,
        tgps=0.0,
        f_ref=50.0,
        d_luminosity=args.d_luminosity,
        t_geocenter=0.0,
    )


def save_event_data(event: data.EventData, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{event.eventname}.npz"
    event.to_npz(filename=path, overwrite=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IMRPhenomXPHM injection.")
    parser.add_argument("--event-name", default="toy_event")
    parser.add_argument("--detectors", default="HL")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20250311)
    parser.add_argument("--output-dir", default="injection")
    parser.add_argument("--mchirp", type=float, default=30.0)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--ra", type=float, default=0.5)
    parser.add_argument("--dec", type=float, default=0.5)
    parser.add_argument("--iota", type=float, default=np.pi / 3)
    parser.add_argument("--psi", type=float, default=1.0)
    parser.add_argument("--phi-ref", type=float, default=12.0)
    parser.add_argument("--s1z", type=float, default=0.6)
    parser.add_argument("--s2z", type=float, default=0.6)
    parser.add_argument("--s1x-n", type=float, default=0.1)
    parser.add_argument("--s1y-n", type=float, default=0.2)
    parser.add_argument("--s2x-n", type=float, default=0.3)
    parser.add_argument("--s2y-n", type=float, default=-0.2)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--d-luminosity", type=float, default=1.5e3)
    args = parser.parse_args()

    event_data = build_event_data(args.detectors, args.duration, args.seed)
    event_data.eventname = args.event_name

    injection_pars = create_injection_params(args)
    event_data.inject_signal(injection_pars, "IMRPhenomXPHM")

    d_h = event_data.injection.get("d_h")
    h_h = event_data.injection.get("h_h")
    if d_h is not None and h_h is not None:
        inj_lnlike = np.sum(d_h - 0.5 * h_h)
        print(f"injection lnlike: {inj_lnlike}")

    output_path = save_event_data(event_data, Path(args.output_dir))
    print(f"Injection saved to {output_path}")


if __name__ == "__main__":
    main()
