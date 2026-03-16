"""
Flexible profiling script for inference.run_and_profile().

Usage examples:

    # Generate fresh data and bank, then profile:
    python gpu/profile_run.py --bank-size 65536

    # Reuse existing data (fast repeated profiling):
    python gpu/profile_run.py \\
        --event-path gpu/artifacts/profile_run/run_001/test_event.npz \\
        --bank-path  gpu/artifacts/profile_run/test_bank

    # GPU run with same data:
    python gpu/profile_run.py --gpu \\
        --event-path gpu/artifacts/profile_run/run_001/test_event.npz \\
        --bank-path  gpu/artifacts/profile_run/test_bank
"""

import argparse
import pstats
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Profile inference.run()")
    p.add_argument("--event-path", type=Path, default=None,
                   help="Path to existing .npz event file. If absent, generate Gaussian-noise injection.")
    p.add_argument("--bank-path", type=Path, nargs="+", default=None,
                   help="Path(s) to existing bank folder(s). If absent, generate a new bank.")
    p.add_argument("--out-dir", type=Path, default=Path("gpu/artifacts/profile_run"),
                   help="Output directory (default: gpu/artifacts/profile_run).")
    p.add_argument("--bank-size", type=int, default=2**16,
                   help="Bank size when generating (default: 65536).")
    p.add_argument("--n-ext", type=int, default=2048,
                   help="Number of extrinsic samples (default: 2048).")
    p.add_argument("--n-phi", type=int, default=50,
                   help="Phase grid points (default: 50).")
    p.add_argument("--n-t", type=int, default=128,
                   help="Time grid points (default: 128).")
    p.add_argument("--blocksize", type=int, default=2048,
                   help="Coherent block size (default: 2048).")
    p.add_argument("--single-detector-blocksize", type=int, default=2048,
                   help="Incoherent block size (default: 2048).")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU-accelerated run (gpu.run instead of dot_pe.inference).")
    p.add_argument("--preload-bank", action="store_true",
                   help="(GPU only) Preload entire waveform bank into GPU VRAM before inference "
                        "(Track G: eliminates per-batch disk I/O; requires --gpu).")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top-N cumulative-time functions to print (default: 20).")
    p.add_argument("--n-pool", type=int, default=4,
                   help="Worker processes for bank generation (default: 4).")
    return p.parse_args()


def _generate_event(out_dir: Path):
    """Generate a Gaussian-noise injection and save to out_dir."""
    from cogwheel import data, gw_utils

    event_data_kwargs = {
        "detector_names": "HLV",
        "duration": 120.0,
        "asd_funcs": ["asd_H_O3", "asd_L_O3", "asd_V_O3"],
        "tgps": 0.0,
        "fmax": 1600.0,
    }
    eventname = "profile_event"
    event_data = data.EventData.gaussian_noise(
        eventname=eventname, **event_data_kwargs, seed=20250314
    )

    chirp_mass = 20.0
    q = 1.0
    m1, m2 = gw_utils.mchirpeta_to_m1m2(chirp_mass, gw_utils.q_to_eta(q))
    injection_par_dic = dict(
        m1=m1, m2=m2,
        ra=0.5, dec=0.5,
        iota=np.pi / 3, psi=1.0,
        phi_ref=0.0,
        s1z=0.0, s2z=0.0,
        s1x_n=0.0, s1y_n=0.0,
        s2x_n=0.0, s2y_n=0.0,
        l1=0.0, l2=0.0,
        tgps=0.0, f_ref=50.0,
        d_luminosity=500.0,
        t_geocenter=0.0,
    )
    event_data.inject_signal(injection_par_dic, "IMRPhenomXODE")

    event_npz = out_dir / f"{eventname}.npz"
    event_data.to_npz(filename=event_npz, overwrite=True)
    print(f"Event saved to {event_npz}")
    return event_data, event_npz


def _generate_bank(out_dir: Path, bank_size: int, n_pool: int = 4):
    """Generate a template bank with waveforms.

    Uses threadpoolctl to limit each forked worker to 1 BLAS thread so that
    n_pool workers on an N-core machine produce load ≈ n_pool, not N × n_pool.
    """
    from dot_pe import sample_banks, config

    try:
        from threadpoolctl import threadpool_limits
        _ctx = threadpool_limits(limits=1)
    except ImportError:
        import contextlib
        _ctx = contextlib.nullcontext()

    bank_dir = out_dir / f"test_bank_{bank_size}"
    bank_dir.mkdir(parents=True, exist_ok=True)
    with _ctx:
        sample_banks.main(
            bank_size=bank_size,
            q_min=1 / 4,
            m_min=10,
            m_max=40,
            inc_faceon_factor=1.0,
            f_ref=50.0,
            fbin=config.DEFAULT_FBIN,
            n_pool=n_pool,
            blocksize=min(bank_size, 1024),
            approximant="IMRPhenomXODE",
            bank_dir=bank_dir,
        )
    print(f"Bank saved to {bank_dir}")
    return bank_dir


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Event ----
    if args.event_path is not None:
        from cogwheel import data
        event_data = data.EventData.from_npz(filename=args.event_path)
        print(f"Loaded event from {args.event_path}")
    else:
        event_data, _ = _generate_event(out_dir)

    # ---- Bank ----
    if args.bank_path is not None:
        bank_folder = args.bank_path[0] if len(args.bank_path) == 1 else args.bank_path
        print(f"Using existing bank: {bank_folder}")
    else:
        bank_folder = _generate_bank(out_dir, args.bank_size, args.n_pool)

    # ---- Run ----
    run_kwargs = dict(
        event=event_data,
        bank_folder=bank_folder,
        event_dir=str(out_dir / event_data.eventname),
        n_ext=args.n_ext,
        n_phi=args.n_phi,
        n_t=args.n_t,
        blocksize=args.blocksize,
        single_detector_blocksize=args.single_detector_blocksize,
        seed=42,
        draw_subset=False,
    )

    if args.gpu:
        import gpu.run as runner
        if args.preload_bank:
            runner._PRELOAD_ENABLED = True
            print("Using GPU-accelerated run with bank preload (Track G).")
        else:
            print("Using GPU-accelerated run.")
    else:
        from dot_pe import inference as runner
        if args.preload_bank:
            print("WARNING: --preload-bank has no effect without --gpu.")
        print("Using CPU run.")

    print("Starting profiled run...")
    rundir = runner.run_and_profile(**run_kwargs)
    print(f"\nResults written to: {rundir}")

    # ---- Print top-N cumulative functions ----
    profile_txt = Path(rundir) / "profile_output.txt"
    if profile_txt.exists():
        print(f"\n{'='*60}")
        print(f"Top-{args.top_n} cumulative-time functions")
        print(f"{'='*60}")
        import pstats, io
        prof_path = Path(rundir) / "profile_output.prof"
        if prof_path.exists():
            stream = io.StringIO()
            ps = pstats.Stats(str(prof_path), stream=stream)
            ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top_n)
            print(stream.getvalue())
        else:
            # Fallback: print the txt
            print(profile_txt.read_text()[:4000])


if __name__ == "__main__":
    main()
