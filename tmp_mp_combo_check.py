import warnings
from pathlib import Path
import multiprocessing as mp

import numpy as np

from dot_pe import mp_inference


def main():
    art = Path("notebooks/03_run_inference/artifacts")
    event = art / "tutorial_inference_event.npz"
    bank = art / "bank"
    out = art / "tmp_combo_checks"
    out.mkdir(exist_ok=True)

    combos = [
        ("none_preselected_nint4", None, 4),
        ("preselected3_nint2_warn", [0, 1, 2], 2),
        ("preselected3_nint10", [0, 1, 2], 10),
        ("preselected3_nintNone", [0, 1, 2], None),
    ]

    for name, preselected, n_int in combos:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mp_inference.run(
                event=event,
                bank_folder=bank,
                n_ext=8,
                n_phi=4,
                n_t=8,
                n_int=n_int,
                blocksize=16,
                single_detector_blocksize=8,
                n_workers=1,
                n_ext_workers=1,
                seed=123,
                size_limit=10**5,
                draw_subset=False,
                n_draws=None,
                event_dir=out / f"mp_{name}",
                rundir=None,
                max_incoherent_lnlike_drop=20,
                max_bestfit_lnlike_diff=20,
                mchirp_guess=None,
                extrinsic_samples=None,
                preselected_indices=preselected,
                profile=False,
                load_inds=False,
                inds_path=None,
            )
            warn_msgs = [
                str(x.message)
                for x in w
                if "n_int (" in str(x.message) and "preselected indices" in str(x.message)
            ]
            bank_run_dirs = sorted((Path(result) / "banks").glob("bank_*"))
            intr_file = bank_run_dirs[0] / "intrinsic_samples.npz"
            accepted = np.load(intr_file)["inds"]
            print(
                f"{name}: accepted={len(accepted)}, first={accepted[:10].tolist()}, "
                f"warnings={len(warn_msgs)}"
            )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
