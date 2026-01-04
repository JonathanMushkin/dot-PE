import json
import shutil
import sys
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from cogwheel import data, gw_utils, gw_plotting, utils
from dot_pe import inference, waveform_banks, config
from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior
from dot_pe.utils import safe_logsumexp


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


SINGLE_BANK_SIZE = 2**16
MULTI_BANK_SIZE = 2**15
N_EXT = 2**10
N_PHI = 32
N_T = 64
BLOCKSIZE = 2**12


def create_injection_event(tmp_path, seed=20223001):
    """Create a mock event data with injected signal."""
    eventname = "test_injection_event"
    event_data = data.EventData.gaussian_noise(
        eventname=eventname,
        detector_names="HLV",
        duration=120.0,
        asd_funcs=["asd_H_O3", "asd_L_O3", "asd_V_O3"],
        tgps=0.0,
        fmax=1600.0,
        seed=seed,
    )

    # Injection parameters
    chirp_mass = 20.0
    q = 0.7
    m1, m2 = gw_utils.mchirpeta_to_m1m2(chirp_mass, gw_utils.q_to_eta(q))

    injection_par_dic = dict(
        m1=m1,
        m2=m2,
        ra=0.5,
        dec=0.5,
        iota=np.pi / 3,
        psi=1.0,
        phi_ref=12.0,
        s1z=0.3,
        s2z=0.3,
        s1x_n=0.1,
        s1y_n=0.2,
        s2x_n=0.3,
        s2y_n=-0.2,
        l1=0.0,
        l2=0.0,
        tgps=0.0,
        f_ref=50.0,
        d_luminosity=1500.0,
        t_geocenter=0.0,
    )

    # Inject signal
    event_data.inject_signal(injection_par_dic, "IMRPhenomXPHM")

    # Save event data
    event_path = tmp_path / f"{eventname}.npz"
    print(
        f"event log likelihood: {event_data.injection.get('d_h') - 1 / 2 * event_data.injection.get('h_h')}"
    )
    event_data.to_npz(filename=event_path, overwrite=True)

    return event_data, injection_par_dic, event_path


def create_bank(
    tmp_path,
    bank_dir_name,
    mchirp_min,
    mchirp_max,
    bank_size,
    seed=777,
    approximant="IMRPhenomXPHM",
    generate_waveforms=True,
    n_pool=1,
):
    """Create a single bank with samples and waveforms."""
    bank_dir = tmp_path / bank_dir_name
    bank_dir.mkdir(parents=True, exist_ok=True)

    q_min = 0.2
    f_ref = 50.0
    blocksize = 4096

    # Generate bank samples
    powerlaw_prior = PowerLawIntrinsicIASPrior(
        mchirp_range=(mchirp_min, mchirp_max),
        q_min=q_min,
        f_ref=f_ref,
    )

    bank_samples = powerlaw_prior.generate_random_samples(
        bank_size, seed=seed, return_lnz=False
    )

    # Compute derived quantities and weights
    bank_samples["mchirp"] = gw_utils.m1m2_to_mchirp(
        bank_samples["m1"], bank_samples["m2"]
    )
    bank_samples["lnq"] = np.log(bank_samples["m2"] / bank_samples["m1"])
    mchirp_values = bank_samples["mchirp"].values
    bank_samples["log_prior_weights"] = 1.7 * np.log(mchirp_values)

    # Save bank
    bank_columns = [
        "m1",
        "m2",
        "s1z",
        "s1x_n",
        "s1y_n",
        "s2z",
        "s2x_n",
        "s2y_n",
        "iota",
        "log_prior_weights",
    ]
    samples_path = bank_dir / "intrinsic_sample_bank.feather"
    bank_samples[bank_columns].to_feather(samples_path)

    # Save bank config
    bank_config = {
        "bank_size": bank_size,
        "mchirp_min": mchirp_min,
        "mchirp_max": mchirp_max,
        "q_min": q_min,
        "f_ref": f_ref,
        "fbin": config.DEFAULT_FBIN.tolist(),
        "approximant": approximant,
        "m_arr": [2, 1, 3, 4],
        "seed": seed,
    }
    bank_config_path = bank_dir / "bank_config.json"
    with open(bank_config_path, "w") as f:
        json.dump(bank_config, f, indent=4)

    # Generate waveforms if requested
    if generate_waveforms:
        waveform_dir = bank_dir / "waveforms"
        waveform_banks.create_waveform_bank_from_samples(
            samples_path=samples_path,
            bank_config_path=bank_config_path,
            waveform_dir=waveform_dir,
            n_pool=n_pool,
            blocksize=blocksize,
            approximant=approximant,
        )

    return bank_dir


def test_multi_bank_inference_3_banks(tmp_path):
    """Test multi-bank inference with 3 banks and proper normalization."""
    # Create injection
    event_data, injection_par_dic, event_path = create_injection_event(tmp_path)

    # First, run single-bank inference for comparison
    print("\n" + "=" * 70)
    print("Running single-bank inference for comparison...")
    print("=" * 70)

    single_bank_size = SINGLE_BANK_SIZE
    single_bank_dir = create_bank(
        tmp_path,
        "bank_single",
        mchirp_min=15,
        mchirp_max=30,
        bank_size=single_bank_size,
        seed=777,
        generate_waveforms=True,
        n_pool=1,
    )

    n_ext = N_EXT
    n_phi = N_PHI
    n_t = N_T
    mchirp_guess = 20.0

    # Create rundir for single-bank inference (use different name to avoid conflicts)
    single_rundir_path = tmp_path / "run_single_bank_comparison"
    single_rundir_path.mkdir(exist_ok=True)

    single_rundir = inference.run(
        event=event_data,
        bank_folder=single_bank_dir,
        n_int=single_bank_size,
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=BLOCKSIZE,
        single_detector_blocksize=BLOCKSIZE,
        rundir=single_rundir_path,
        mchirp_guess=mchirp_guess,
        max_incoherent_lnlike_drop=20,
        max_bestfit_lnlike_diff=20,
        draw_subset=True,
        seed=42,
    )

    single_summary = utils.read_json(single_rundir / "summary_results.json")
    single_samples = pd.read_feather(single_rundir / "samples.feather")

    # Validate single-bank results
    assert np.isfinite(single_summary["ln_evidence"]), "ln_evidence is not finite"
    assert single_summary["n_effective"] > 0, "n_effective should be positive"
    assert single_summary["n_banks"] == 1, "Should have 1 bank"
    assert len(single_samples) > 0, "Samples should not be empty"

    # Save single-bank summary for comparison
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    separator = "=" * 70
    single_summary_txt = (
        f"{separator}\n"
        f"SINGLE BANK RESULTS\n"
        f"{separator}\n"
        f"n_banks: {single_summary['n_banks']}\n"
        f"ln_evidence: {single_summary['ln_evidence']:.6f}\n"
        f"n_effective: {single_summary['n_effective']:.2f}\n"
        f"n_effective_i: {single_summary['n_effective_i']:.2f}\n"
        f"n_effective_e: {single_summary['n_effective_e']:.2f}\n"
        f"bestfit_lnlike_max: {single_summary.get('bestfit_lnlike_max', 'N/A')}\n"
        f"lnl_marginalized_max: {single_summary.get('lnl_marginalized_max', 'N/A')}\n"
        f"n_i_inds_used: {single_summary.get('n_i_inds_used', 'N/A')}\n"
        f"n_distance_marginalizations: {single_summary.get('n_distance_marginalizations', 'N/A')}\n"
        f"n_samples: {len(single_samples)}\n"
        f"{separator}\n"
    )
    print(single_summary_txt)
    with open(plots_dir / "single_bank_summary.txt", "w") as f:
        f.write(single_summary_txt)

    # Create single-bank corner plot
    try:
        true_mchirp = gw_utils.m1m2_to_mchirp(
            injection_par_dic["m1"], injection_par_dic["m2"]
        )
        true_lnq = np.log(injection_par_dic["m2"] / injection_par_dic["m1"])
        true_chieff = gw_utils.chieff(
            injection_par_dic["m1"],
            injection_par_dic["m2"],
            injection_par_dic["s1z"],
            injection_par_dic["s2z"],
        )

        true_values = injection_par_dic | {
            "mchirp": true_mchirp,
            "lnq": true_lnq,
            "chieff": true_chieff,
        }

        params = ["mchirp", "lnq", "chieff", "iota", "ra", "dec", "d_luminosity"]
        corner_plot = gw_plotting.CornerPlot(single_samples, params=params, smooth=1.0)
        corner_plot.plot(max_figsize=5)
        corner_plot.scatter_points(
            true_values, colors="red", marker=".", s=200, label="Injection"
        )
        plot_path = plots_dir / "single_bank_corner.png"
        corner_plot.fig.savefig(plot_path, dpi=100)
        print(f"Single-bank corner plot saved to: {plot_path}")
    except Exception as e:
        print(f"Warning: Could not create single-bank corner plot: {e}")

    # Now create 3 banks split by mchirp ranges
    print("\n" + "=" * 70)
    print("Running multi-bank inference with 3 banks...")
    print("=" * 70)

    bank_size = MULTI_BANK_SIZE  # Smaller for test speed
    mchirp_ranges = [(15, 20), (20, 25), (25, 30)]
    bank_names = ["bank_0", "bank_1", "bank_2"]
    bank_dirs = []

    for i, (bank_name, (mchirp_min, mchirp_max)) in enumerate(
        zip(bank_names, mchirp_ranges)
    ):
        bank_dir = create_bank(
            tmp_path,
            bank_name,
            mchirp_min=mchirp_min,
            mchirp_max=mchirp_max,
            bank_size=bank_size,
            seed=888 + i,
            generate_waveforms=True,
            n_pool=1,  # Serial for tests
        )
        bank_dirs.append(bank_dir)

    # No normalization needed - each bank computes its own Monte Carlo average
    # The evidence is just the sum of the per-bank averages
    bank_logw_override_dict = None

    # Use same n_int for all banks (decoupled from normalization)
    n_int_dict = {f"bank_{i}": bank_size for i in range(3)}

    # Create rundir for multi-bank inference
    multi_rundir_path = tmp_path / "run_multi_bank"
    multi_rundir_path.mkdir(exist_ok=True)

    rundir = inference.run(
        event=event_data,
        bank_folder=bank_dirs,
        n_int=n_int_dict,
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=BLOCKSIZE,
        single_detector_blocksize=BLOCKSIZE,
        rundir=multi_rundir_path,
        mchirp_guess=mchirp_guess,
        max_incoherent_lnlike_drop=20,
        max_bestfit_lnlike_diff=20,
        bank_logw_override=bank_logw_override_dict,
        draw_subset=True,
        seed=42,
    )

    # Validate results
    summary_path = rundir / "summary_results.json"
    assert summary_path.exists(), "summary_results.json not found"
    summary = utils.read_json(summary_path)

    assert summary["n_banks"] == 3, "Should have 3 banks"
    assert np.isfinite(summary["ln_evidence"]), "ln_evidence is not finite"
    assert summary["n_effective"] > 0, "n_effective should be positive"

    # Prepare plots directory
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print and save multi-bank results
    summary_lines = [
        "=" * 70,
        "MULTI-BANK (3 BANKS) RESULTS",
        "=" * 70,
        f"n_banks: {summary['n_banks']}",
        f"ln_evidence: {summary['ln_evidence']:.6f}",
        f"n_effective: {summary['n_effective']:.2f}",
        f"n_effective_i: {summary['n_effective_i']:.2f}",
        f"n_effective_e: {summary['n_effective_e']:.2f}",
        f"bestfit_lnlike_max: {summary.get('bestfit_lnlike_max', 'N/A')}",
        f"lnl_marginalized_max: {summary.get('lnl_marginalized_max', 'N/A')}",
        f"n_i_inds_used: {summary.get('n_i_inds_used', 'N/A')}",
        f"n_distance_marginalizations: {summary.get('n_distance_marginalizations', 'N/A')}",
        "",
    ]

    # Verify per-bank results exist
    assert "per_bank_results" in summary, "per_bank_results should exist"
    per_bank = summary["per_bank_results"]
    assert len(per_bank) == 3, "Should have 3 per-bank results"

    summary_lines.append("Per-bank results:")
    for bank_id in sorted(per_bank.keys()):
        bank_result = per_bank[bank_id]
        summary_lines.extend(
            [
                f"  {bank_id}:",
                f"    ln_evidence: {bank_result['ln_evidence']:.6f}",
                f"    n_effective: {bank_result['n_effective']:.2f}",
                f"    N_k: {bank_result['N_k']}",
                f"    n_inds_used: {bank_result['n_inds_used']}",
            ]
        )

    summary_txt = "\n".join(summary_lines) + "\n" + "=" * 70 + "\n"
    print(summary_txt)
    with open(plots_dir / "multi_bank_summary.txt", "w") as f:
        f.write(summary_txt)

    # Check evidence aggregation formula
    # lnZ_total = logsumexp(lnZ_k + log(N_k)) - log(N_total)
    # Use only banks that actually have results
    bank_results = [
        {
            "lnZ_k": per_bank[bank_id]["ln_evidence"],
            "N_k": per_bank[bank_id]["N_k"],
        }
        for bank_id in sorted(per_bank.keys())
    ]

    lnZ_values = [r["lnZ_k"] + np.log(r["N_k"]) for r in bank_results]
    N_total = sum(r["N_k"] for r in bank_results)
    lnZ_total_expected = safe_logsumexp(lnZ_values) - np.log(N_total)

    assert np.isclose(
        summary["ln_evidence"],
        lnZ_total_expected,
        rtol=1e-5,
        atol=1e-6,
    ), (
        f"Evidence aggregation mismatch: {summary['ln_evidence']} vs {lnZ_total_expected}"
    )

    # Load samples and verify bank_id column exists
    samples_path = rundir / "samples.feather"
    assert samples_path.exists(), "samples.feather not found"
    samples = pd.read_feather(samples_path)
    assert len(samples) > 0, "Samples should not be empty"
    assert "bank_id" in samples.columns, "samples should have bank_id column"

    # Check which banks actually have samples (no assumption all banks must have samples)
    unique_bank_ids = samples["bank_id"].unique()
    print(f"\n  Total samples: {len(samples)}")
    print(f"  Banks with samples: {len(unique_bank_ids)} out of {len(per_bank)}")
    for bank_id in sorted(unique_bank_ids):
        n_bank_samples = (samples["bank_id"] == bank_id).sum()
        print(f"    {bank_id}: {n_bank_samples} samples")

    try:
        true_mchirp = gw_utils.m1m2_to_mchirp(
            injection_par_dic["m1"], injection_par_dic["m2"]
        )
        true_lnq = np.log(injection_par_dic["m2"] / injection_par_dic["m1"])
        true_chieff = gw_utils.chieff(
            injection_par_dic["m1"],
            injection_par_dic["m2"],
            injection_par_dic["s1z"],
            injection_par_dic["s2z"],
        )

        true_values = injection_par_dic | {
            "mchirp": true_mchirp,
            "lnq": true_lnq,
            "chieff": true_chieff,
        }

        params = ["mchirp", "lnq", "chieff", "iota", "ra", "dec", "d_luminosity"]

        # Create MultiCornerPlot with single-bank and combined multi-bank
        plot_samples = [single_samples, samples]
        plot_labels = ["Single Bank", "Multi-Bank (Combined)"]

        multi_plot = gw_plotting.MultiCornerPlot(
            plot_samples,
            params=params,
            smooth=1.0,
            labels=plot_labels,
        )
        multi_plot.plot(max_figsize=7)

        # Add injection point to all subplots
        multi_plot.scatter_points(
            true_values, colors="red", marker=".", s=200, label="Injection"
        )

        plot_path = plots_dir / "comparison_corner.png"
        fig = plt.gcf()
        fig.savefig(plot_path, dpi=100)
        print(f"\nComparison plot saved to: {plot_path}")
        print(f"  Showing {len(plot_samples)} sets: {', '.join(plot_labels)}")
    except Exception as e:
        # Corner plot is optional, don't fail test if it fails
        print(f"Warning: Could not create corner plot: {e}")


if __name__ == "__main__":
    # Use fixed location for test artifacts
    tmpdir = Path(__file__).parent / "artifacts" / "test_disjoint_bank_weights"

    # Clean and create directory to avoid overwrite issues
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    plots_dir = tmpdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Simple stdout redirector to both console and file

    output_file = plots_dir / "test_output.txt"
    tee = Tee(output_file)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print(f"Running tests, saving to: {tmpdir}")
        print(f"All output saved to: {output_file}")
        print("=" * 70)

        # Run multi-bank test (includes single-bank comparison)
        print("\n>>> Running test_multi_bank_inference_3_banks <<<")
        test_multi_bank_inference_3_banks(tmpdir)

        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print(f"Results saved in: {tmpdir}")
        print(f"  - Single bank comparison: {tmpdir / 'run_single_bank_comparison'}")
        print(f"  - Multi-bank results: {tmpdir / 'run_multi_bank'}")
        print(f"  - Plots and summaries: {tmpdir / 'plots'}")
        print("=" * 70)

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"Test failed with error: {e}")
        print(f"{'=' * 70}")
        import traceback

        traceback.print_exc()
        sys.stdout = old_stdout
        tee.close()
        raise
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"\nAll output saved to: {output_file}")
        print(f"Test artifacts saved to: {tmpdir}")
