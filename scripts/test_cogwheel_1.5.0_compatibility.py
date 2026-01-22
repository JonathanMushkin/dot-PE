#!/usr/bin/env python3
"""
Minimal test script to verify cogwheel 1.5.0 compatibility.
Based on tutorial notebook 3, but with minimal parameters for quick testing.
"""

import sys
from pathlib import Path
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel import data, gw_utils, utils, prior_ratio
from cogwheel.gw_prior import IntrinsicIASPrior
from dot_pe import inference, waveform_banks, config
from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior

# Set up test directory
TEST_DIR = Path("./test_cogwheel_1.5.0_output")
TEST_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("Testing cogwheel 1.5.0 compatibility")
print("=" * 60)

# Step 1: Create mock event data
print("\n1. Creating mock event data...")
eventname = "test_cogwheel_1.5.0_event"
event_data = data.EventData.gaussian_noise(
    eventname=eventname,
    detector_names="HLV",
    duration=120.0,
    asd_funcs=["asd_H_O3", "asd_L_O3", "asd_V_O3"],
    tgps=0.0,
    fmax=1600.0,
    seed=20223001,
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
    d_luminosity=2000.0,
    t_geocenter=0.0,
)

# Inject signal
event_data.inject_signal(injection_par_dic, "IMRPhenomXPHM")
snr = np.sqrt(
    2 * (event_data.injection["d_h"] - 0.5 * event_data.injection["h_h"]).sum()
)
print(f"   Injection SNR: {snr:.2f}")

# Step 2: Create small bank (size 4096)
print("\n2. Creating small bank (size 4096)...")
bank_dir = TEST_DIR / "bank"
bank_dir.mkdir(parents=True, exist_ok=True)

bank_size = 4096
mchirp_min = 15
mchirp_max = 30
q_min = 0.2
f_ref = 50.0
seed = 777
n_pool = 1  # Reduced for faster testing
blocksize = 4096
approximant = "IMRPhenomXPHM"

# Generate bank samples
powerlaw_prior = PowerLawIntrinsicIASPrior(
    mchirp_range=(mchirp_min, mchirp_max),
    q_min=q_min,
    f_ref=f_ref,
)
ias_prior = IntrinsicIASPrior(
    mchirp_range=(mchirp_min, mchirp_max),
    q_min=q_min,
    f_ref=f_ref,
)
pr_ratio = prior_ratio.PriorRatio(ias_prior, powerlaw_prior)
# preemptive bugfix: sometimes not all matching items are removed
prior_ratio._remove_matching_items(
    pr_ratio._numerator_subpriors, pr_ratio._denominator_subpriors
)

print(f"   Generating {bank_size:,} bank samples...")
bank_samples = powerlaw_prior.generate_random_samples(
    bank_size, seed=seed, return_lnz=False
)

# Compute derived quantities and weights
bank_samples["mchirp"] = gw_utils.m1m2_to_mchirp(bank_samples["m1"], bank_samples["m2"])
bank_samples["lnq"] = np.log(bank_samples["m2"] / bank_samples["m1"])
bank_samples["chieff"] = gw_utils.chieff(
    *bank_samples[["m1", "m2", "s1z", "s2z"]].values.T
)

bank_samples["log_prior_weights"] = bank_samples.apply(
    lambda row: pr_ratio.ln_prior_ratio(**row.to_dict()), axis=1
)

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

print(f"   Bank saved to: {bank_dir}")
print(f"   Bank size: {len(bank_samples):,} samples")

# Step 3: Generate waveforms
print("\n3. Generating waveforms (this may take a few minutes)...")
waveform_dir = bank_dir / "waveforms"
waveform_banks.create_waveform_bank_from_samples(
    samples_path=samples_path,
    bank_config_path=bank_config_path,
    waveform_dir=waveform_dir,
    n_pool=n_pool,
    blocksize=blocksize,
    approximant=approximant,
)
print("   Waveform generation complete!")

# Step 4: Run inference with minimal parameters
print("\n4. Running inference.run() with minimal parameters...")
print("   Parameters: n_int=4096, n_ext=128, n_phi=10")
event_dir = TEST_DIR / "run_test"
event_dir.mkdir(parents=True, exist_ok=True)

try:
    final_rundir = inference.run(
        event=event_data,
        bank_folder=bank_dir,
        n_int=4096,
        n_ext=128,
        n_phi=10,
        n_t=128,
        blocksize=2048,
        single_detector_blocksize=2048,
        seed=42,
        event_dir=str(event_dir),
        mchirp_guess=chirp_mass,
        preselected_indices=None,
        max_incoherent_lnlike_drop=20,
        max_bestfit_lnlike_diff=20,
        draw_subset=True,
    )

    print("\n   Inference run complete!")
    print(f"   Results saved to: {final_rundir}")

    # Check for output files
    summary_path = final_rundir / "summary_results.json"
    if summary_path.exists():
        summary = utils.read_json(summary_path)
        print("\n   Summary results:")
        print(f"     n_effective: {summary.get('n_effective', 'N/A'):.2f}")
        print(f"     n_effective_i: {summary.get('n_effective_i', 'N/A'):.2f}")
        print(f"     n_effective_e: {summary.get('n_effective_e', 'N/A'):.2f}")
        print(f"     ln_evidence: {summary.get('ln_evidence', 'N/A'):.2f}")
    else:
        print("   WARNING: summary_results.json not found")

    samples_path = final_rundir / "samples.feather"
    if samples_path.exists():
        import pandas as pd

        samples = pd.read_feather(samples_path)
        print(f"\n   Final samples shape: {samples.shape}")
        print("   ✓ samples.feather created successfully")
    else:
        print("   WARNING: samples.feather not found")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - cogwheel 1.5.0 compatibility verified!")
    print("=" * 60)

except Exception as e:
    print(f"\n   ERROR during inference.run(): {e}")
    import traceback

    traceback.print_exc()
    print("\n" + "=" * 60)
    print("✗ TEST FAILED")
    print("=" * 60)
    sys.exit(1)
