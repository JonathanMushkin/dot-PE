#!/usr/bin/env python3
"""Example: Create a new bank from a saved Zoomer.

Two ways to create a bank from iter1 results:

1. CLI approach (recommended):
   python zoom_iteration.py \
       --prev-run-dir conditioned_sampling/infernece_example/zoom/iter1/run_0 \
       --bank-dir conditioned_sampling/infernece_example/zoom/iter1/bank \
       --load-zoomer conditioned_sampling/infernece_example/zoom/iter1/run_0/Zoomer.json \
       --output-dir conditioned_sampling/infernece_example/zoom/iter1_new_bank \
       --approximant IMRPhenomXPHM \
       --n-samples 4096 \
       --bank-blocksize 1024 \
       --n-ext 1024 \
       --inference-blocksize 1024 \
       --n-pool 8 \
       --seed 42

2. Python approach (this script):
   python create_bank_from_zoomer_example.py
"""

from pathlib import Path
import sys

sys.path.insert(0, "/Users/jonatahm/Work/GW/dot-pe-future")

import json
import numpy as np
from dot_pe.zoom.zoom import Zoomer
from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler
from conditioned_sampling.infernece_example.zoom_iteration import (
    draw_from_zoomer,
    format_intrinsic_bank,
)

# Paths
zoom_run_dir = Path("conditioned_sampling/infernece_example/zoom/iter1/run_0")
small_zoom_bank_dir = Path("conditioned_sampling/infernece_example/zoom/iter1/bank")
output_dir = Path("conditioned_sampling/infernece_example/zoom/zoomed_bank")

# Load Zoomer
zoomer_path = zoom_run_dir / "Zoomer.json"
zoomer = Zoomer.from_json(zoomer_path)
print(f"Loaded Zoomer from {zoomer_path}")

# Load or create ConditionalPriorSampler
cond_sampler_path = zoom_run_dir / "ConditionalPriorSampler.json"
if cond_sampler_path.exists():
    cond_sampler = ConditionalPriorSampler.from_json(cond_sampler_path)
    print(f"Loaded ConditionalPriorSampler from {cond_sampler_path}")
else:
    # Load bank_config to get prior_kwargs
    bank_config_path = small_zoom_bank_dir / "bank_config.json"
    with open(bank_config_path, "r") as f:
        bank_config = json.load(f)

    mchirp_range = bank_config.get("mchirp_range")
    if mchirp_range is None:
        mchirp_range = (bank_config["mchirp_min"], bank_config["mchirp_max"])

    prior_kwargs = dict(
        mchirp_range=tuple(mchirp_range),
        q_min=bank_config.get("q_min"),
        f_ref=bank_config.get("f_ref"),
    )
    cond_sampler = ConditionalPriorSampler(**prior_kwargs, seed=42)
    print("Created ConditionalPriorSampler from bank_config")

# Get bounds (from Zoomer or bank_config)
if zoomer.bounds is not None:
    bounds = zoomer.bounds
    print(f"Using bounds from Zoomer: {bounds}")
else:
    # Fallback to bank_config bounds
    bank_config_path = small_zoom_bank_dir / "bank_config.json"
    with open(bank_config_path, "r") as f:
        bank_config = json.load(f)

    mchirp_range = bank_config.get("mchirp_range")
    if mchirp_range is None:
        mchirp_range = (bank_config["mchirp_min"], bank_config["mchirp_max"])

    q_min = bank_config.get("q_min", 0.1)
    bounds = {
        0: (mchirp_range[0], mchirp_range[1]),
        1: (np.log(q_min), 0.0),
        2: (-1.0, 1.0),
    }
    print(f"Using bounds from bank_config: {bounds}")

# Draw samples
n_samples = 2**16
print(f"\nDrawing {n_samples} samples from Zoomer...")
new_bank = draw_from_zoomer(
    zoomer=zoomer,
    cond_sampler=cond_sampler,
    bounds=bounds,
    n_samples=n_samples,
    seed=42,
)

# Format and save bank
output_dir.mkdir(parents=True, exist_ok=True)
bank_output_dir = output_dir / "bank"
bank_output_dir.mkdir(parents=True, exist_ok=True)

formatted_bank = format_intrinsic_bank(new_bank)
samples_path = bank_output_dir / "intrinsic_sample_bank.feather"
formatted_bank.to_feather(samples_path)
print(f"Saved bank to {samples_path}")

# Optionally generate waveforms
# Uncomment the following to also generate waveforms:
"""
from dot_pe import waveform_banks
import os

approximant = "IMRPhenomXPHM"
bank_blocksize = 1024
n_pool = max(1, os.cpu_count() or 1)

# Load bank_config for waveform generation
bank_config_path = iter1_bank_dir / "bank_config.json"
with open(bank_config_path, "r") as f:
    bank_config = json.load(f)

zoom_config = dict(bank_config)
zoom_config.update({
    "bank_size": len(formatted_bank),
    "parent_bank_dir": str(iter1_bank_dir),
    "zoom_iteration": True,
    "zoom_seed": 42,
    "zoom_n_samples": n_samples,
    "approximant": approximant,
    "blocksize": bank_blocksize,
    "n_pool": n_pool,
})

config_path = bank_output_dir / "bank_config.json"
with open(config_path, "w") as f:
    json.dump(zoom_config, f, indent=2)

waveform_dir = bank_output_dir / "waveforms"
waveform_dir.mkdir(exist_ok=True)

print(f"Generating waveforms in {waveform_dir}...")
waveform_banks.create_waveform_bank_from_samples(
    samples_path=samples_path,
    bank_config_path=config_path,
    waveform_dir=waveform_dir,
    n_pool=n_pool,
    blocksize=bank_blocksize,
    approximant=approximant,
)
print("Waveform generation complete.")
"""
