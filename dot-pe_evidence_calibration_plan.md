---
name: Evidence Calibration Plan
overview: Given a completed rundir from a real-event run, replace the data with Gaussian noise and recompute the Bayesian evidence. Starting point is a done rundir—do NOT call inference.run(). Load from rundir, swap in noise data, accumulate the grand-sum. Interpretation is out of scope.
todos: []
isProject: false
---

# Evidence Calibration: ln_evidence on Gaussian Noise

## Goal

Given a **completed rundir** from a real-event run, replace the data with Gaussian noise and recompute the Bayesian evidence. **Do NOT call `inference.run()`.** The starting point is a done rundir—load par_dic_0, extrinsic samples, bank indices, etc. from it, swap in Gaussian noise as the data, and run only the likelihood/evidence computation. Scope: swap the data, accumulate the grand-sum, get ln_evidence. Interpretation is out of scope.

---

## Prerequisites: par_dic_0 from Rundir

We do **not** call `inference.run()` or `prepare_run_objects`. par_dic_0 is loaded from the completed rundir (already computed for the real run). No optimization on noise.

---

## Critical: No Optimization

**The coherent posterior / likelihood objects must NOT try to optimize anything.** They should accept par_dic_0 as given and only compute the relative binning weights. No `find_bestfit_pars`, no fitting to the data—we are running on noise; any optimization would fit a waveform to noise, which is wrong.

## Critical: Extrinsic Samples from Rundir

Extrinsic samples are taken **directly from the rundir** (e.g. `rundir/extrinsic_samples.feather`). Not from elsewhere.

---

## Workflow

### 1. Prerequisite: Completed Rundir

A real-event run has already finished. The rundir contains par_dic_0, extrinsic samples, bank indices, bank configs, `run_kwargs.json`, `banks/<bank_id>/` with CLP state, etc.

### 2. Create Gaussian-Noise EventData

Use the same metadata as the real event so the setup is identical:

- Same detectors, duration, PSD, `tgps`, `fmax`
- No injection: `EventData.gaussian_noise(psd_funcs=..., tgps=..., ...)` or equivalent
- The strain is pure Gaussian noise (different realization each time if you run multiple calibration runs)

Example (cogwheel API):

```python
event_data_noise = EventData.gaussian_noise(
    "",
    detector_names=event_data_real.detector_names,
    duration=event_data_real.duration,
    tgps=event_data_real.tgps,
    asd_funcs=...,  # same as real run
    fmax=event_data_real.fmax,
)
```

### 3. Load from Rundir, Swap Data, Compute Evidence

- Load par_dic_0, extrinsic samples, bank indices, bank folders, etc. from rundir
- Re-run the likelihood computation with `event_data_noise` (**no** `inference.run()`)
- Accumulate the posterior contribution → ln_evidence

Entry point:

```python
ln_evidence = compute_evidence_on_noise(rundir, event_data_noise)
```

### 4. Optional: Multiple Noise Realizations

- Generate N independent `event_data_noise` realizations
- Call `compute_evidence_on_noise(rundir, event_data_noise_i)` for each

---

## Difference from Regular Inference: Samples Not Needed

Unlike a regular inference run (where individual samples matter for posteriors, corner plots, etc.), evidence calibration only needs the **grand-sum**—the log evidence. We do not need to keep the samples themselves. Just accumulate the posterior contribution (e.g. `logsumexp(ln_posterior)`). Light samples or other per-sample outputs that must be accumulated for normal runs are not important for this mode.

---

## Implementation Scope

| Component | Responsibility |
|-----------|----------------|
| **Dot-pe** | New entry point `compute_evidence_on_noise(rundir, event_data_noise)`. Loads par_dic_0 and extrinsic samples from rundir. Coherent/likelihood objects accept par_dic_0 as given and compute relative binning weights only—**no optimization**. Returns ln_evidence. Does not call `inference.run()`. |
| **Caller** | Create Gaussian noise EventData (same metadata as real event), call `compute_evidence_on_noise(rundir, event_data_noise)`. |

---

## Outputs

- `ln_evidence` (returned by `compute_evidence_on_noise`; optionally save to file)

---

## Checklist for Caller

1. Have a completed rundir from real-event PE
2. Create `EventData` with Gaussian noise (same detectors, PSD, duration, tgps as real event)
3. Call `compute_evidence_on_noise(rundir, event_data_noise)` — **not** `inference.run()`
