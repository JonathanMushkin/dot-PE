---
name: Dot-PE Evidence Calibration Support
overview: Add a mode to dot-pe inference that runs the same pipeline as `run()` but accepts a pre-specified reference waveform (par_dic_0) and skips the ReferenceWaveformFinder optimization. This enables pure-noise evidence calibration without fitting a waveform to noise.
todos: []
isProject: false
---

# Dot-PE: Support for Evidence Calibration (No-Optimization Mode)

## Handoff Context

This plan is for implementing changes in the **dot-pe** repository to support "pure-noise evidence calibration." The caller (e.g. o4a-pe-project or any PE pipeline) will provide:

- `EventData` (which may be Gaussian noise, not real strain)
- A fixed `par_dic_0` (reference waveform parameters)
- External extrinsic samples and bank indices from a previous run

The goal: run the same inference pipeline as `run()`, but **skip** any optimization that fits a waveform to the data. The reference waveform must be taken as given.

---

## Use Case: Pure-Noise Evidence Calibration

For evidence calibration, we run PE on Gaussian noise (no signal) using:

- The same banks and extrinsic samples as a real-event run
- A fixed reference waveform (e.g. median chirp mass from bank, q=1, chieff=0.5)
- New relative-binning weights computed from that fixed reference

The evidence from these noise runs is used to calibrate the Bayesian evidence. Fitting a waveform to noise would be wrong; we need a fixed reference.

---

## Current Behavior (Problem)

In `prepare_run_objects`, dot-pe calls:

```python
coherent_posterior = Posterior.from_event(
    event=event_data,
    mchirp_guess=mchirp_guess,
    ...
)
par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()
```

Cogwheel's `Posterior.from_event` uses `ReferenceWaveformFinder.from_event`, which **always** calls `find_bestfit_pars()` (except for injections with aligned spins). That method:

- Runs differential evolution over mchirp, eta, chieff
- Optimizes time, sky, inclination, phase, distance

So the reference waveform is always optimized on the data. For pure noise, this fits a waveform to noise.

---

## Required Changes

### 1. Cogwheel: Add `par_dic_0` bypass in ReferenceWaveformFinder

**File:** `cogwheel/likelihood/reference_waveform_finder.py`

In `ReferenceWaveformFinder.from_event`:

- Add optional parameter: `par_dic_0: dict | None = None`
- If `par_dic_0` is provided:
  - Construct the finder with that dict
  - **Do not call** `find_bestfit_pars()`
- If `par_dic_0` is None: keep current behavior (call `find_bestfit_pars()`)

The provided `par_dic_0` must have all required keys for the reference waveform (m1, m2, spins, iota, d_luminosity, sky, t_geocenter, f_ref, etc.). The caller is responsible for supplying a valid, self-consistent dict.

### 2. Cogwheel: Wire through Posterior.from_event

**File:** `cogwheel/posterior.py`

In `Posterior.from_event`:

- Add optional parameter: `par_dic_0: dict | None = None`
- Pass it to `ReferenceWaveformFinder.from_event(..., par_dic_0=par_dic_0)`

### 3. Dot-pe: Add `par_dic_0` support in prepare_run_objects

**File:** `dot_pe/inference.py`

In `prepare_run_objects`:

- Add parameter: `par_dic_0: dict | None = None`
- When `par_dic_0` is not None:
  - Pass it to `Posterior.from_event(..., par_dic_0=par_dic_0)`
  - Skip the usual ReferenceWaveformFinder optimization path (cogwheel will handle this)
- When `par_dic_0` is None: keep current behavior (mchirp_guess only)

### 4. Dot-pe: Expose `par_dic_0` in `run()` and `run_and_profile()`

**File:** `dot_pe/inference.py`

- Add `par_dic_0: dict | None = None` to the `run()` (and related) signature(s)
- Forward it to `prepare_run_objects`

---

## API Summary

```python
# New optional parameter in dot_pe.inference.run()
rundir = inference.run(
    event=event_data,           # Can be EventData from noise
    bank_folder=...,
    extrinsic_samples=...,      # Already supported
    mchirp_guess=...,           # Used only when par_dic_0 is None
    par_dic_0=par_dic_0,        # NEW: when provided, skip ref-wf optimization
    ...
)
```

When `par_dic_0` is provided:

- `mchirp_guess` is ignored for posterior creation (par_dic_0 takes precedence)
- Cogwheel skips `find_bestfit_pars`
- Relative binning uses the given reference waveform

---

## Caller Responsibility

The caller (e.g. o4a-pe-project) must supply a valid `par_dic_0` when using this mode. It should include at least:

- Intrinsic: m1, m2, s1z, s2z, in-plane spins (zero), iota, f_ref
- Extrinsic: d_luminosity, lat, lon, t_geocenter, phi_ref, psi

Example construction for "fixed reference" (median mchirp, q=1, chieff=0.5, no in-plane spins):

```python
m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp_median, eta_from_q(1))  # q=1 -> eta=0.25
# s1z, s2z for chieff=0.5 with q=1
# Plus extrinsic params (sky, time, phase, distance) - can use arbitrary valid values
```

---

## Testing

1. **Regression:** With `par_dic_0=None`, behavior is unchanged.
2. **New path:** With `par_dic_0` provided and noise EventData, inference runs without calling `find_bestfit_pars`. Check that no optimization printouts appear (e.g. "Searching incoherent solution...").
3. **Output:** Posterior.json, samples.feather, etc. should be produced as in a normal run.

---

## Dependencies

- Cogwheel changes must be merged (or available) before or together with dot-pe changes
- Dot-pe depends on cogwheel; the new `par_dic_0` argument is passed through to cogwheel's `Posterior.from_event`

