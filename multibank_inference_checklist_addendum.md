# Cursor Addendum: Checklist + Code Navigation Pointers (inference.py)

This file is intended to be pasted alongside the main instructions to keep
Cursor focused and reduce ambiguity in a large file with similarly named
functions.

## 10-line implementation checklist (use this to prevent scope creep)
1. Do not refactor core logic; implement multi-bank via an outer loop in `run(...)`.
2. Keep single-bank behavior identical (paths, formats, weights) when only one bank is passed.
3. Parse `bank_folder` into a list; if length==1, run the original code path unchanged.
4. In multi-bank mode, run per-bank inference into `rundir/banks/<bank_id>/` without changing per-bank internals.
5. Record `N_k` per bank (the intrinsic normalization used for that bank’s coherent run).
6. Combine log-evidence using: `lnZ_total = logsumexp(lnZ_k + log(N_k)) - log(sum_k N_k)`.
7. Combine `prob_samples` across banks by recomputing global weights from `ln_posterior - log(N_k)`.
8. Add `bank_id` only in multi-bank outputs; do not pollute single-bank outputs.
9. Update `postprocess(...)` and `standardize_samples(...)` to accept intrinsic/cache per bank keyed by `bank_id`.
10. Add fail-fast compatibility checks across `bank_config.json` (approximant/fbin/f_ref/m_arr as used).

## Navigation pointers (function locations in the current inference.py)
These line numbers refer to the uploaded `inference.py` at the time of writing.

- `run(...)` starts at approximately **line 650**  
  - This is where to add:
    - bank parsing
    - multi-bank outer loop
    - combined evidence/sample aggregation
  - Current single-bank call to `run_coherent_inference(...)` occurs around **line 877**.

- `postprocess(...)` starts at approximately **line 531**  
  - This is where to:
    - detect `bank_id` in `prob_samples`
    - load intrinsic banks and caches per bank into dicts
    - call `standardize_samples(...)` with dicts in multi-bank mode

- `standardize_samples(...)` starts at approximately **line 382**  
  - This is where to:
    - keep the existing path when `bank_id` column is absent
    - add a grouped-by-bank selection path when `bank_id` exists
    - lookup cached timing by `(bank_id, i)` in multi-bank mode

- `collect_int_samples_from_single_detectors(...)` starts at approximately **line 180**  
  - Prefer not to touch this for minimality.
  - In multi-bank mode, call it once per bank from within `run(...)`.

- `run_coherent_inference(...)` starts at approximately **line 275**  
  - Prefer not to touch this for minimality.
  - In multi-bank mode, call it once per bank and read its outputs.

## Minimal data contract changes (multi-bank only)
- Combined `rundir/prob_samples.feather` must include:
  - `bank_id` (string)
  - `ln_weight_global` (float) — optional but recommended for debugging
  - `weights` recomputed globally
- Per-bank `prob_samples.feather` inside each bank subdir can remain unchanged.

## “Stop and warn” conditions (add explicit errors/warnings)
- Error if bank configs mismatch on waveform-critical keys (used globally in `run(...)`).
- Warning if any bank returns `N_k <= 0` or missing.
- Warning if multi-bank mode is used with obviously overlapping coverage (if detectable cheaply); do not attempt dedup in this patch.

