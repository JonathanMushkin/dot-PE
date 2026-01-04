# Minimal Multi-Bank Support for dot_pe/inference.py

## Objective
Add support for running inference over multiple banks and combining the results,
while keeping changes as local as possible (ideally confined to
`dot_pe/inference.py`). The existing single-bank workflow must continue to work
unchanged.

The multi-bank mode should:
- Run the existing single-bank inference independently for each bank
- Combine evidences correctly using Monte Carlo aggregation
- Combine posterior samples with correct global weights
- Allow intrinsic samples and caches to be bank-specific
- Remain readable, minimal, and backward-compatible

---

## Core principles (do not violate)

### Be careful
Do **not** assume the strategy is flawless. Add validation checks and raise
explicit warnings where assumptions could break (see “Hazards” section).

### Be minimal
The current code works. Do not redesign the architecture or rewrite core logic.
Prefer:
- outer loops over banks
- light branching based on single vs multi-bank
- reuse of existing functions and file formats

### Be transparent
Avoid deep wrapper classes or opaque abstractions. Favor explicit loops,
straightforward data structures (`list`, `dict`), and readable control flow.

### Backward compatibility
If the user provides a single bank:
- behavior must be identical to the current implementation
- output paths and formats should remain unchanged
- users should not need to learn about multi-bank at all

Multi-bank logic should activate automatically only when multiple banks are
detected.

---

## High-level strategy

Treat each bank as an **independent chunk of intrinsic Monte Carlo samples**.
Run the existing pipeline per bank, then combine the results as if they came from
one larger Monte Carlo draw.

This means:
- Do **not** merge intrinsic banks into one dataframe
- Do **not** modify waveform loading or coherent likelihood internals
- Combine results mathematically, not structurally

---

## Step-by-step implementation plan (inference.py only)

### 1. Accept multiple bank folders

Add a small helper function:

- Accept:
  - a single path-like (current behavior)
  - a list/tuple of paths
  - optionally a comma-separated string `"bankA,bankB,bankC"`
- Return a `List[Path]`

If only one bank is found, treat it as single-bank mode.

Assign each bank a stable `bank_id`:
- Prefer folder name, otherwise `bank0`, `bank1`, …
- `bank_id` must be written into outputs **only in multi-bank mode**

---

### 2. Validate bank compatibility (fail fast)

Before running inference:
- Load each bank’s `bank_config.json`
- Assert that critical waveform-related keys match across banks:
  - approximant
  - frequency grid / fbin
  - reference frequency
  - any other parameters already assumed global in the code

If incompatible:
- raise a clear error
- do **not** attempt auto-conversion or silent fallback

This is a guardrail, not a refactor.

---

### 3. Run the existing pipeline per bank

In `run(...)`:

#### Single-bank
- Keep current behavior unchanged

#### Multi-bank
- Create a subdirectory structure:
  ```
  rundir/
    banks/
      bankA/
      bankB/
      ...
  ```
- For each bank:
  1. Call `collect_int_samples_from_single_detectors(...)` with that bank
  2. Call `run_coherent_inference(...)` unchanged, writing into that bank’s subdir
  3. Record:
     - `lnZ_k` (log evidence)
     - `N_k` (number of intrinsic samples used for that bank)

Do **not** modify `run_coherent_inference` internals.

---

### 4. Combine evidences correctly

Each bank computes an evidence of the form:

```
lnZ_k = logsumexp(ln_posterior_k) - log(n_phi * n_ext * N_k)
```

To combine banks as a single Monte Carlo estimate:

```
N_total = sum_k N_k
lnZ_total = logsumexp( lnZ_k + log(N_k) ) - log(N_total)
```

Notes:
- Do **not** sum `lnZ_k` directly
- This aggregation assumes banks represent chunks of one larger MC draw

Store:
- `lnZ_total` as the main result
- optionally keep per-bank `lnZ_k` for diagnostics

---

### 5. Combine posterior samples with global weights

Each bank produces a `prob_samples.feather` normalized **within that bank**.
These weights are not globally correct and must be recomputed.

For each bank:
1. Load `prob_samples.feather`
2. Add:
   - `bank_id`
   - `ln_weight_global = ln_posterior - log(N_k)`

After concatenating all banks:
1. Normalize globally:
   ```
   weights = exp(ln_weight_global - max) / sum(exp(...))
   ```
2. Write combined output to:
   ```
   rundir/prob_samples.feather
   ```

Keep per-bank outputs in their subdirectories.

Backward compatibility:
- In single-bank mode, keep the old normalization logic and output schema

---

### 6. Intrinsic samples and caches per bank

Multi-bank postprocessing requires intrinsic data to be bank-aware.

#### In `postprocess(...)`
- Single-bank: keep current logic
- Multi-bank:
  - Load intrinsic banks into:
    ```
    intrinsic_samples_by_bank: dict[bank_id] -> DataFrame
    ```
  - Load cached linear-free timing into:
    ```
    cached_dt_by_bank: dict[bank_id] -> dict[int -> float]
    ```

Pass these dicts into `standardize_samples(...)`.

---

### 7. Update `standardize_samples(...)` minimally

Add support for both modes:

#### If `bank_id` NOT in samples
- Run existing code unchanged

#### If `bank_id` IS present
- For intrinsic parameters:
  - Group samples by `bank_id`
  - For each group, select rows via `.iloc[group["i"]]`
  - Reassemble in original order
- For cached timing:
  - Treat cache keys as `(bank_id, i)`
  - Safest approach:
    - build a dataframe of unique `(bank_id, i)` pairs
    - attach `dt_linfree` via lookup
    - merge back into samples

Avoid clever indexing tricks; clarity over micro-optimizations.

---

## Hazards and required warnings

### 1. Overlapping intrinsic coverage
If the same intrinsic point appears in multiple banks, posterior mass may be
duplicated.

Minimal action:
- Log a warning if overlap is detected or strongly implied
- Do **not** attempt deduplication or reweighting in this patch

### 2. Bank config incompatibility
Combining incompatible waveform models is invalid. Enforce strict checks.

### 3. Meaning of N_k
`N_k` must exactly match the intrinsic normalization used in that bank’s
coherent run. If subsampling or cuts change this, weights will be wrong.

### 4. Memory usage
Concatenating large `prob_samples` may be heavy. This patch prioritizes
correctness and simplicity over streaming or chunked I/O.

### 5. Downstream schema assumptions
Adding `bank_id` should not break code, but some users may assume exact columns.
Keep single-bank outputs unchanged where possible.

---

## Acceptance sanity checks

1. Single-bank run:
- Identical behavior and outputs as before
- Weights sum to 1

2. Multi-bank run:
- Combined `prob_samples.feather` has `bank_id`
- Weights sum to 1
- `lnZ_total` is finite

3. Evidence aggregation check:
- Two identical banks with equal `N_k` should yield
  `lnZ_total ≈ lnZ_k` (not `lnZ_k + log(2)`)

---

## Final note
This patch intentionally prioritizes:
- correctness
- minimal surface area
- readability

It does **not** attempt to solve intrinsic overlap, proposal optimization, or
bank unification. Those belong in a future, explicit redesign.
