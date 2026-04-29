# Multi-bank `prob_samples`: retention budget and bank-boundary artifacts

Handoff for **dot-PE** maintainers. Describes why multi-bank runs can show **spurious structure at bank edges** in combined posteriors, how that ties to **`size_limit`**, and the **simplest upstream fix**.

## Summary

In multi-bank MP inference, each bank’s coherent stage **caps** retained rows at **`size_limit`** by keeping the **best** rows by **`bestfit_lnlike`** (e.g. `argpartition` in `mp_inference.py`). Those per-bank tables are **concatenated** in `aggregate_and_save_results`, then—if the concat exceeds **`size_limit`**—trimmed to the **global** top **`size_limit`** rows by **`bestfit_lnlike`** (same ranking rule as per-bank). So **`rundir/prob_samples.feather`** has **at most** **`size_limit`** rows, with one implicit global cutoff on coherent hits for that artifact. **Previously**, without that post-merge cap, the merged file could hold up to **`K × size_limit`** rows and **min** `bestfit_lnlike` **differed by bank**, which could make mass / \(q\) mixtures look **asymmetric** at bank edges even when **IS weights and `log_prior_weights` are correct** — the issue was **which rows exist**, not softmax algebra alone.

## What the code does today (MP path)

- **Per bank:** While merging worker blocks, if `accumulated` rows exceed `size_limit`, keep only the **`size_limit` largest `bestfit_lnlike`** (`dot_pe/mp_inference.py`, coherent pool loop).
- **Across banks:** `inference.aggregate_and_save_results` reads each `banks_dir/<bank_id>/prob_samples.feather`, **`pd.concat`**, assigns `ln_weight_unnormalized` from per-bank `N_k`, then if **`len(combined) > size_limit`** keeps only the **`size_limit`** largest **`bestfit_lnlike`** (`nlargest`), recomputes **`weights`**, and writes **`rundir/prob_samples.feather`**.

So **`size_limit` is currently “per bank”** for what gets written under `banks_dir/`, and the **merged** file inherits **up to `K × size_limit`** rows.

## Observable symptom

- **1D / 2D features aligned with bank edges** in \((\mathcal{M}_c, q)\) (or related mass parameters).
- **Not** fixed by recomputing weights from `lnl_marginalized` + intrinsic `log_prior_weights` alone: stored vs “corrected” panels can **match** and **both** still show the jump (see `pipeline/sessions/20260428_q_mchirp_multibank_combination/`).

Per-bank **min** of `bestfit_lnlike` in stored `prob_samples` is a useful diagnostic (large spread across banks \(\Rightarrow\) stronger effect).

## Combinatorics (why a post-merge cap is coherent)

Fix banks \(k=1,\ldots,K\). Suppose each bank’s coherent stage ends with a **finite multiset** of scores and each bank keeps exactly the **top `M`** by score (same `M` as `size_limit`). Concatenate those \(K\) top-\(M\) multisets (\(K M\) rows). If you then keep the **top `M`** scores over that concat, you get the **same** multiset as: concatenate **all** coherent rows from all banks, then take the **top `M`** globally. *Proof sketch:* any row not in its bank’s top `M` is beaten by `M` rows **in that bank alone**, so it cannot belong to the global top `M` over the full union.

So: **per-bank top `M` → concat → global top `M`** is the same retained set as **one merged pool → global top `M`**, when both steps are literally “largest `M` by `bestfit_lnlike`” on the **saved numbers**.

## Implemented fix inside dot-PE

**Post-merge cap** in `aggregate_and_save_results`: after building `combined_prob_samples` and setting `ln_weight_unnormalized` per bank, if `len(combined_prob_samples) > size_limit`, keep only the **`size_limit` rows with largest `bestfit_lnlike`**, then set **`weights`** via `exp_normalize` on the trimmed `ln_weight_unnormalized` and write `prob_samples.feather`.

Effects:

- **`size_limit` is a property of the merged table** (at most `size_limit` rows in `rundir/prob_samples.feather`), not “`size_limit` per bank × `K`” for that artifact.
- **One implicit global cutoff** on retained coherent hits for the merged file, matching the combinatorics above when per-bank caps use the same `M`.

The per-bank coherent loop **keeps** its cap **for memory** during streaming; with the **same** `M`, this final global top-`M` on the concat of per-bank top-`M` tables matches the global top-`M` on the union of all coherent rows. Optionally simplify later by relying only on the merge-time trim if RAM allows.

## Post-hoc mitigation (outside the package)

On already-written `prob_samples.feather`, a **global floor**

\[
T_{\mathrm{global}} = \max_k \bigl(\min_j \mathrm{bestfit\_lnlike}_{j,k}\bigr)
\]

with discard + renormalize **aligns** floors across banks for diagnostics; it **does not** recover rows dropped before export. Implementation: `pipeline/sessions/20260428_mp_max_size_problem/run_post_mortem_sampling.py` → `post_mortem_sampling/` next to a run.

## Other options

- **Raise `size_limit`** so truncation is rare.
- **Document** for consumers: top-level `rundir/prob_samples.feather` is capped at **`size_limit`** rows; per-bank `banks_dir/<id>/prob_samples.feather` can still hold up to **`size_limit`** each during the run.

## References in this repo

- `pipeline/sessions/20260428_q_mchirp_multibank_combination/context.md` — examples, weighting checks, figures.
- Upstream: `dot_pe/mp_inference.py` (per-bank accumulation), `dot_pe/inference.py` (`aggregate_and_save_results`). Search **`size_limit`**, **`bestfit_lnlike`**, **`argpartition`**.
