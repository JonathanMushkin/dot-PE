# Multiprocessing Plan

## Hypothesis
Python `multiprocessing.Pool` with `OMP_NUM_THREADS=1` per worker is fundamentally
different from BLAS threading (which the user already tested). Independent processes
with separate working sets may avoid the memory-bus saturation seen at 8 BLAS threads.
This is untested and the goal of this implementation.

## Approach
Single LSF job (`bsub -n N -R "span[hosts=1]"`), or run interactively on a node.
Use `multiprocessing.Pool(n_workers)` to parallelize the two heavy loops.
Everything else — setup, extrinsic sampling, postprocess — stays serial and unchanged
(delegated directly to `dot_pe.inference`).

**No changes to `dot_pe/`.** All code is in `MP/`.

---

## Files

```
MP/
├── PLAN.md          (this file)
└── run_mp.py        single entry point — drop-in replacement for inference.run()
```

That's it. One file.

---

## Design

`run_mp.py` exposes a `run(...)` function with the same signature as `inference.run()`.
It replaces the two parallelizable stages with `Pool.imap_unordered`; everything else
is a direct call to the existing `inference` module functions.

### Stage 1 — Setup (unchanged)
```python
ctx = inference.prepare_run_objects(...)
```

### Stage 2 — Incoherent (parallelized)

Serial code in `collect_int_samples_from_single_detectors`:
```python
for batch_start in tqdm(range(0, n_int, blocksize)):
    for d, det_name in enumerate(detectors):
        result = run_for_single_detector(...)   # h_impb reused across detectors
```

MP replacement: split `range(0, n_int, blocksize)` into `n_workers` chunks.
Each worker processes its chunk of batches serially (preserving the h_impb reuse
across detectors within each batch). Workers return `(inds_chunk, lnlike_di_chunk)`.
Main process concatenates and applies threshold identically to serial code.

```python
def _score_chunk(args):
    # args: (chunk_start, chunk_end, event_data, par_dic_0, bank_folder,
    #        fbin, approximant, n_phi, blocksize, m_arr, n_t)
    # Iterates over batches within this chunk, reusing h_impb per batch
    # Returns (inds, lnlike_di)

with Pool(n_workers, initializer=_init_worker) as pool:
    results = pool.map(_score_chunk, chunk_args)
```

`_init_worker` sets `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

### Stage 3 — Threshold + cross-bank selection (unchanged)
```python
inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood(...)
```

### Stage 4 — Extrinsic sampling (unchanged, serial)
```python
inference.draw_extrinsic_samples(...)
```

### Stage 5 — Coherent (parallelized)

Serial code in `create_likelihood_blocks`:
```python
for i_block in i_blocks:
    load waveforms for i_block → h_impb
    for e_block in e_blocks:
        create_a_likelihood_block(h_impb, response[..., e_block], ...)
        combine_prob_samples_with_next_block()
```

MP replacement: one worker per i_block. Each worker loads its own waveforms, iterates
over all e_blocks, accumulates a local `prob_samples` DataFrame + discarded counters.
Returns `(prob_samples_chunk, n_discarded, logsumexp_discarded, logsumsqrexp_discarded,
cached_dt_dict)`.
Main process merges all worker results by calling `combine_prob_samples_with_next_block`
sequentially (fast, serial merge of already-reduced DataFrames).

```python
def _coherent_iblock(args):
    # args: (i_block, e_blocks, bank_file, waveform_dir, n_phi, m_arr,
    #        likelihood_linfree, rundir, max_bestfit_lnlike_diff, size_limit,
    #        intrinsic_logw_lookup)
    # Returns (clp.prob_samples, discarded stats, cached_dt)

with Pool(n_workers, initializer=_init_worker) as pool:
    results = pool.map(_coherent_iblock, iblock_args)
# merge results into final prob_samples
```

### Stage 6 — Postprocess (unchanged)
```python
inference.aggregate_and_save_results(...)
```

---

## Key design decisions

### fork() shares read-only data for free
On Linux, `Pool` uses `fork()`. The parent process holds event_data, weight matrices,
response/timeshift arrays in memory. Child processes inherit these via copy-on-write —
no serialization, no redundant copies in RAM as long as workers don't write to them.
This is critical for the coherent phase where `dh_weights_dmpb` (~6 MB) and
`hh_weights_dmppb` (~26 MB) are the same for all i_blocks.

### OMP=1 per worker, set before fork
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```
Set in main process BEFORE `Pool()` is created, so all forked workers inherit them.
This is the key difference from what the user already tested (BLAS threading): workers
are truly single-threaded, no internal thread contention.

### Incoherent: chunk-level parallelism, not batch-level
Workers process chunks of consecutive batches (not one batch each) to amortize Python
overhead and keep each worker's working set in cache. Chunk size ≈ n_batches / n_workers.

### Coherent: one worker per i_block
Matches the natural structure of the serial code. Waveforms are loaded once per i_block
within the worker. No state shared between workers.

### No files between stages
Workers return results to the main process directly via `pool.map`. No intermediate
`.npz` files, no polling, no filesystem coordination. Results live in RAM until the
main process writes the final outputs (same as serial code).

### n_workers
Default: `min(n_cpus_available, len(i_blocks))` for coherent;
`min(n_cpus_available, 16)` for incoherent. User-overridable via `--n-workers`.

---

## Usage

```bash
# On the server, single node:
bsub -q physics-medium -n 16 -R "span[hosts=1]" \
     -o run_%J.out -e run_%J.err \
     python MP/run_mp.py \
         --event /path/to/event.npz \
         --bank /path/to/bank \
         --rundir /path/to/output \
         --n-ext 4096 --n-phi 100 --n-workers 16
```

Or interactively on a node with 16+ cores:
```bash
python MP/run_mp.py --event ... --bank ... --n-workers 16
```

---

## Expected outcome

**If memory bandwidth is the bottleneck** (same DRAM bus for all workers):
- Will plateau similarly to BLAS threading; no improvement over 8 workers
- Conclusion: LSF multi-node is necessary

**If BLAS threading overhead was the bottleneck** (cache thrashing, not DRAM):
- Will scale beyond 8 workers; ~16x speedup on incoherent, ~N_i_blocks speedup on coherent
- Conclusion: single-node multiprocessing is sufficient

The empirical test on the server will distinguish these cases.

---

## What this does NOT address
- I/O from NFS (waveform loading); if NFS is the bottleneck, neither approach works well
  on a single node — only multi-node (LSF swarm) distributes NFS load
- Extrinsic sampling (~549 s); stays serial in both approaches
