![dot-pe logo](dot-pe-logo-white-bg.png)

## Purpose

DOT-PE is a Python package for parameter estimation and evidence integration using data from the gravitational wave interferometer observatories LIGO, Virgo, and KAGRA. Unlike traditional approaches that rely on stochastic samplers, DOT-PE performs parameter estimation and evidence integration using matrix multiplications for fast likelihood evaluation. All interfacing with gravitational wave data, waveform generation, and sampling tools is handled through the [`cogwheel`](https://github.com/jroulet/cogwheel) package.

## Installation

Clone the repository:
```bash
git clone https://github.com/JonathanMushkin/dot-PE.git
cd dot-PE
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dot-pe
```

## Usage

See the notebooks in the `notebooks/` directory for examples.

## Parallelization (HPC benchmarking)

Two parallel implementations are under development in branch `lsf-swarm`:

| Approach | Location | Plan |
|---|---|---|
| Python `multiprocessing.Pool` (single node) | `MP/` | `MP/PLAN.md` |
| LSF job array (multi-node) | `lsf_swarm/` | `lsf_swarm/LSF-SWARM-PLAN.md` |

See `log.md` for background and design decisions.

### Test data setup

Create mock event data and two banks (small for correctness checks, large for
timing benchmarks) with a single idempotent script:

```bash
python test_data/setup.py --n-pool 4
```

This creates:

```
test_data/
├── event/tutorial_event.npz    — gaussian noise, HLV, IMRPhenomXPHM injection
├── bank_small/                 — 4 096 samples  (fast smoke-test)
└── bank_large/                 — 262 144 (2^18) samples  (timing benchmark)
```

Re-running the script is safe — each artifact is skipped if already present.
The event and banks are shared by both `MP/` and `lsf_swarm/` benchmarks.
The setup follows `notebooks/03_run_inference/03_run_inference.ipynb` exactly.

---

### MP (multiprocessing, single node)

`MP/run_mp.py` — drop-in for `inference.run()`.

**Smoke-test** (small bank, should finish in a few minutes):

```bash
python MP/run_mp.py \
    --event  test_data/event/tutorial_event.npz \
    --bank   test_data/bank_small \
    --rundir /tmp/mp_smoke \
    --n-ext 512 --n-phi 50 --n-workers 4
```

**Timing benchmark** (large bank, single LSF node):

```bash
bsub -q physics-medium -n 16 -R "span[hosts=1]" \
     -o mp_%J.out -e mp_%J.err \
     python MP/run_mp.py \
         --event  test_data/event/tutorial_event.npz \
         --bank   test_data/bank_large \
         --n-ext 4096 --n-phi 100 --n-workers 16
```

After the run, check `<rundir>/run_N/summary_results.json` and the printed
`Total wall-clock time` line.

---

### LSF swarm (multi-node)

`lsf_swarm/run_swarm.py` — orchestrator that submits short worker jobs.
Must itself run inside an LSF job (physics-medium) so it can call `bsub`.

**Smoke-test**:

```bash
bsub -q physics-medium -n 1 -W 60 \
     -o swarm_%J.out -e swarm_%J.err \
     python lsf_swarm/run_swarm.py \
         --event  test_data/event/tutorial_event.npz \
         --bank   test_data/bank_small \
         --rundir /tmp/swarm_smoke \
         --n-ext 512 --n-phi 50
```

**Timing benchmark** (large bank):

```bash
bsub -q physics-medium -n 1 -W 120 \
     -o swarm_%J.out -e swarm_%J.err \
     python lsf_swarm/run_swarm.py \
         --event  test_data/event/tutorial_event.npz \
         --bank   test_data/bank_large \
         --rundir /tmp/swarm_bench \
         --n-ext 4096 --n-phi 100
```

The run is resumable: if it fails, re-run the same command — completed stages
are skipped via `swarm_setup/stage_N.done` marker files.

---

### What to report after a test run

1. `n_workers` (MP) or `--max-concurrent` (swarm) used
2. **Total wall-clock time** — printed as `Total wall-clock time: X s` at end
3. Contents of `<rundir>/run_N/summary_results.json`
4. Any errors or warnings

## Reference

- dot-PE: Sampler-free gravitational wave inference using matrix multiplication. [https://journals.aps.org/prd/abstract/10.1103/vqj2-7qpz](https://journals.aps.org/prd/abstract/10.1103/vqj2-7qpz). [https://arxiv.org/abs/2507.16022](https://arxiv.org/abs/2507.16022).

## License

This project is licensed under the GNU General Public License v3.0 – see the [LICENSE](LICENSE) file for details. 

