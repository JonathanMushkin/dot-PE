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
python test_data/setup.py --n-pool 4 --base-dir artifacts/banks
```

This creates:

```
artifacts/banks/
├── event/tutorial_event.npz    — gaussian noise, HLV, IMRPhenomXPHM injection
├── bank_small/                 — 4 096 (2^12) samples  (fast smoke-test)
└── bank_large/                 — 262 144 (2^18) samples  (timing benchmark)
```

Re-running the script is safe — each artifact is skipped if already present.
The event and banks are shared by both `MP/` and `lsf_swarm/` benchmarks.

---

### Running benchmarks

All three modes (serial, MP, swarm) are submitted via a single script that
handles LSF queue selection, memory sizing, and output directory naming:

```bash
python experiments/run_experiment.py --mode <serial|mp|swarm> \
    --bank <small|large> --n-ext <N> [--n-workers <W>]
```

To run all phases unattended inside a tmux session:

```bash
tmux new -s benchmark
bash experiments/auto_run.sh --skip-banks
# Ctrl+B D to detach
```

`auto_run.sh` runs Phases A–D sequentially, waits for each job, and appends
results to `experiments/PROGRESS.md`. See that file for the full phase
definitions and session history.

After jobs complete, compare results:

```bash
python experiments/compare.py
```

Experiment outputs land in timestamped subdirectories under `artifacts/experiments/`.

---

### Parallelization modes

| Mode | Entry point | Description |
|------|-------------|-------------|
| `serial` | `dot_pe/inference.py` | Single-process baseline |
| `mp` | `MP/run_mp.py` | `multiprocessing.Pool`, single node |
| `swarm` | `lsf_swarm/run_swarm.py` | LSF job array, multi-node |

The swarm orchestrator is resumable — if it fails, resubmit the same command
and completed stages (marked by `stage_N.done` files) are skipped.

## Reference

- dot-PE: Sampler-free gravitational wave inference using matrix multiplication. [https://journals.aps.org/prd/abstract/10.1103/vqj2-7qpz](https://journals.aps.org/prd/abstract/10.1103/vqj2-7qpz). [https://arxiv.org/abs/2507.16022](https://arxiv.org/abs/2507.16022).

## License

This project is licensed under the GNU General Public License v3.0 – see the [LICENSE](LICENSE) file for details. 

