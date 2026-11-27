"""Read throughput of bank files against reader threads (P3 gate 4 follow-up; CPU job).

Each thread count reads a disjoint set of `n_files` amplitude+phase file pairs (no page-cache
reuse within the job) with dot_pe_gpu.bank.iter_blocks, and reports GB/s.
Usage: python research/incoherent-gpu/read_bench.py BANK_FOLDER [n_files]
"""

import sys
import time

from dot_pe_gpu.bank import bank_layout, iter_blocks


def main(bank, n_files=24):
    _, _, nf = bank_layout(bank)
    start = 64  # skip files read most recently by other jobs
    for j, threads in enumerate((4, 8, 16, 32)):
        files = range(start + j * n_files, start + (j + 1) * n_files)
        assert files[-1] < nf
        t0 = time.perf_counter()
        nbytes = 0
        for _, _, amp, phase in iter_blocks(bank, files, threads=threads, prefetch=2 * threads):
            nbytes += amp.nbytes + phase.nbytes
        dt = time.perf_counter() - t0
        print(f"threads {threads:3d}: {nbytes / 1e9:.2f} GB in {dt:.1f} s = {nbytes / 1e9 / dt:.3f} GB/s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
