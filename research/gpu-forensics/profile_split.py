"""CPU-seconds inside the coherent and incoherent stages, by kind of work (P1, measure 4).

Reads `profile_output.prof` (aggregated over the MP workers) of the given rundirs.
Usage: python research/gpu-forensics/profile_split.py RUNDIR [RUNDIR ...]
"""

import pstats
import sys
from pathlib import Path

# (label, function name, file suffix or None); cumulative CPU-seconds.
ITEMS = [
    ("coherent total", "run_thin_iblock", "thin_coherent.py"),
    ("  <d|h>,<h|h> on the phase grid", "get_dh_hh_ieo", "likelihood_calculating.py"),
    ("  distance-marginalization lookup", "lnlike_marginalized", "lookup_table.py"),
    ("  DataFrame construction", "__init__", "pandas/core/frame.py"),
    ("  argpartition (top-k)", "argpartition", "fromnumeric.py"),
    ("incoherent total", "_incoherent_chunk_worker_inner", "mp_inference.py"),
    ("  single-detector kernel", "get_response_over_distance_and_lnlike", "single_detector.py"),
    ("  waveform reads (load_amp_and_phase)", "load_amp_and_phase", "sample_processing.py"),
    ("  linear-free dt fit", "get_relative_linfree_dt_from_waveform", "sample_processing.py"),
]


def main():
    for rd in sys.argv[1:]:
        st = pstats.Stats(str(Path(rd) / "profile_output.prof")).stats
        tot = sum(v[2] for v in st.values())
        print(f"\n{Path(rd).parent.name}/{Path(rd).name}: {tot:,.0f} CPU-s in total (tottime sum)")
        for label, fn, suffix in ITEMS:
            c = sum(v[3] for k, v in st.items() if k[2] == fn and (suffix is None or k[0].endswith(suffix)))
            print(f"  {label:40s} {c:10,.0f}")
        own = sum(v[2] for k, v in st.items() if k[2] == "run_thin_iblock")
        print(f"  {'  element-wise numpy inside run_thin_iblock':40s} {own:10,.0f}")


if __name__ == "__main__":
    main()
