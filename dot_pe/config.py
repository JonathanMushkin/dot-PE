"""
Default configurations for sampler-free inference.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from cogwheel.data import EventData
from cogwheel.waveform_models.xode import waveform

DEFAULT_FBIN_PATH = Path(__file__).parent / "fbin.npy"
APPROXIMANT = "IMRPhenomXODE"
# relative binning frequency grid generated from injection event data
# with pn_phase_tol = 0.05.
EVENT_DATA_KWARGS = {
    "detector_names": "HLV",
    "duration": 120.0,
    "asd_funcs": ["asd_H_O3", "asd_L_O3", "asd_V_O3"],
    "tgps": 0.0,
    "fmax": 1600.0,
}


def fbin_from_event_data(event_data, pn_phase_tol=0.05):
    """
    generate a relative binning frequency grid from event_data.
    Copied from cogwheel/likelihood/relative_binning.BaseRelativeBinning
    @pn_phase_tol.setter, @fbin.setter
    """
    pn_exponents = [-5 / 3, -2 / 3, 1]
    if waveform.APPROXIMANTS[APPROXIMANT].tides:
        pn_exponents.append(5 / 3)
    pn_exponents = np.array(pn_exponents)

    pn_coeff_rng = (
        2
        * np.pi
        / np.abs(np.subtract(*event_data.fbounds[:, np.newaxis] ** pn_exponents))
    )

    f_arr = np.linspace(*event_data.fbounds, 10000)

    diff_phase = np.sum(
        [
            np.sign(exp) * rng * f_arr**exp
            for rng, exp in zip(pn_coeff_rng, pn_exponents)
        ],
        axis=0,
    )
    diff_phase -= diff_phase[0]  # Worst case scenario differential phase

    # Construct frequency bins on arbitrary grid
    nbin = np.ceil(diff_phase[-1] / pn_phase_tol).astype(int)
    diff_phase_arr = np.linspace(0, diff_phase[-1], nbin + 1)
    fbin_raw = np.interp(diff_phase_arr, diff_phase, f_arr)
    # find frequencies on the grid defined by event_data
    fbin_ind = np.unique(
        np.searchsorted(event_data.frequencies, fbin_raw - event_data.df / 2)
    )
    fbin_on_grid = event_data.frequencies[fbin_ind]
    return fbin_on_grid


if DEFAULT_FBIN_PATH.exists():
    DEFAULT_FBIN = np.load(DEFAULT_FBIN_PATH)
else:
    DEFAULT_FBIN = fbin_from_event_data(
        EventData.gaussian_noise("", **EVENT_DATA_KWARGS)
    )
    np.save(DEFAULT_FBIN_PATH, DEFAULT_FBIN)


DEFAULT_F_REF = 50.0

harmonic_modes = waveform.APPROXIMANTS[APPROXIMANT].harmonic_modes

_harmonic_modes_by_m = defaultdict(list)
for l, m in harmonic_modes:
    _harmonic_modes_by_m[m].append((l, m))
M_ARR = np.fromiter(_harmonic_modes_by_m.keys(), dtype=int)

# f_ref should depend on the relevnat masses.
# for example, for the masses used here, f_merger could be below 100,
# so using f_ref = 100 could have side effects.
# Although we did not verify this, we used f_ref=50.0 for high-mass
# injections

DEFAULT_PARAMS_DICT = {
    "l1": 0.0,
    "l2": 0.0,
    "d_luminosity": 1.0,
    "phi_ref": 0.0,
    "f_ref": DEFAULT_F_REF,
}
