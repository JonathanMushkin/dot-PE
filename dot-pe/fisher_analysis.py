"""
This module provides tools for performing Fisher matrix and overlap analyses for gravitational wave (GW) parameter estimation using the gwfast and cogwheel frameworks.

Dependencies
------------
- numpy
- matplotlib
- gwfast (waveforms, signal, network, utils, globals, fisherTools)
- cogwheel (utils, gw_utils)
- pathlib
- copy
- sys

Typical Usage
-------------
1. Prepare GW parameters in either cogwheel or gwfast format.
2. Optionally convert parameters and add extrinsic parameters.
3. Get parameters and detector names from COGWHEEL event data.
4. Call `fisher_and_overlap_analysis` to compute Fisher matrix and overlaps.
5. Use `plot_overlap_scan` to visualize the overlap as a function of chirp mass.


Example
---------
cogwheel_par_dic = {
    "m1": 10,
    "m2": 10,
    "s1z": 0.1,
    "s1x_n": 0.0,
    "s1y_n": 0.0,
    "s2z": 0.1,
    "s2x_n": 0.0,
    "s2y_n": 0.0,
    "iota": 0.5,
}
output_dict = fisher_analysis.fisher_and_overlap_analysis(
    cogwheel_par_dic,
    detector_names=["H", "L"],
    wf_model=None,
    coghweel_params=True,
    add_arb_extrinsic_params=True,
)

par_dic = output_dict["par_dic"]
Mc_arr = output_dict["Mc_arr"]
overlaps = output_dict["overlaps"]
sigma_Mc = output_dict["sigma_Mc"]
fig = fisher_analysis.plot_overlap_scan(
    par_dic["Mc"][0], Mc_arr, overlaps, sigma_Mc
)
fig.legend()
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from gwfast.waveforms import WaveFormModel
from pathlib import Path
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast.waveforms import IMRPhenomHM
from gwfast import gwfastUtils
from gwfast import gwfastGlobals
from gwfast import fisherTools

from cogwheel import data
from cogwheel.gw_utils import m1m2_to_mchirp, q_to_eta


def get_detectors(det_names=["H", "L", "V"]):
    alldetectors = copy.deepcopy(gwfastGlobals.detectors)
    detnames_dict = dict(H="H1", L="L1", V="Virgo")
    detnames_dict = {k: detnames_dict.get(k) for k in det_names}

    psd_paths_dict = {
        k: str(next(Path(gwfastGlobals.detPath).glob(f"LVC_O1O2O3/O3-{k}*")))
        for k, v in detnames_dict.items()
    }
    detectors = {k: alldetectors[v] for k, v in detnames_dict.items()}

    for k, v in psd_paths_dict.items():
        detectors[k]["psd_path"] = v
    return detectors


def cogwheel_to_gwfast_params(cogwheel_params: dict) -> dict:
    """
    gwFast params: Mc,dL,theta,phi,iota,psi,eta,Phicoal,chi1z,chi2z,
    Cogwheel params : m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n,

    "fmin": np.array([20.0]),
    "fmax": np.array([2048.0]),
    """
    m1, m2 = cogwheel_params["m1"], cogwheel_params["m2"]
    mchirp = m1m2_to_mchirp(m1, m2)
    eta = q_to_eta(m2 / m1)
    s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n = [
        cogwheel_params.get(k)
        for k in ["s1z", "s2z", "s1x_n", "s1y_n", "s2x_n", "s2y_n"]
    ]
    iota = cogwheel_params["iota"]
    gwfast_params = dict(
        Mc=mchirp,
        eta=eta,
        chi1z=s1z,
        chi1x=s1x_n,
        chi1y=s1y_n,
        chi2z=s2z,
        chi2x=s2x_n,
        chi2y=s2y_n,
        iota=iota,
    )
    for k, v in gwfast_params.items():
        if type(v) is not np.ndarray:
            gwfast_params[k] = np.atleast_1d(np.array(v))
    return gwfast_params


def add_arb_ext_params(gwfast_params: dict, inplace: bool = False):
    """
    add sky position (theta,phi), polarizaiton angle psi, and coalescence phase Phicoal,
    """
    if not inplace:
        gwfast_params = copy.copy(gwfast_params)
    gwfast_params["theta"] = 1.0
    gwfast_params["phi"] = 1.0
    gwfast_params["dL"] = 1.0  # in Gpc
    gwfast_params["Phicoal"] = 0.0
    gwfast_params["psi"] = 0.0
    tGPS = 0.0
    gwfast_params["tcoal"] = gwfastUtils.GPSt_to_LMST(tGPS, 0.0, 0.0)
    for k in ["theta", "phi", "dL", "Phicoal", "psi", "tcoal"]:
        if type(gwfast_params[k]) is not np.ndarray:
            gwfast_params[k] = np.atleast_1d(np.array(gwfast_params[k]))
    if not inplace:
        return gwfast_params


def fisher_and_overlap_analysis(
    par_dic: dict,
    detector_names: list = ["H", "L", "V"],
    wf_model: WaveFormModel = None,
    fmin: float = 20.0,
    fmax: float = 2048.0,
    n_overlaps: int = 1001,
    snr_target: float = 8,
    coghweel_params: bool = False,
    add_arb_extrinsic_params: bool = False,
    fisher_matrix_kwargs: dict = {},
    overlap_kwargs: dict = {},
) -> tuple:
    """
    Perform Fisher matrix and overlap analysis for a given set of GW parameters and detectors.

    Parameters
    ----------
    par_dic : dict
        Dictionary of GW parameters (e.g., 'Mc', 'eta', 'chi1z', etc.).
    detector_names : list, optional
        List of detector short names to use (default: ["H", "L", "V"]).
    wf_model : WaveFormModel, optional
        Waveform model to use. If None, uses IMRPhenomHM.
    fmin : float, optional
        Minimum frequency for analysis (default: 20.0).
    fmax : float, optional
        Maximum frequency for analysis (default: 2048.0).
    n_overlaps : int, optioanl
        Number of chirp mass values to scan over (default: 1001).
    res : int, optional
        Resolution for frequencies in overlap calculation (default: 5000).
    snr_target : float, optional
        Target SNR for rescaling luminosity distance (default: 8).
    coghweel_params : bool, optional
        If True, convert parameters from cogwheel to gwfast format.
    add_arb_extrinsic_params : bool, optional
        If True, add arbitrary extrinsic parameters.

    Returns
    -------
    tuple
        (fisher_matrix, cov, err, sigma_Mc, Mc_arr, overlaps)
        - fisher_matrix: Fisher information matrix after removing sky position.
        - cov: Covariance matrix.
        - err: Error estimates.
        - sigma_Mc: Standard deviation of chirp mass.
        - Mc_arr: Array of chirp mass values scanned.
        - overlaps: Overlap values for the chirp mass scan.
    """

    if coghweel_params:
        par_dic = cogwheel_to_gwfast_params(par_dic)
    if add_arb_extrinsic_params:
        par_dic = add_arb_ext_params(par_dic, inplace=False)
    # Use global wf_model if not provided
    if wf_model is None:
        wf_model = IMRPhenomHM()
    if overlap_kwargs is None:
        overlap_kwargs = {}
    if fisher_matrix_kwargs is None:
        fisher_matrix_kwargs = {}
    # Prepare detectors
    signals = {}
    detectors = get_detectors(det_names=detector_names)
    for d in detector_names:
        det = detectors[d]
        signals[d] = GWSignal(
            wf_model,
            psd_path=det["psd_path"],
            detector_shape=det["shape"],
            det_lat=det["lat"],
            det_long=det["long"],
            det_xax=det["xax"],
            fmin=fmin,
            fmax=fmax,
            verbose=False,
            useEarthMotion=False,
            IntTablePath=None,
        )
    det_network = DetNet(signals)

    # Rescale dL to target SNR if needed
    snr = det_network.SNR(par_dic)[0]
    if snr_target is not None and snr > 0:
        par_dic = par_dic.copy()
        par_dic["dL"] = snr * par_dic["dL"] / snr_target

    # Fisher matrix and covariance
    raw_fisher_matrix = det_network.FisherMatr(par_dic, **fisher_matrix_kwargs)
    ParNums = wf_model.ParNums
    fisher_matrix, ParNums = fisherTools.fixParams(
        raw_fisher_matrix, ParNums, ["theta", "phi"]
    )

    cov, err = fisherTools.CovMatr(fisher_matrix)
    Mc_Num = ParNums["Mc"]
    sigma_Mc = cov[Mc_Num, Mc_Num, 0] ** 0.5

    if np.isnan(sigma_Mc) or (sigma_Mc > par_dic["Mc"] / 5):
        if par_dic["Mc"] > 10:
            scale = (par_dic["Mc"] / 10) ** (5 / 3) / 5
        else:
            scale = (par_dic["Mc"] / 10) ** (8 / 3) / 5
    else:
        scale = sigma_Mc

    Mc_arr = par_dic["Mc"][0] + np.linspace(-5, 5, n_overlaps) * scale
    test_events = {k: v.repeat(n_overlaps) for k, v in par_dic.items()}
    test_events["Mc"] = Mc_arr
    ref_events = {k: v.repeat(n_overlaps) for k, v in par_dic.items()}
    overlaps = det_network.WFOverlap(
        wf_model, wf_model, ref_events, test_events, **overlap_kwargs
    )
    output = {
        "fisher_matrix": fisher_matrix,
        "cov": cov,
        "err": err,
        "sigma_Mc": sigma_Mc,
        "Mc_arr": Mc_arr,
        "overlaps": overlaps,
        "par_dic": par_dic,
        "ParNums": ParNums,
    }
    return output


def plot_overlap_scan(Mc, Mc_arr, overlaps, sigma_Mc=None):
    """
    Plot the overlap scan and return the matplotlib figure object.

    Parameters
    ----------
    Mc : float
        Reference chirp mass.
    Mc_arr : array-like
        Array of chirp mass values.
    overlaps : array-like
        Overlap values corresponding to Mc_arr.
    sigma_Mc : float
        Standard deviation of chirp mass.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    shifts = Mc_arr - Mc
    ax.plot(
        Mc_arr,
        overlaps**2,
        color="k",
        lw=2,
        alpha=1,
        label=r"$|{\rm overlap}|^2$",
    )
    Mc = float(Mc)

    if sigma_Mc is not None:
        ax.plot(Mc_arr, np.exp(-1 / 2 * (shifts / sigma_Mc) ** 2), label="Gaussian")
        n_up = int(np.ceil((Mc_arr[-1] - Mc) / sigma_Mc))
        n_down = int(np.floor((Mc_arr[0] - Mc) / sigma_Mc))
        sigma_grid = np.arange(n_down, n_up + 1) * sigma_Mc
        for i, step in enumerate(sigma_grid):
            label = r"$1 \sigma_{\mathcal{M}}$ jumps" if i == 0 else None
            ax.axvline(step + Mc, c="m", label=label, alpha=0.5)
        ax.axvline(0.5 * sigma_Mc + Mc, c="b", label=r"$\pm 0.5 \sigma_{\mathcal{M}}$")
        ax.axvline(-0.5 * sigma_Mc + Mc, c="b")

    ax.axhline(np.exp(-1 / 2), c="k", label=r"1 $\sigma$ drop", lw=0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r"$\mathcal{M}$")
    ax.set_ylabel("Overlap$^2$")
    fig.tight_layout()
    return fig


def event_data_to_analysis_input(event: Union[data.EventData, str, Path]):
    """
    Converts event data into analysis input parameters for GWFast.
    Parameters:
        event (Union[data.EventData, str, Path]): The event data object or path to the event data file.
    Returns:
        Tuple[Dict[str, np.ndarray], List[str]]:
            - Dictionary of GWFast-compatible parameters.
            - List of detector names.
    """

    if isinstance(event, data.EventData):
        event_data = event
    else:
        try:
            event_data = data.EventData.from_npz(event)
        except FileNotFoundError:
            event_data = data.EventData.from_npz(filename=event)

    cw_par_dic = event_data.injection["par_dic"]

    gwfast_par_dic = cogwheel_to_gwfast_params(cw_par_dic)
    gwfast_par_dic["theta"] = np.pi / 2 - cw_par_dic["dec"]
    gwfast_par_dic["phi"] = cw_par_dic["ra"]
    gwfast_par_dic["tGPS"] = cw_par_dic["t_geocenter"]
    gwfast_par_dic["Phicoal"] = cw_par_dic["phi_ref"]  # not remotely true
    gwfast_par_dic["dL"] = cw_par_dic["d_luminosity"] / 1e3  # Mpc to Gpc
    gwfast_par_dic["psi"] = cw_par_dic["psi"]
    for k, v in gwfast_par_dic.items():
        if not isinstance(v, np.ndarray):
            gwfast_par_dic[k] = np.array([v])
    detector_names = event_data.detector_names
    return gwfast_par_dic, detector_names


def find_overlap_sigma_points(Mc_arr, Mc, overlaps):
    """
    Find the two points (left and right of Mc) where overlaps^2 crosses exp(-1/2).

    Parameters
    ----------
    Mc_arr : np.ndarray
        Array of chirp mass values.
    Mc : float
        Reference chirp mass.
    overlaps : np.ndarray
        Overlap values corresponding to Mc_arr.

    Returns
    -------
    left_sigma_point : float
        The Mc value to the left of Mc where overlaps^2 rises to exp(-1/2).
    right_sigma_point : float
        The Mc value to the right of Mc where overlaps^2 drops to exp(-1/2).
    """
    # Find all crossings where overlaps^2 crosses exp(-1/2)
    y = overlaps**2
    threshold = np.exp(-0.5)
    crossings = np.where(np.diff((y > threshold).astype(int)) != 0)[0]

    # Find crossings to the left and right of Mc
    idx_central = np.argmin(np.abs(Mc_arr - Mc))
    left_crossings = crossings[Mc_arr[crossings] < Mc]
    right_crossings = crossings[Mc_arr[crossings] > Mc]

    # Furthest left: smallest index, furthest right: largest index
    if len(left_crossings) > 0:
        left_idx = left_crossings[0]
        # Linear interpolation for more accurate crossing
        x0, x1 = Mc_arr[left_idx], Mc_arr[left_idx + 1]
        y0, y1 = y[left_idx], y[left_idx + 1]
        left_sigma_point = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    else:
        left_sigma_point = Mc_arr[0]

    if len(right_crossings) > 0:
        right_idx = right_crossings[-1]
        x0, x1 = Mc_arr[right_idx], Mc_arr[right_idx + 1]
        y0, y1 = y[right_idx], y[right_idx + 1]
        right_sigma_point = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    else:
        right_sigma_point = Mc_arr[-1]

    return left_sigma_point, right_sigma_point
