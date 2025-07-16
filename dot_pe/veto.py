"""
Veto module for sampler free inference.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from lal import GreenwichMeanSiderealTime
from numpy.typing import NDArray
from scipy.stats import chi2

from cogwheel import skyloc_angles
from cogwheel.data import EventData
from cogwheel.gw_utils import DETECTORS, get_fplus_fcross_0, get_geocenter_delays
from cogwheel.likelihood import CBCLikelihood, RelativeBinningLikelihood
from cogwheel.posterior import Posterior
from cogwheel.utils import get_rundir, mkdirs, NumpyEncoder
from cogwheel.waveform import WaveformGenerator

from .evidence_calculator import LinearFree
from .sampler_free_utils import get_event_data
from .single_detector import BlockLikelihood


class Vetoer:
    """
    A class to perform a single-detector veto.
    """

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def get_complex_overlaps(
        self, par_dic: dict, n_bands: int = 6
    ) -> NDArray[np.complex128]:
        """
        get complex overlaps for the chi2 test statistic.

        Parameters
        ----------
        like: CBCLikelihood
            likelihood object.
        par_dic: dict
            dictionary of parameters.
        n_bands: int
            number of equal SNR^2 frequency bands to split the data into.

        Returns
        -------
        complex_overlaps: np.ndarray
            complex overlaps. Under the null hypothesis, they should be
            distributed as CN(mean=scale, var=1).
        """
        h_f = self.likelihood._get_h_f(par_dic, by_m=False)[0]
        d_f = self.likelihood.event_data.strain[0]
        factor = 4 * self.likelihood.event_data.df * self.likelihood.asd_drift[0] ** -2
        wht_filter = self.likelihood.event_data.wht_filter[0]
        dh_f = factor * (d_f * h_f.conj() * wht_filter**2)  # complex
        hh_f = factor * (h_f * h_f.conj() * wht_filter**2).real  # real
        hh_cumsum = np.cumsum(hh_f)
        hh_cumsum = hh_cumsum / hh_cumsum[-1]
        split_inds = np.searchsorted(hh_cumsum, np.linspace(0, 1, n_bands + 1))

        hh_split = np.array(
            [
                hh_f[split_inds[i] : split_inds[i + 1]].sum()
                for i in range(0, len(split_inds) - 1)
            ]
        )
        dh_split = np.array(
            [
                dh_f[split_inds[i] : split_inds[i + 1]].sum()
                for i in range(0, len(split_inds) - 1)
            ]
        )

        complex_overlaps = (dh_split - hh_split) / np.sqrt(hh_split)

        return complex_overlaps

    @staticmethod
    def chi2_test_score(
        complex_overlaps: NDArray[np.complex128],
        veto_threshold: float = 1e-3,
    ) -> Tuple[bool, float, float]:
        """
        Perform chi2 test on complex overlaps.

        Parameters
        ----------
        complex_overlaps: np.ndarray
            complex overlaps.
        veto_threshold: float
            False-postitive error rate. Default is 1e-3.

        Returns
        -------
        vetoed_out: bool
            True if the null hypothesis is rejected, i.e. data is
            inconsistent with waveform that gave complex_overlaps.
        ts: float
            test statistic.
        sf: float
            survival function at the test statistic.
        """

        dof = len(complex_overlaps) * 2
        ts = np.sum(np.abs(complex_overlaps) ** 2)  # test statistic
        sf = chi2(dof).sf(ts)  # survival funciton at test-statistic
        threshold = chi2(dof).isf(
            veto_threshold
        )  # test-statistic at error-rate veto_threshold
        vetoed_out = ts > threshold
        return vetoed_out, ts, sf

    @staticmethod
    def plot_complex_overlaps(
        complex_overlaps: NDArray[np.complex128],
        labels: Union[None, List[str]] = None,
    ) -> plt.Figure:
        """
        Plot complex overlaps.

        Parameters
        ----------
        complex_overlaps: np.ndarray
            complex overlaps.

        labels: list
            list of labels for the complex overlaps.

        Returns
        -------
        fig: matplotlib.figure.Figure
        """

        xlim = [-3, 3]
        ylim = [-3, 3]

        fig = plt.figure()
        ax = plt.gca()
        colors = ["r", "b", "m"]
        n = complex_overlaps.shape[-1]
        radius = chi2(2).isf(1 / n) ** (1 / 2)
        label = rf"$\sqrt{{{{\rm ISF}}_{{\chi^2(2)}}(1/{n})}}$"

        ax.plot(
            radius * np.cos(np.linspace(0, np.pi * 2, 100)),
            radius * np.sin(np.linspace(0, np.pi * 2, 100)),
            c="k",
            alpha=0.75,
            label=label,
        )

        for co, c, lab in zip(complex_overlaps, colors, labels):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.scatter(co.real, co.imag, c=c, marker="x", label=lab)
            plt.axhline(0, color="k", linewidth=1)
            plt.axvline(0, color="k", linewidth=1)
        if labels is not None:
            if any(x is not None for x in labels):
                plt.legend()
        return fig


def find_max_lnlike_and_argmax(
    block_likelihood: BlockLikelihood,
    timeshifts_dbt: NDArray[np.complex128],
    blocksize: Union[None, int] = 512,
) -> Tuple[np.ndarray, float, int, int, int]:
    if blocksize is None:
        blocksize = block_likelihood.h_impb.shape[0]

    num_blocks = -(block_likelihood.h_impb.shape[0] // -blocksize)
    max_lnlike = -np.inf
    max_indices = None
    max_r = None

    for block_num in range(num_blocks):
        start_idx = block_num * blocksize
        end_idx = min(start_idx + blocksize, block_likelihood.h_impb.shape[0])

        r_iotp, lnlike_iot = block_likelihood.get_response_over_distance_and_lnlike(
            block_likelihood.dh_weights_dmpb,
            block_likelihood.hh_weights_dmppb,
            block_likelihood.h_impb[start_idx:end_idx],
            timeshifts_dbt,
            block_likelihood.likelihood.asd_drift,
            block_likelihood.evidence.n_phi,
            block_likelihood.evidence.m_arr,
        )

        i_block, o, t = np.unravel_index(np.argmax(lnlike_iot), lnlike_iot.shape)
        lnlike = lnlike_iot[i_block, o, t]
        r = r_iotp[i_block, o, t]
        if lnlike > max_lnlike:
            max_lnlike = lnlike
            max_indices = (i_block + start_idx, o, t)
            max_r = r_iotp[i_block, o, t]

    i, o, t = max_indices
    lnlike = max_lnlike
    r = max_r

    return r, lnlike, i, o, t


def find_bestfit_parameters(
    rundir: Union[str, Path],
    event: Union[str, Path, EventData],
    block_likelihood: BlockLikelihood,
    par_dic_0: dict,
    n_t: int,
    n_phi: int,
    dt_fraction: float = 1.0,
    blocksize: int = 512,
) -> Tuple[dict, dict]:
    """
    Perform sampling run using BlockLikelihood object and configuration
    parameters. Create rundir in event_dir to store the run results.
    """

    event_data = get_event_data(event)

    tgps = event_data.tgps
    gmst = GreenwichMeanSiderealTime(tgps)
    det_name = event_data.detector_names[0]
    tempdir = rundir / "temp"
    if not tempdir.exists():
        mkdirs(tempdir)
    print("Starting run....")

    # fix sky position to above detector
    lat, lon = skyloc_angles.cart3d_to_latlon(
        skyloc_angles.normalize(DETECTORS[det_name[0]].location)
    )
    r0 = get_fplus_fcross_0(det_name, lat, lon).squeeze()

    par_dic = block_likelihood.transform_par_dic_by_sky_poisition(
        det_name, par_dic_0, lon, lat, tgps
    )

    # fix time to be at the geocenter
    delay = get_geocenter_delays(det_name, par_dic["lat"], par_dic["lon"])[0]
    tcoarse = block_likelihood.likelihood.event_data.tcoarse
    t_grid = (np.arange(n_t) - n_t // 2) * (
        block_likelihood.likelihood.event_data.times[1] * dt_fraction
    )
    t_grid += par_dic["t_geocenter"] + tcoarse + delay

    timeshifts_dbt = np.exp(
        -2j
        * np.pi
        * t_grid[None, None, :]
        * block_likelihood.likelihood.fbin[None, :, None]
    )
    # find bestfit response-over-distance, and corresponding likelihood.

    r, lnlike, i, o, t = find_max_lnlike_and_argmax(
        block_likelihood, timeshifts_dbt, blocksize
    )

    d_luminosity = 1 / np.linalg.norm(r)
    arg_r = np.arctan2(r[1], r[0])
    arg_r0 = np.arctan2(r0[1], r0[0])
    psi = -(arg_r - arg_r0) / 2

    bestfit_par_dic = block_likelihood.intrinsic_sample_bank.iloc[i].to_dict()
    bestfit_par_dic["lat"] = lat
    bestfit_par_dic["lon"] = lon
    bestfit_par_dic["ra"] = skyloc_angles.lon_to_ra(lon, gmst)
    bestfit_par_dic["dec"] = bestfit_par_dic["lat"]
    bestfit_par_dic["d_luminosity"] = d_luminosity
    bestfit_par_dic["psi"] = psi

    dt, dphi = block_likelihood.intrinsic_sample_processor.load_linfree_dt_and_dphi(
        block_likelihood.waveform_dir, [i]
    )
    dt, dphi = dt[0], dphi[0]
    bestfit_par_dic["t_geocenter"] = t_grid[t] - tcoarse - delay + dt
    bestfit_par_dic["phi_ref"] = np.linspace(0, 2 * np.pi, n_phi)[o] - dphi

    metadata = dict(r=r, i=i, o=o, t=t, lnlike=lnlike)
    return bestfit_par_dic, metadata


def create_posterior(event_data: EventData, bank_folder: Union[str, Path]) -> Posterior:
    bank_folder = Path(bank_folder)

    with open(bank_folder / "bank_config.json", "r") as f:
        bank_config = json.load(f)
        approximant = bank_config["approximant"]
        mchirp_range = (bank_config["min_mchirp"], bank_config["max_mchirp"])
        fbin = np.array([bank_config["fbin"]])
        mchirp_guess = np.mean(mchirp_range)

    posterior_kwargs = {
        "likelihood_class": RelativeBinningLikelihood,
        "approximant": approximant,
        "prior_class": "CartesianIASPrior",
    }
    likelihood_kwargs = {"fbin": fbin, "pn_phase_tol": None}
    ref_wf_finder_kwargs = {
        "time_range": (-1e-1, +1e-1),
        "mchirp_range": mchirp_range,
    }

    post = Posterior.from_event(
        event=event_data,
        mchirp_guess=mchirp_guess,
        likelihood_kwargs=likelihood_kwargs,
        ref_wf_finder_kwargs=ref_wf_finder_kwargs,
        **posterior_kwargs,
    )

    return post


def get_par_dic_0(
    event_data: EventData,
    bank_folder: Union[str, Path],
    save_path: Union[str, Path] = None,
) -> dict:
    post = create_posterior(event_data, bank_folder)
    par_dic_0 = post.likelihood.par_dic_0
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(par_dic_0, f)
    return par_dic_0


def get_block_likelihood(
    event: Union[str, Path, EventData],
    bank_folder: Union[str, Path],
    par_dic_0: Union[str, Path, dict, None] = None,
    blocksize: int = 512,
    n_phi: int = 32,
    size_limit: int = 10,
    int_samples=Union[int, NDArray],
) -> BlockLikelihood:
    """
    Create a "slim" block likelihood evaluator.
    """
    bank_folder = Path(bank_folder)
    waveform_dir = bank_folder / "waveforms"
    with open(bank_folder / "bank_config.json", "r") as f:
        bank_config = json.load(f)
        approximant = bank_config["approximant"]
        fbin = np.array([bank_config["fbin"]])
        m_arr = np.array(bank_config["m_arr"])
    event_data = get_event_data(event)
    wfg = WaveformGenerator.from_event_data(event_data, approximant=approximant)

    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)
    block_likelihood = BlockLikelihood(
        intrinsic_bank_file=bank_folder / "intrinsic_sample_bank.feather",
        waveform_dir=bank_folder / "waveforms",
        n_phi=n_phi,
        m_arr=m_arr,
        likelihood=likelihood_linfree,
        size_limit=size_limit,
        ext_block_size=blocksize,
        int_block_size=blocksize,
    )
    if isinstance(int_samples, int):
        int_samples = np.array(range(int_samples))

    amp, phase = block_likelihood.intrinsic_sample_processor.load_amp_and_phase(
        waveform_dir, int_samples
    )

    # block_likelihood.logger = None
    block_likelihood.h_impb = amp * np.exp(1j * phase)

    return block_likelihood


def plot_wfs_and_data(
    event: Union[str, Path, EventData],
    par_dic_list: Union[dict, List[dict]],
    approximant_list: Union[str, List[str]] = "IMRPhenomXODE",
    labels: Union[str, List[str], None] = None,
    wf_plot_kwargs_list: List[dict] = None,
    trng: tuple = (-0.7, 0.1),
    savepath: Union[None, str, Path] = None,
):
    if not isinstance(par_dic_list, list):
        par_dic_list = [par_dic_list]
    if not isinstance(labels, list):
        labels = [None] * len(par_dic_list)
    if not isinstance(approximant_list, list):
        approximant_list = [approximant_list] * len(par_dic_list)

    if wf_plot_kwargs_list is None:
        wf_plot_kwargs_list = [{}] * len(par_dic_list)
    elif isinstance(wf_plot_kwargs_list, dict):
        wf_plot_kwargs_list = [wf_plot_kwargs_list] * len(par_dic_list)

    event_data = get_event_data(event)
    wfgs = [
        WaveformGenerator.from_event_data(event_data, approximant)
        for approximant in approximant_list
    ]

    # there is no asd_drift correction going on, so a few percent
    # errors in likelihood could happen
    likeihood_objects = [CBCLikelihood(event_data, wfg) for wfg in wfgs]

    fig = plot_wht_wfs(
        likelihood_objects=likeihood_objects,
        par_dics=par_dic_list,
        trng=trng,
        fig=None,
        figsize=None,
        data_plot_kwargs={"alpha": 0.5, "c": "k"},
        wf_plot_kwargs_list=wf_plot_kwargs_list,
        labels=labels,
    )
    if savepath is not None:
        fig.savefig(savepath, format="pdf")
    return fig


def plot_wht_wfs(
    likelihood_objects: Union[CBCLikelihood, List[CBCLikelihood]],
    par_dics: List[dict],
    labels: Union[None, List[str]] = None,
    trng: Tuple[float] = (-0.7, 0.1),
    fig: plt.Figure = None,
    figsize=None,
    data_plot_kwargs: Union[None, dict] = None,
    wf_plot_kwargs_list: Union[None, dict, List[dict]] = None,
) -> plt.Figure:
    create_legend = labels is not None
    if labels is None:
        labels = [None] * len(par_dics)
    if not isinstance(likelihood_objects, list):
        likelihood_objects = [likelihood_objects] * len(par_dics)
    if fig is None:
        fig = likelihood_objects[0]._setup_data_figure(figsize)
    axes = fig.get_axes()
    data_plot_kwargs = data_plot_kwargs if data_plot_kwargs else {}
    data_plot_kwargs = {
        "c": "C0",
        "lw": 0.2,
        "label": "Data",
    } | data_plot_kwargs

    if wf_plot_kwargs_list is None:
        wf_plot_kwargs_list = [{}] * len(par_dics)
    if isinstance(wf_plot_kwargs_list, dict):
        wf_plot_kwargs_list = [
            wf_plot_kwargs_list,
        ] * len(par_dics)

    time = (
        likelihood_objects[0].event_data.times
        - likelihood_objects[0].event_data.tcoarse
    )
    data_t_wht = likelihood_objects[0]._get_whitened_td(
        likelihood_objects[0].event_data.strain
    )
    wfs_t_wht = [
        like._get_whitened_td(like._get_h_f(par_dic, by_m=False))
        for like, par_dic in zip(likelihood_objects, par_dics)
    ]  # assume 1-detector wfs

    # Plot
    data_plotted = [False] * len(axes)
    for wf_t_wht, label, wf_plot_kwargs in zip(wfs_t_wht, labels, wf_plot_kwargs_list):
        for i, (ax, data_det, wf_det) in enumerate(zip(axes, data_t_wht, wf_t_wht)):
            if data_plotted[i] is False:
                ax.plot(time, data_det, **data_plot_kwargs)
                data_plotted[i] = True
            ax.plot(time, wf_det, label=label, **wf_plot_kwargs)

    plt.xlim(trng)
    if create_legend:
        plt.legend()
    return fig


def veto(
    event: Union[str, Path, EventData],
    par_dic: dict,
    n_bands: int = 6,
    veto_threshold: float = 1e-3,
    approximant: str = "IMRPhenomXODE",
) -> dict:
    """
    Perform veto analysis: Get complex overlaps & scale,
    perform chi2 test, plot and save results.

    Parameters:
        rundir (Union[str, Path]): The directory where the
        results will be saved.
        event (Union[str, Path, EventData]): The event data or
        path to the event data file.
        par_dic (dict): Dictionary of parameters for the analysis.
        n_bands (int, optional): Number of frequency bands to use
        in the analysis. Default is 6.
        veto_threshold (float, optional): Threshold for the veto
        chi2 test. Default is 1e-3.
        approximant (str, optional): The waveform approximant to
        use. Default is "IMRPhenomXODE".

    Returns:
        tuple: A tuple containing:
            - vetoed_out (bool): True if the null hypothesis is rejected.
            - ts (float): Test statistic.
            - sf (float): Survival function at the test statistic.
            - complex_overlaps (np.ndarray): Complex overlaps.
    """
    event_data = get_event_data(event)

    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    like = CBCLikelihood(event_data, wfg)
    vetoer = Vetoer(like)
    complex_overlaps = vetoer.get_complex_overlaps(par_dic, n_bands)
    vetoed_out, ts, sf = vetoer.chi2_test_score(complex_overlaps, veto_threshold)

    return vetoed_out, ts, sf, complex_overlaps


def main(
    bank_folder: Union[str, Path],
    event: Union[str, Path, EventData],
    rundir_home: Union[str, Path],
    par_dic_0: Union[dict, str, Path, None] = None,
    blocksize: int = 512,
    n_phi: int = 32,
    n_t: int = 128,
    dt_fraction: float = 1,
    size_limit: int = 10,
    int_samples: Union[int, NDArray] = 1024,
    n_bands: int = 6,
    veto_threshold: float = 1e-3,
    results_save_path: Union[str, Path] = None,
    wf_plot_kwargs: dict = {},
):
    """ """
    event_data = get_event_data(event)
    bank_folder = Path(bank_folder)
    with open(bank_folder / "bank_config.json", "r") as f:
        bank_approximant = json.load(f)["approximant"]
    if hasattr(event_data, "injection") and event_data.injection is not None:
        if event_data.injection["approximant"] != bank_approximant:
            event_data = EventData(**event_data.get_init_dict())
            event_data.injection = None

    rundir_home = Path(rundir_home)
    if not rundir_home.exists():
        mkdirs(rundir_home)
    rundir = get_rundir(rundir_home)

    if par_dic_0 is None:
        par_dic_0 = get_par_dic_0(event_data, bank_folder)
    elif isinstance(par_dic_0, (str, Path)):
        with open(par_dic_0, "r") as f:
            par_dic_0 = json.load(f)

    block_likelihood = get_block_likelihood(
        event,
        bank_folder,
        par_dic_0,
        blocksize,
        n_phi,
        size_limit,
        int_samples,
    )
    approximant = block_likelihood.likelihood.waveform_generator.approximant
    bestfit_par_dic, metadata = find_bestfit_parameters(
        rundir,
        event,
        block_likelihood,
        par_dic_0,
        n_t,
        n_phi,
        dt_fraction,
        blocksize,
    )
    with open(rundir / "bestfit_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, cls=NumpyEncoder, indent=4)
    with open(rundir / "bestfit_parameters.json", "w", encoding="utf-8") as fp:
        json.dump(bestfit_par_dic, fp, indent=4)

    _ = plot_wfs_and_data(
        event,
        bestfit_par_dic,
        approximant,
        labels=["Bestfit"],
        wf_plot_kwargs_list=wf_plot_kwargs,
    )

    (vetoed_out, ts, sf, complex_overlaps) = veto(
        event,
        bestfit_par_dic,
        n_bands=n_bands,
        veto_threshold=veto_threshold,
        approximant=approximant,
    )
    if results_save_path is None:
        results_save_path = rundir / "veto_results.json"

    with open(rundir / "results.json", "w", encoding="utf-8") as f:
        results = dict(
            vetoed_out=vetoed_out,
            ts=ts,
            sf=sf,
            complex_overlaps=complex_overlaps,
        )
        json.dump(results, f, cls=NumpyEncoder, indent=4)

    return rundir


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run single-detector veto analysis.")
    parser.add_argument("bank_folder", type=str, help="Path to the bank folder.")
    parser.add_argument("event", type=str, help="Path to the event data file.")
    parser.add_argument("rundir_home", type=str, help="Path to the run directory.")
    parser.add_argument(
        "--par_dic_0",
        type=str,
        default=None,
        help="Path to the initial parameter dictionary.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="Block size for the likelihood evaluation.",
    )
    parser.add_argument("--n_phi", type=int, default=32, help="Number of phi samples.")
    parser.add_argument("--n_t", type=int, default=128, help="Number of time samples.")
    parser.add_argument(
        "--dt_fraction",
        type=float,
        default=1.0,
        help="Fraction of the time step.",
    )
    parser.add_argument(
        "--size_limit",
        type=int,
        default=10,
        help="Size limit for the block likelihood.",
    )
    parser.add_argument(
        "--int_samples",
        type=int,
        default=1024,
        help="Number of intrinsic samples.",
    )
    parser.add_argument(
        "--n_bands",
        type=int,
        default=6,
        help="Number of frequency bands for the chi2 test.",
    )
    parser.add_argument(
        "--veto_threshold",
        type=float,
        default=1e-3,
        help="Threshold for the veto chi2 test.",
    )
    parser.add_argument(
        "--results_save_path",
        type=str,
        default=None,
        help="Path to save the results.",
    )
    parser.add_argument(
        "--wf_plot_kwargs",
        type=dict,
        default={},
        help="Waveform plot keyword arguments.",
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
