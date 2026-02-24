"""
Evidence calibration: compute Bayesian evidence on Gaussian noise.

Given a completed inference rundir, replace the data with Gaussian noise and
recompute the evidence. No optimization—par_dic_0 and extrinsic samples are
loaded from the rundir.

Uses a slim evidence path that does not store prob_samples or apply size_limit;
it computes ln_posterior wherever the likelihood is non-negligible and
accumulates logsumexp incrementally.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from cogwheel.data import EventData
from cogwheel.utils import read_json

from . import likelihood_calculating, sample_processing
from .likelihood_calculating import LinearFree
from .utils import inds_to_blocks, parse_bank_folders, safe_logsumexp, validate_bank_configs

# Threshold for dropping (i,e,o) combinations with approx lnlike below max - cut.
# Large value = keep almost all non-negligible contributions.
_APPROX_LNL_DROP_THRESHOLD = 30.0

# O4 ASD function names per detector (cogwheel convention)
_ASD_O4_BY_DET = {"H": "asd_H_O4", "L": "asd_L_O4", "V": "asd_V_O4"}


def _create_noise_from_event_psd(
    original: EventData,
    seed: Optional[int] = None,
    fmin: float = 15.0,
    df_taper: float = 1.0,
) -> EventData:
    """
    Create EventData with Gaussian noise using the original event's PSD.

    Recovers ASD from wht_filter (asd = highpass / wht_filter) and generates
    strain with the same spectral shape. No cogwheel asd_funcs needed.
    """
    from cogwheel.data import highpass_filter

    highpass = highpass_filter(
        original.frequencies, fmin=fmin, df_taper=df_taper
    )
    # wht_filter = highpass / asd  =>  asd = highpass / wht_filter
    asd = np.divide(
        highpass,
        original.wht_filter,
        out=np.full_like(original.wht_filter, np.nan, dtype=float),
        where=original.wht_filter != 0,
    )
    # Bins with wht_filter=0 are excluded by fslice; use finite asd for generation
    np.nan_to_num(asd, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

    duration = 1.0 / original.df
    rng = np.random.default_rng(seed)
    real, imag = rng.normal(
        scale=np.sqrt(duration) / 2 * asd, size=(2,) + asd.shape
    )
    strain = real + 1j * imag
    strain[:, [0, -1]] = strain[:, [0, -1]].real

    return EventData(
        eventname="",
        frequencies=original.frequencies,
        strain=strain,
        wht_filter=original.wht_filter,
        detector_names=original.detector_names,
        tgps=original.tgps,
        tcoarse=original.tcoarse,
        injection=None,
    )


def create_noise_event_data(
    rundir: Union[str, Path],
    *,
    event_path: Optional[Union[str, Path]] = None,
    use_event_psd: bool = True,
    asd_funcs: Optional[List[str]] = None,
    observing_run: str = "O4",
    seed: Optional[int] = None,
) -> EventData:
    """
    Create EventData with Gaussian noise for evidence calibration.

    Loads the original event from rundir. By default uses the event's own PSD
    (recovered from wht_filter) to generate noise—no cogwheel asd_funcs needed.
    Set use_event_psd=False to use cogwheel ASD models (asd_funcs) instead.

    Parameters
    ----------
    rundir : Union[str, Path]
        Path to completed inference rundir.
    event_path : Union[str, Path], optional
        Path to original event npz. If None, resolved from run_kwargs["event"].
    use_event_psd : bool
        If True (default), use the event's PSD from wht_filter. If False,
        use cogwheel asd_funcs (requires asd_funcs or observing_run).
    asd_funcs : list[str], optional
        ASD function names per detector. Only used if use_event_psd=False.
    observing_run : str
        Observing run for default ASD names. Only used if use_event_psd=False.
    seed : int, optional
        Random seed for noise realization.

    Returns
    -------
    EventData
        EventData with Gaussian noise.
    """
    rundir = Path(rundir)
    if event_path is not None:
        event_path = Path(event_path)
        if not event_path.exists():
            raise FileNotFoundError(f"Event file not found: {event_path}")
    else:
        run_kwargs_path = rundir / "run_kwargs.json"
        if not run_kwargs_path.exists():
            raise FileNotFoundError(f"run_kwargs.json not found in {rundir}")
        with open(run_kwargs_path, encoding="utf-8") as f:
            run_kwargs = json.load(f)
        event_path = Path(run_kwargs.get("event", ""))
        event_dir = run_kwargs.get("event_dir")
        if not event_path or not str(event_path).strip():
            raise ValueError("run_kwargs.json has no 'event' key")
        if not event_path.exists():
            for base in [rundir, Path(event_dir) if event_dir else None, Path.cwd()]:
                if base is None:
                    continue
                candidate = base / (event_path.name or event_path)
                if candidate.exists():
                    event_path = candidate
                    break
        if not event_path.exists():
            raise FileNotFoundError(
                f"Event file not found: {event_path}. Pass event_path explicitly."
            )

    original = EventData.from_npz(filename=str(event_path))
    if use_event_psd:
        return _create_noise_from_event_psd(original, seed=seed)
    if asd_funcs is None:
        asd_funcs = [
            _ASD_O4_BY_DET.get(det, f"asd_{det}_{observing_run}")
            for det in original.detector_names
        ]
    kwargs = {
        "eventname": "",
        "detector_names": original.detector_names,
        "duration": 1.0 / original.df,
        "asd_funcs": asd_funcs,
        "tgps": original.tgps,
        "tcoarse": original.tcoarse,
        "fmax": original.frequencies[-1],
    }
    if seed is not None:
        kwargs["seed"] = seed
    return EventData.gaussian_noise(**kwargs)


def _compute_evidence_per_bank_slim(
    *,
    event_data: EventData,
    bank_path: Path,
    top_rundir: Path,
    par_dic_0: Dict,
    inds: NDArray[np.int_],
    n_ext: int,
    n_phi: int,
    m_arr: NDArray[np.int_],
    blocksize: int,
) -> Tuple[float, float]:
    """
    Compute ln_evidence for a single bank using a slim block-wise accumulation.

    No prob_samples, no size_limit. Accumulates logsumexp(ln_posterior) over
    blocks, keeping only (i,e,o) where approx distance-marginalized lnlike
    is within _APPROX_LNL_DROP_THRESHOLD of the block maximum.

    Returns
    -------
    ln_evidence : float
    ln_evidence_discarded : float
        Always -np.inf (we do not track discarded in slim path).
    """
    from cogwheel.waveform import WaveformGenerator

    bank_path = Path(bank_path)
    waveform_dir = bank_path / "waveforms"
    bank_file_path = bank_path / "intrinsic_sample_bank.feather"
    with open(bank_path / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]

    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    intrinsic_sample_processor = sample_processing.IntrinsicSampleProcessor(
        likelihood_linfree, waveform_dir
    )
    likelihood_calc = likelihood_calculating.LikelihoodCalculator(
        n_phi=n_phi, m_arr=np.array(m_arr)
    )
    dh_weights_dmpb, hh_weights_dmppb = intrinsic_sample_processor.get_summary()

    bank = intrinsic_sample_processor.load_bank(
        bank_file_path,
        indices=inds,
        renormalize_log_prior_weights=False,
    )
    full_log_prior_weights_i = bank["log_prior_weights"].values

    top_rundir = Path(top_rundir)
    extrinsic_samples = pd.read_feather(top_rundir / "extrinsic_samples.feather")
    full_log_prior_weights_e = extrinsic_samples["log_prior_weights"].values

    response_dpe = np.load(top_rundir / "response_dpe.npy")
    timeshift_dbe = np.load(top_rundir / "timeshift_dbe.npy")

    n_total_samples = len(inds) * n_ext * n_phi

    i_blocks = inds_to_blocks(inds, blocksize)
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

    running_logsumexp_ln_posterior = -np.inf

    total_blocks = len(i_blocks) * len(e_blocks)
    with tqdm(total=total_blocks, desc="Slim evidence blocks") as pbar:
        for i_block in i_blocks:
            amp_impb, phase_impb = intrinsic_sample_processor.load_amp_and_phase(
                waveform_dir, i_block
            )
            h_impb = amp_impb * np.exp(1j * phase_impb)

            sort_order = np.argsort(inds)
            pos_in_inds = sort_order[np.searchsorted(inds[sort_order], i_block)]
            log_prior_weights_i_block = full_log_prior_weights_i[pos_in_inds]

            for e_block in e_blocks:
                response_subset = response_dpe[..., e_block]
                timeshift_subset = timeshift_dbe[..., e_block]
                log_prior_weights_e_block = full_log_prior_weights_e[e_block]

                dh_ieo, hh_ieo = likelihood_calc.get_dh_hh_ieo(
                    dh_weights_dmpb,
                    h_impb,
                    response_subset,
                    timeshift_subset,
                    hh_weights_dmppb,
                    likelihood_linfree.asd_drift,
                )

                (
                    inds_i_k,
                    inds_e_k,
                    inds_o_k,
                ) = likelihood_calc.select_ieo_by_approx_lnlike_dist_marginalized(
                    dh_ieo,
                    hh_ieo,
                    log_prior_weights_i_block,
                    log_prior_weights_e_block,
                    cut_threshold=_APPROX_LNL_DROP_THRESHOLD,
                )

                if len(inds_i_k) == 0:
                    pbar.update(1)
                    continue

                dh_k = dh_ieo[inds_i_k, inds_e_k, inds_o_k]
                hh_k = hh_ieo[inds_i_k, inds_e_k, inds_o_k]
                dist_marg_lnlike_k = likelihood_calc.lookup_table.lnlike_marginalized(
                    dh_k, hh_k
                )

                log_prior_k = (
                    log_prior_weights_i_block[inds_i_k]
                    + log_prior_weights_e_block[inds_e_k]
                    - np.log(n_total_samples)
                )
                ln_posterior_k = dist_marg_lnlike_k + log_prior_k
                log_contrib = np.logaddexp.reduce(ln_posterior_k)
                running_logsumexp_ln_posterior = safe_logsumexp(
                    [running_logsumexp_ln_posterior, log_contrib]
                )
                pbar.update(1)

    ln_evidence = running_logsumexp_ln_posterior
    return ln_evidence, -np.inf


def _run_evidence_slim_per_bank(
    *,
    banks: Dict[str, Path],
    event_data: EventData,
    rundir: Path,
    par_dic_0: Dict,
    selected_inds_by_bank: Dict[str, NDArray[np.int_]],
    n_ext: int,
    n_phi: int,
    m_arr: NDArray[np.int_],
    blocksize: int,
) -> List[Dict]:
    """Run slim evidence computation for each bank. Returns same structure as
    run_coherent_inference_per_bank but with lnZ_k from slim path."""
    bank_results = []

    for bank_id, bank_path in banks.items():
        inds = selected_inds_by_bank[bank_id]
        n_int_k = len(
            pd.read_feather(bank_path / "intrinsic_sample_bank.feather")
        )
        n_total_samples = n_phi * n_ext * n_int_k

        if len(inds) == 0:
            bank_results.append(
                {
                    "bank_id": bank_id,
                    "lnZ_k": -np.inf,
                    "lnZ_discarded_k": -np.inf,
                    "N_k": n_total_samples,
                }
            )
            continue

        lnZ_k, lnZ_discarded_k = _compute_evidence_per_bank_slim(
            event_data=event_data,
            bank_path=bank_path,
            top_rundir=rundir,
            par_dic_0=par_dic_0,
            inds=inds,
            n_ext=n_ext,
            n_phi=n_phi,
            m_arr=m_arr,
            blocksize=blocksize,
        )
        bank_results.append(
            {
                "bank_id": bank_id,
                "lnZ_k": lnZ_k,
                "lnZ_discarded_k": lnZ_discarded_k,
                "N_k": n_total_samples,
            }
        )

    return bank_results


def compute_evidence_on_noise(
    rundir: Union[str, Path],
    event_data_noise: Optional[Union[str, Path, EventData]] = None,
    *,
    event_path: Optional[Union[str, Path]] = None,
    use_event_psd: bool = True,
    asd_funcs: Optional[List[str]] = None,
    seed: Optional[int] = None,
    selected_inds_by_bank: Optional[Dict[str, NDArray[np.int_]]] = None,
    use_selected_inds: bool = False,
) -> Tuple[float, float]:
    """
    Compute Bayesian evidence on Gaussian noise using setup from a completed rundir.

    Loads par_dic_0, extrinsic samples, and bank indices from rundir. Uses the
    provided event_data_noise as the data. No optimization is performed—par_dic_0
    is taken as given. Extrinsic samples are loaded directly from the rundir.

    Parameters
    ----------
    rundir : Union[str, Path]
        Path to a completed inference rundir.
    event_data_noise : EventData, str, Path, or None
        EventData with Gaussian noise, or path to noise npz. If None, creates
        noise from rundir using create_noise_event_data().
    event_path : str or Path, optional
        Path to original event npz when creating noise (if run_kwargs resolution
        fails). Only used if event_data_noise is None.
    use_event_psd : bool
        Use event's PSD when creating noise (default True). Only if
        event_data_noise is None.
    asd_funcs : list[str], optional
        ASD functions when creating noise with use_event_psd=False.
    seed : int, optional
        Random seed when creating noise (only used if event_data_noise is None).
    selected_inds_by_bank : dict, optional
        Explicit {bank_id: inds}. If provided, selection is skipped and these
        indices are used.
    use_selected_inds : bool
        If True, load selection from intrinsic_samples.npz (as in inference.run).
        If False (default), use entire bank. Ignored when selected_inds_by_bank
        is provided.

    Returns
    -------
    ln_evidence : float
        Log of the total evidence.
    ln_evidence_discarded : float
        Log of the discarded evidence.
    """
    rundir = Path(rundir)
    if event_data_noise is None:
        event_data_noise = create_noise_event_data(
            rundir,
            event_path=event_path,
            use_event_psd=use_event_psd,
            asd_funcs=asd_funcs,
            seed=seed,
        )
    elif isinstance(event_data_noise, (str, Path)):
        event_data_noise = EventData.from_npz(filename=str(event_data_noise))
    if not rundir.exists():
        raise FileNotFoundError(f"Rundir not found: {rundir}")

    # Load par_dic_0 from Posterior.json (no optimization—taken as given)
    posterior_path = rundir / "Posterior.json"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Posterior.json not found in {rundir}. "
            "Rundir must be from a completed inference run."
        )
    posterior = read_json(posterior_path)
    par_dic_0 = posterior.likelihood.par_dic_0.copy()

    # Load run configuration
    run_kwargs_path = rundir / "run_kwargs.json"
    if not run_kwargs_path.exists():
        raise FileNotFoundError(f"run_kwargs.json not found in {rundir}")
    with open(run_kwargs_path, "r", encoding="utf-8") as f:
        run_kwargs = json.load(f)

    bank_folder = run_kwargs["bank_folder"]
    banks = parse_bank_folders(bank_folder)
    bank_paths = list(banks.values())
    bank_config = validate_bank_configs(bank_paths)
    m_arr = np.array(bank_config["m_arr"])

    n_ext = int(run_kwargs["n_ext"])
    n_phi = int(run_kwargs["n_phi"])
    blocksize = int(run_kwargs["blocksize"])

    # Intrinsic indices: explicit override, load selection, or use entire bank
    if selected_inds_by_bank is not None:
        # Caller passed indices; validate bank_ids match
        for bank_id in banks:
            if bank_id not in selected_inds_by_bank:
                raise ValueError(
                    f"selected_inds_by_bank missing bank_id {bank_id}"
                )
    else:
        banks_dir = rundir / "banks"
        selected_inds_by_bank = {}
        for bank_id, bank_path in banks.items():
            bank_df = pd.read_feather(
                bank_path / "intrinsic_sample_bank.feather"
            )
            n_int = len(bank_df)
            if use_selected_inds:
                if not banks_dir.exists():
                    raise FileNotFoundError(
                        f"banks directory not found in {rundir} "
                        "(required when use_selected_inds=True)"
                    )
                inds_path = banks_dir / bank_id / "intrinsic_samples.npz"
                if not inds_path.exists():
                    raise FileNotFoundError(
                        f"intrinsic_samples.npz not found for bank {bank_id}"
                    )
                selected_inds_by_bank[bank_id] = np.load(inds_path)["inds"]
            else:
                selected_inds_by_bank[bank_id] = np.arange(n_int)

    # Verify extrinsic samples exist in rundir
    extrinsic_samples_path = rundir / "extrinsic_samples.feather"
    if not extrinsic_samples_path.exists():
        raise FileNotFoundError(
            f"extrinsic_samples.feather not found in {rundir}"
        )

    # Run slim evidence per bank (no prob_samples, no size_limit)
    per_bank_results = _run_evidence_slim_per_bank(
        banks=banks,
        event_data=event_data_noise,
        rundir=rundir,
        par_dic_0=par_dic_0,
        selected_inds_by_bank=selected_inds_by_bank,
        n_ext=n_ext,
        n_phi=n_phi,
        m_arr=m_arr,
        blocksize=blocksize,
    )

    # Aggregate evidence across banks
    lnZ_values = [r["lnZ_k"] for r in per_bank_results]
    ln_evidence = safe_logsumexp(lnZ_values)
    lnZ_discarded_values = [r["lnZ_discarded_k"] for r in per_bank_results]
    ln_evidence_discarded = safe_logsumexp(lnZ_discarded_values)

    return ln_evidence, ln_evidence_discarded
