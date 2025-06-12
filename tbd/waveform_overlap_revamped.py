"""
Waveform overlap calculations for sampler free inference.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import qmc
from lal import MSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

# COGWHEEL imports
from cogwheel import data, gw_utils, posterior, utils, waveform
from cogwheel.likelihood import RelativeBinningLikelihood

# TBD imports
from tbd import config
from tbd.evidence_calculator import (
    IntrinsicSampleProcessor,
    LinearFree,
)
from tbd.sampler_free_utils import (
    flex_reshape,
    get_device_per_dtype,
    torch_dtype,
    safe_move_and_cast,
    setup_logger,
)

import cProfile
import pstats
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import coo_matrix, lil_matrix

import torch
from torch.types import Device

PRECOMPUTE_BLOCKSIZE = 128
EVENT_DATA_KWARGS = {
    "detector_names": "L",
    "duration": 120.0,
    "asd_funcs": [
        "asd_L_O3",
    ],
    "tgps": 0.0,
    "fmax": 1600.0,
}


def eta_chieff_to_chieff_hat(eta, chieff):
    """A simplified version of chieff_hat.
    See full version at https://arxiv.org/pdf/1107.1267
    """
    return eta + chieff / 4 - 1 / 4


def bin_bank_by_mchirp_chieff_hat(
    df: pd.DataFrame,
    mchirp_bins: int,
    chieff_hat_bins: int,
    inplace: bool = True,
) -> Union[None, pd.DataFrame]:
    if not inplace:
        df = df.copy()
    if "eta" not in df.columns:
        eta = gw_utils.q_to_eta(df["m2"] / df["m1"])
    else:
        eta = df["eta"]
    if "chieff" not in df.columns:
        chieff = gw_utils.chieff(
            df["m1"],
            df["m2"],
            df["s1z"],
            df["s2z"],
        )
    else:
        chieff = df["chieff"]

    if "mchirp" not in df.columns:
        df["mchirp"] = gw_utils.m1m2_to_mchirp(df["m1"], df["m2"])
    if "mchirp_hat" not in df.columns:
        df["chieff_hat"] = -1 / 4 + eta + chieff / 4
    mchirp_grid = np.linspace(
        df.mchirp.min() - 1e-10,
        df.mchirp.max() + 1e-10,
        mchirp_bins + 1,
        endpoint=True,
    )
    chieff_hat_grid = np.linspace(
        df.chieff_hat.min() - 1e-10,
        df.chieff_hat.max() + 1e-10,
        chieff_hat_bins + 1,
        endpoint=True,
    )

    df["mchirp_bin"] = pd.cut(df["mchirp"], bins=mchirp_grid)
    df["chieff_hat_bin"] = pd.cut(df["chieff_hat"], bins=chieff_hat_grid)
    if not inplace:
        return df


def split_df_by_bins(df: pd.DataFrame, by: Union[str, List[str]]) -> List[pd.DataFrame]:
    x = [g for _, g in df.groupby(by=by, observed=False)]
    for i, g in enumerate(x):
        g["original_index"] = g.index
        g.reset_index(drop=True, inplace=True)
    return x


def select_from_df_by_closeness(df, sub_bank, params, distances):
    if not isinstance(params, list):
        params = [params]
    if not isinstance(distances, list):
        distances = [distances]
    cond = np.all(
        np.abs(df[params].values - sub_bank[params].median().values) < distances,
        axis=1,
    )
    return df.loc[cond]


def _get_par_dic_0(bank_folder: Union[str, Path]) -> dict:
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        f_ref = bank_config["f_ref"]
        approximant = bank_config["approximant"]
    if waveform.APPROXIMANTS[approximant].aligned_spins:
        inplane_spin = 0
    else:
        inplane_spin = 0.2
    bank = pd.read_feather(bank_folder / "intrinsic_sample_bank.feather")
    complementaty_params = dict(
        l1=0,
        l2=0,
        psi=0,
        phi_ref=0,
        ra=1,
        dec=1,
        f_ref=f_ref,
        d_luminosity=1,
        t_geocenter=0,
        iota=1.0,
        s1x_n=inplane_spin,
        s1y_n=inplane_spin,
        s1z=0.2,
        s2x_n=inplane_spin,
        s2y_n=inplane_spin,
        s2z=0.2,
    )
    init_params = [
        "m1",
        "m2",
        "s1x_n",
        "s1y_n",
        "s1z",
        "s2x_n",
        "s2y_n",
        "s2z",
    ]
    par_dic_0 = bank[init_params].median().to_dict() | complementaty_params
    return par_dic_0


def _get_event_data() -> data.EventData:
    return data.EventData.gaussian_noise("", **EVENT_DATA_KWARGS)


def get_strain_at_detector_mpdf(
    waveform_generator,
    event_data,
    par_dic,
    f=None,
):
    if f is None:
        f = event_data.frequencies
        n_f = len(f)
        fslice = event_data.fslice
        f_for_wfg = f[fslice]
    else:
        n_f = len(f)
        fslice = slice(0, n_f)
        f_for_wfg = f

    n_m = len(waveform_generator.m_arr)
    n_p = 2
    n_d = len(event_data.detector_names)
    x = np.zeros((n_m, n_p, n_d, n_f), dtype=complex)
    x[..., fslice] = waveform_generator.get_hplus_hcross_at_detectors(
        f_for_wfg, par_dic, by_m=True
    )
    return np.moveaxis(x, (0, 1, 2), (1, 2, 0))


def get_weights(bank_folder):
    """
    Get relative binning wights for <h1,h2> inner products, usable
    for a specific bank.
    """
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        fbin = bank_config["fbin"]
        approximant = bank_config["approximant"]

    par_dic_0 = _get_par_dic_0(bank_folder)

    event_data = _get_event_data()

    wfg = waveform.WaveformGenerator.from_event_data(
        event_data, approximant=approximant
    )

    lk = RelativeBinningLikelihood(
        event_data,
        wfg,
        par_dic_0=par_dic_0,
        fbin=fbin,
    )

    h0_dmpf = get_strain_at_detector_mpdf(wfg, event_data, par_dic_0)
    h0_dmpb = get_strain_at_detector_mpdf(wfg, event_data, par_dic_0, lk.fbin)

    shift_f = np.exp(-2j * np.pi * event_data.frequencies * event_data.tcoarse)
    shift_fb = np.exp(-2j * lk.fbin)
    h0_dmpf *= shift_f.conj()
    h0_dmpb *= shift_fb.conj()
    lk._stall_ringdown(h0_dmpf, h0_dmpb)
    whitened_h0_dmpf = h0_dmpf * event_data.wht_filter[:, None, None, :]
    h0_dmpf *= shift_f.conj()
    h0_dmpb *= shift_fb.conj()
    h0_h0_dmmppf = np.einsum(
        "dmpf, dMPf-> dmMpPf",
        whitened_h0_dmpf,
        whitened_h0_dmpf.conj(),
        optimize=True,
    )
    h_h_weights_dmmppb = lk._get_summary_weights(h0_h0_dmmppf)

    h_h_weights_dmmppb = np.einsum(
        "dmMpPb, dmpb, dMPb-> dmMpPb",
        h_h_weights_dmmppb,
        1 / h0_dmpb,
        1 / h0_dmpb.conj(),
    )
    return h_h_weights_dmmppb


def get_intrinsic_sample_processor(bank_folder: Union[str, Path]):
    """
    Get IntrinsicSampleProcessor using the bank folder.
    """
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        fbin = bank_config["fbin"]
        approximant = bank_config["approximant"]
    waveform_dir = bank_folder / "waveforms"
    event_data = _get_event_data()
    par_dic_0 = _get_par_dic_0(bank_folder)
    wfg = waveform.WaveformGenerator.from_event_data(
        event_data, approximant=approximant
    )

    linear_free_likelihood = LinearFree(event_data, wfg, par_dic_0, fbin)

    intrinsic_sample_processor = IntrinsicSampleProcessor(
        linear_free_likelihood, waveform_dir
    )

    return intrinsic_sample_processor


def eig_2by2(mat):
    """
    Compute the eigenvalues and eigenvectors of real, symmetric,
    positive-definite matrices, each 2x2. GPU implementation.

    Parameters
    ----------
    mat : (..., 2, 2) array
        The matrices to diagonalize.

    Returns
    -------
    eigvals : (..., 2) array
        eigenvalues, sorted in descending order.
    eigvecs : (..., 2, 2) array
        eigenvectors, corresponding to the eigenvalues,
        in the columns of eigvecs.

    """
    mat = torch.as_tensor(mat)
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    d = mat[..., 1, 1]
    # assume mat[...,1,0] == mat[...,0,1]
    trace = a + d
    det = a * d - b**2

    term = safe_sqrt(trace**2 - 4 * det)
    lambda1 = (trace + term) / 2
    lambda2 = (trace - term) / 2
    # verify results using
    # trace = a+b = lambda1 + lambda2,
    # det = lambda1 * lambda2,
    v1 = torch.stack([b, lambda1 - a], axis=-1)
    v2 = torch.stack([d - lambda2, -b], axis=-1)
    v1 = v1 / torch.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / torch.linalg.norm(v2, axis=-1, keepdims=True)
    eigvecs = torch.stack([v1, v2], axis=-1)
    eigvals = torch.stack([lambda1, lambda2], axis=-1)

    return eigvals, eigvecs


def compute_and_save_hh(
    bank_folder: Union[str, Path],
    target_folder: Union[str, Path],
    n_phi: int,
    blocksize: Union[int, None] = None,
    overwrite: bool = False,
    device: Union[str, Device] = "cpu",
    ip_dtype=torch.float32,
    wf_dtype=torch.float64,
):
    """
    Compute and save the inner product matrices of the waveforms with
    themselves, and with phase-shifted versions of themselves.
    bank_folder : str or Path
        Path to the folder containing the bank configuration file.
    n_phi : int
        Number of orbital phase shifts to consider, evenly spaced
        between 0 and 2pi.
    blocksize : int
        Number of waveforms to process at once. If None, the blocksize
        from the bank configuration file is used. Larger blocks require
        more available GPU memory. Smaller blocks are slower.
    """

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        blocksize = blocksize if blocksize else bank_config["blocksize"]
        bank_size = bank_config["bank_size"]

    hh_ipp_filepath = target_folder / "hh_ipp.npy"
    hh_iopp_filepath = target_folder / "hh_iopp.npy"
    norm_i_filepath = target_folder / "norm_i.npy"
    norm_io_filepath = target_folder / "norm_io.npy"
    if (
        not overwrite
        and hh_ipp_filepath.exists()
        and hh_iopp_filepath.exists()
        and norm_i_filepath.exists()
        and norm_io_filepath.exists()
    ):
        shape = np.load(hh_iopp_filepath).shape
        if (shape[0] == bank_size) and (shape[1] == n_phi):
            print("Inner products already computed. Skipping.")
            return

    intrinsic_sample_processor = get_intrinsic_sample_processor(bank_folder)
    waveform_dir = bank_folder / "waveforms"
    weights = torch.as_tensor(get_weights(bank_folder), device=device)
    phi_grid = torch.as_tensor(
        np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
        device=device,
        dtype=wf_dtype,
    )
    numpy_ip_dtype = getattr(np, str(ip_dtype).split(".")[-1])
    hh_ipp = np.zeros((bank_size, 2, 2), dtype=numpy_ip_dtype)
    hh_iopp = np.zeros((bank_size, n_phi, 2, 2), dtype=numpy_ip_dtype)
    n_blocks = -(bank_size // -blocksize)

    for bi in tqdm(
        iterable=range(n_blocks),
        desc="Precomputing inner products in blocks",
        total=n_blocks,
    ):
        inds_i = np.arange(bi * blocksize, np.min([(bi + 1) * blocksize, bank_size]))
        amp, phase = intrinsic_sample_processor.load_amp_and_phase(waveform_dir, inds_i)
        amp = torch.as_tensor(amp, device=device, dtype=wf_dtype)
        phase = torch.as_tensor(phase, device=device, dtype=wf_dtype)
        h_impb = amp * torch.exp(1j * phase)
        del amp, phase
        h_iompb = apply_phase_shift(bank_folder, h_impb, phi_grid)
        hh_ipp[inds_i] = get_inner_product_matrix(
            weights,
            h_impb,
            h_impb,
            broadcast="both",
            ip_dtype=ip_dtype,
            wf_device=device,
            ip_device="cpu",
        ).cpu()
        hh_iopp[inds_i] = get_inner_product_matrix(
            weights,
            h_iompb,
            h_iompb,
            broadcast="both",
            ip_dtype=ip_dtype,
            wf_device=device,
            ip_device="cpu",
        ).cpu()
    # normalize to reduce required precision
    norm_i = hh_ipp[..., 0, 0] + hh_ipp[..., 1, 1]  # i
    norm_io = hh_iopp[..., 0, 0] + hh_iopp[..., 1, 1]  # o
    hh_ipp = hh_ipp / norm_i[..., None, None]
    hh_iopp = hh_iopp / norm_io[..., None, None]
    # save results

    if not target_folder.exists():
        utils.mkdirs(target_folder)
    np.save(hh_ipp_filepath, hh_ipp)
    np.save(hh_iopp_filepath, hh_iopp)
    np.save(norm_i_filepath, norm_i)
    np.save(norm_io_filepath, norm_io)


def safe_sqrt(x):
    x = torch.as_tensor(x)
    return torch.sqrt(x.maximum(torch.zeros_like(x)))


def get_optimal_overlap_ijtopp(
    h1h1_iopp: torch.Tensor,
    h2h2_jpp: torch.Tensor,
    h1h2_itojpp: torch.Tensor,
    clip: bool = True,
    normalize: bool = False,
):
    """
    Same as get_optimal_overlap, but for multiple waveforms (i) in h1,
    and mutliple phases and times in h2. Assuming shapes:
    h1h1_ipp: (n_i, n_o, n_p, n_p)
    h2h2_jopp: (n_j, n_p, n_p)
    h1h2_itojpp: (n_i, n_o, n_t, n_j, n_p, n_p)

    return
    max_overlap : (n_i, n_t, n_o, n_j)
    """

    if normalize:
        norm1_io = h1h1_iopp[..., 0, 0] + h1h1_iopp[..., 1, 1]  # i
        norm2_j = h2h2_jpp[..., 0, 0] + h2h2_jpp[..., 1, 1]  # o

        h1h1_iopp = h1h1_iopp / norm1_io[..., None, None]
        h2h2_jpp = h2h2_jpp / norm2_j[..., None, None]
        h1h2_itojpp = h1h2_itojpp / np.sqrt(
            flex_reshape(norm1_io, "io", "itojpP")
            * flex_reshape(norm2_j, "j", "itojpP")
        )

    eigvals1_iop, eigvecs1_iopp = eig_2by2(h1h1_iopp)  # (i,i,2) vals, (i,2) vecs
    eigvals2_jp, eigvecs2_jpp = eig_2by2(h2h2_jpp)  # (j,2) vals, (j,2) vecs

    d1_inv_iopp = torch.zeros_like(h1h1_iopp)
    d2_inv_jpp = torch.zeros_like(h2h2_jpp)
    d1_inv_iopp[..., 0, 0] = 1 / torch.sqrt(eigvals1_iop[..., 0])
    d1_inv_iopp[..., 1, 1] = 1 / torch.sqrt(eigvals1_iop[..., 1])
    d2_inv_jpp[..., 0, 0] = 1 / torch.sqrt(eigvals2_jp[..., 0])
    d2_inv_jpp[..., 1, 1] = 1 / torch.sqrt(eigvals2_jp[..., 1])

    # from orthonormality, the inverse of the eigenvectors
    # matrix is its own transpose

    leftmost_iopp = d1_inv_iopp @ eigvecs1_iopp.transpose(-1, -2)
    rightmost_jpp = eigvecs2_jpp @ d2_inv_jpp
    C_itojpp = h1h2_itojpp @ rightmost_jpp
    C_jtiopp = torch.moveaxis(C_itojpp, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 4, 5))
    D_jtiopp = leftmost_iopp @ C_jtiopp
    D_ijtopp = torch.moveaxis(D_jtiopp, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))

    # the maximal overlap is the maximal singular value of D_itopp
    # This is the sqrt of the maximal eigenvalue of D_itopp.T @ D_itopp

    # trace of the "square" of a matrix is the sum of squared elements
    # det of a "square" of a matrix is the determinant of the matrix, squared
    trace = torch.sum(D_ijtopp**2, axis=(-1, -2))
    det = (
        D_ijtopp[..., 0, 0] * D_ijtopp[..., 1, 1]
        - D_ijtopp[..., 0, 1] * D_ijtopp[..., 1, 0]
    ) ** 2
    larger_eigenvalue = (trace + safe_sqrt(trace**2 - 4 * det)) / 2

    max_overlap = torch.sqrt(larger_eigenvalue)
    if clip:
        tol = 1e-2
        value_too_large = torch.max(max_overlap) > (1 + tol)
        if value_too_large:
            print("Error! Max overlap exceeds 1:", torch.max(max_overlap))
            torch.clip(max_overlap, None, 1, out=max_overlap)
    return max_overlap


def get_inner_product_matrix(
    weights,
    h1,
    h2,
    ip_dtype=torch.float32,
    wf_device: Union[str, Device] = "cpu",
    ip_device: Union[str, Device] = "cpu",
    broadcast: Union[str, bool] = False,
):
    """
    Get the inner product matrix of <h1,h2> per polarizations p
    and P. Allow for different forms of broadcasting.

    Parameters
    ----------
    weights : array-like
        Weight matrix for the inner product.
    h1 : array-like
        First set of waveforms.
    h2 : array-like
        Second set of waveforms.
    broadcast : str or bool, optional
        Broadcasting mode. Options are "none", "1", "2",
        "both" or "flexible". Default is False (interpreted as "none").

    Returns
    -------
    h1h2_pp : ndarray
        Inner product matrix with dimensions depending on the
        broadcasting.

    Raises
    ------
    ValueError
        If an invalid broadcasting mode is provided.
    """
    weights = torch.as_tensor(weights, device=wf_device)
    h1 = torch.as_tensor(h1, device=wf_device)
    h2 = torch.as_tensor(h2, device=wf_device)

    if not broadcast or broadcast == "none":
        h1h2_pp = torch.einsum(
            "dmMpPb, mpb, ...MPb->...pP",
            weights,
            h1,
            h2.conj(),
        ).real
    elif broadcast == "1":
        h1h2_pp = torch.einsum(
            "dmMpPb, ...mpb, MPb->...pP",
            weights,
            h1,
            h2.conj(),
        ).real
    elif broadcast == "2":
        h1h2_pp = torch.einsum(
            "dmMpPb, mpb, ...MPb->...pP",
            weights,
            h1,
            h2.conj(),
        ).real
    elif broadcast == "both":
        h1h2_pp = torch.einsum(
            "dmMpPb, ...mpb, ...MPb->...pP",
            weights,
            h1,
            h2.conj(),
        ).real
    elif broadcast == "flexible":
        h1h2_pp = get_inner_product_matrix_flexible(weights, h1, h2)
    else:
        raise ValueError(
            f"Invalid broadcasting mode: {broadcast}. "
            + "Choose from 'none', '1', '2', or 'both'."
        )

    return h1h2_pp.type(ip_dtype).to(ip_device)


def apply_time_and_phase_shift(bank_folder, h, t, phi):
    """
    Apply time and phase shift to the waveform.

    Parameters
    ----------
    bank_folder : str or Path
        Path to the folder containing the bank configuration file.
    h : (..., n_m, n_p, n_b) array
        The waveform(s) to be shifted.
    t : (n_t,) array
        Time shifts to apply.
    phi : (n_phi,) array
        Orbital phase shifts to apply.

    Returns
    -------
    h_shifted : (..., n_t, n_phi, n_m, n_p, n_b) array
        The shifted waveform(s).
    """
    h = torch.as_tensor(h)
    t = torch.as_tensor(t, device=h.device)
    phi = torch.as_tensor(phi, device=h.device)
    bank_folder = Path(bank_folder)

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        fbin = torch.tensor(bank_config["fbin"], device=h.device)  # Shape (n_b,)
        m_arr = torch.tensor(bank_config["m_arr"], device=h.device)

    # Reshape fbin and m_arr to match the last dimensions of h
    fbin = fbin.reshape((1,) * (h.ndim - 1) + (-1,))  # Shape (..., 1, n_b)
    m_arr = m_arr.reshape((1,) * (h.ndim - 3) + (-1, 1, 1))  # Shape (..., n_m, 1, 1)

    # Reshape t and phi for broadcasting
    t = t[:, None, None, None, None]  # Shape (n_t, 1, 1, 1, 1)
    phi = phi[None, :, None, None, None]  # Shape (1, n_phi, 1, 1, 1)
    # TODO: consider removing conversion

    # Calculate shifts with broadcasting
    timeshift = torch.exp(-2j * np.pi * fbin * t)  # Shape (n_t, ..., 1, n_b)
    phaseshift = torch.exp(1j * phi * m_arr)  # Shape (1, n_phi, ..., n_m, 1, 1)

    # Apply shifts to h and return
    h_shifted = h[..., None, None, :, :, :] * timeshift * phaseshift
    return h_shifted


def apply_phase_shift(bank_folder, h, phi):
    """
    Apply time and phase shift to the waveform.

    Parameters
    ----------
    bank_folder : str or Path
        Path to the folder containing the bank configuration file.
    h : (..., n_m, n_p, n_b) array
        The waveform(s) to be shifted.
    phi : (n_phi,) array
        Orbital phase shifts to apply.

    Returns
    -------
    h_shifted : (..., n_t, n_phi, n_m, n_p, n_b) array
        The shifted waveform(s).
    """
    bank_folder = Path(bank_folder)
    h = torch.as_tensor(h)

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        m_arr = torch.tensor(bank_config["m_arr"], device=h.device)  # Shape (n_m,)

    # Reshape fbin and m_arr to match the last dimensions of h
    m_arr = m_arr.reshape((1,) * (h.ndim - 3) + (-1, 1, 1))  # Shape (..., n_m, 1, 1)

    # Reshape t and phi for broadcasting
    phi = phi[:, None, None, None]  # Shape (n_phi, 1, 1, 1)
    phi = torch.as_tensor(phi, device=h.device)

    # Calculate shifts with broadcasting
    phaseshift = torch.exp(1j * phi * m_arr).to(
        h.device
    )  # Shape (1, n_phi, ..., n_m, 1, 1)

    # Apply shifts to h and return
    h_shifted = h[..., None, :, :, :] * phaseshift
    return h_shifted


def get_inner_product_matrix_flexible(
    w: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
):
    """
    Compute the inner product matrix <h1p, h2P> for flexible h1 and h2.

    This function calculates the inner product matrix with a weight
    matrix `w` and two tensors `h1` and `h2` that have flexible
    dimensions. The dimensions of `w` are fixed as "dmMpPb", while `h1`
    and `h2` have flexible additional dimensions plus "mpb" and "MPb"
    respectively.
    w, h1 and h2 must match on dtype and deivce.
    w : torch.Tensor
        Relative Weight matrix for the inner product calculation with
        shape (1, n_m, n_m, n_p, n_p, n_b).
    h1 : torch.Tensor
        Left operand tensor with flexible additional dimensions plus
        "mpb".
    h2 : torch.Tensor
        Right operand tensor with flexible additional dimensions plus
        "MPb".
    torch.Tensor
        Resulting inner product matrix.
    """

    # Dynamically determine the flexible dimensions
    h1_extra_dims = "".join(
        chr(ord("i") + i) for i in range(h1.ndim - len("mpb"))
    )  # 'i', 'j', ...
    h2_extra_dims = "".join(
        chr(ord("J") + i) for i in range(h2.ndim - len("MPb"))
    )  # 'J', 'K', ...
    w_dims = "dmMpPb"
    h1_dims = h1_extra_dims + "mpb"
    h2_dims = h2_extra_dims + "MPb"
    result_dims = h1_extra_dims + h2_extra_dims + "pP"
    einsum_expr = f"{w_dims}, {h1_dims}, {h2_dims} -> {result_dims}"

    res = torch.einsum(einsum_expr, w, h1, h2.conj()).real

    return res


def find_last_file(target_dir):
    target_dir = Path(target_dir)
    basename = "overlap_matrix"
    files = sorted(
        target_dir.glob(f"{basename}_sub_bank_*.npz"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )

    for file in reversed(files):
        try:
            file_data = np.load(file)
            overlaps_ij = file_data["overlaps_ij"]
            i_inds = file_data["i_inds"]
            j_inds = file_data["j_inds"]
            return file
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    return None


def get_last_calculated_sub_bank(target_dir):
    """Identify the current m block and last-calculated n block from the
    last .npz file."""
    latest_file = find_last_file(target_dir)
    if latest_file is None:
        return -1

    sub_bank_index = int(latest_file.stem.split("_")[-1])
    return sub_bank_index


def get_h1h2_itojpp(
    weights: torch.Tensor,
    h1_impb: torch.Tensor,
    h2_jmpb: torch.Tensor,
    phasor_mo: torch.Tensor,
    timeshift_exp_bt: torch.Tensor,
) -> torch.Tensor:
    h1h2_imptjP = torch.einsum(
        "dmMpPb, impb,jMPb,bt->imptjP",
        weights,
        h1_impb,
        h2_jmpb.conj(),
        timeshift_exp_bt,
    )
    h1h2_ijtpPm = h1h2_imptjP.permute(0, 4, 3, 2, 5, 1)
    return torch.einsum("ijtpPm,mo->ijtopP", h1h2_ijtpPm, phasor_mo).real


def _calculate_overlap_matrix_by_subbanks(
    bank_folder: Union[str, Path],
    target_folder: Union[str, Path],
    i_start: int = 0,
    i_end: Union[int, None] = None,
    h1_blocksize: int = 16,
    h2_blocksize: int = 512,
    n_t: int = 16,
    n_phi: int = 32,
    ip_dtype=torch.float32,
    wf_dtype=torch.float64,
    continue_flag: bool = True,
    mchirp_bins: int = 8,
    chieff_hat_bins: int = 8,
    wf_device="cpu",
    ip_device="cpu",
):
    """
    Calculate overlap matrix <h1|h2> by sub_banks.
    """

    def wf_tensor(x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=wf_device, dtype=wf_dtype)
        return safe_move_and_cast(x, wf_dtype, wf_device)

    def ip_tensor(x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=ip_device, dtype=ip_dtype)
        return safe_move_and_cast(x, ip_dtype, ip_device)

    # setup
    bank_folder = Path(bank_folder)
    target_folder = Path(target_folder)
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        bank_size = bank_config["bank_size"]
        m_arr = np.array(bank_config["m_arr"])
        fbin = np.array(bank_config["fbin"])
    intrinsic_sample_bank = pd.read_feather(
        bank_folder / "intrinsic_sample_bank.feather"
    )
    i_end = i_end if i_end else intrinsic_sample_bank.shape[0]
    intrinsic_sample_bank = intrinsic_sample_bank.iloc[i_start:i_end]

    bin_bank_by_mchirp_chieff_hat(
        intrinsic_sample_bank, mchirp_bins, chieff_hat_bins, inplace=True
    )
    sub_banks = split_df_by_bins(
        intrinsic_sample_bank, by=["mchirp_bin", "chieff_hat_bin"]
    )
    print(f"Dividing the bank into {len(sub_banks)} sub-banks.")
    mchirp_dist = (
        2
        * (
            intrinsic_sample_bank["mchirp"].max()
            - intrinsic_sample_bank["mchirp"].min()
        )
        / mchirp_bins
    )

    chieff_hat_dist = (
        2
        * (
            intrinsic_sample_bank["chieff_hat"].max()
            - intrinsic_sample_bank["chieff_hat"].min()
        )
        / chieff_hat_bins
    )
    distances = [mchirp_dist, chieff_hat_dist]

    filetype = "npz"
    basename = "overlap_matrix"

    i_end = i_end if i_end else bank_size
    weights = torch.tensor(
        get_weights(bank_folder), device=wf_device, dtype=wf_dtype.to_complex()
    )

    intrinsic_sample_processor = get_intrinsic_sample_processor(bank_folder)

    dt = (
        intrinsic_sample_processor.likelihood.event_data.times[1]
        - intrinsic_sample_processor.likelihood.event_data.times[0]
    )
    t_grid = (np.arange(n_t) - n_t // 2) * dt
    timeshift_exp_bt = torch.tensor(
        np.exp(-2j * np.pi * t_grid[:, None] * fbin[None, :]),
        device=wf_device,
        dtype=wf_dtype.to_complex(),
    )
    phi_grid = np.linspace(0, 2 * np.pi, n_phi)
    phasor_mo = torch.tensor(
        np.exp(1j * phi_grid[:, None] * m_arr[None, :]),
        device=wf_device,
        dtype=wf_dtype.to_complex(),
    )

    if continue_flag:
        last_done_sub_bank_ind = get_last_calculated_sub_bank(target_folder)
        sb_start = last_done_sub_bank_ind + 1
    else:
        sb_start = 0
    print(f"Starting calculation by sub-bank, starting from {sb_start}")
    # loop over sub-banks
    for sb, sub_bank in enumerate(sub_banks[sb_start:], start=sb_start):
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{datetime_str} : Starting sub bank {sb}")
        num_m_blocks = -(sub_bank.shape[0] // -h1_blocksize)

        # find appropriate h1 samples

        close_samples = select_from_df_by_closeness(
            intrinsic_sample_bank,
            sub_bank,
            ["mchirp", "chieff_hat"],
            distances,
        )

        overlaps_ij = np.zeros(
            (sub_bank.shape[0], close_samples.shape[0]),
            dtype=np.float32,
        )

        i_inds = sub_bank.original_index.values
        j_inds = close_samples.index.values

        num_n_blocks = -(close_samples.shape[0] // -h2_blocksize)

        # loop over m blocks
        for m in tqdm(
            range(num_m_blocks),
            total=num_m_blocks,
            desc=f"Sub-bank {sb}, {len(i_inds)}x{(len(j_inds))}",
        ):
            # m = 0
            i_rel_start = m * h1_blocksize
            i_rel_end = np.min([(m + 1) * h1_blocksize, sub_bank.shape[0]])
            i_rel_inds = np.arange(i_rel_start, i_rel_end)
            i_bank_inds = sub_bank.original_index.values[i_rel_inds]

            amp, phase = intrinsic_sample_processor.load_amp_and_phase(
                intrinsic_sample_processor.waveform_dir, i_bank_inds
            )
            amp = wf_tensor(amp)
            phase = wf_tensor(phase)
            h1_impb = amp * torch.exp(1j * phase)
            norm_io = ip_tensor(
                np.load(target_folder / "norm_io.npy")[i_bank_inds],
            )

            h1h1_iopp = ip_tensor(
                np.load(target_folder / "hh_iopp.npy")[i_bank_inds],
            )

            # loop over h2 samples
            for n in tqdm(range(num_n_blocks), total=num_n_blocks, leave=False):
                # n = 0
                j_rel_start = n * h2_blocksize
                j_rel_end = np.min([(n + 1) * h2_blocksize, close_samples.shape[0]])
                j_rel_inds = np.arange(j_rel_start, j_rel_end)
                j_abs_inds = close_samples.index.values[j_rel_inds]

                amp, phase = intrinsic_sample_processor.load_amp_and_phase(
                    intrinsic_sample_processor.waveform_dir, j_abs_inds
                )
                amp = wf_tensor(amp)
                phase = wf_tensor(phase)
                h2_jmpb = amp * torch.exp(1j * phase)

                norm_j = ip_tensor(
                    np.load(target_folder / "norm_io.npy")[j_abs_inds, 0],
                )
                h2_jmpb /= wf_tensor(torch.sqrt(flex_reshape(norm_j, "j", "jmpb")))
                h2h2_jpp = ip_tensor(
                    np.load(
                        target_folder / "hh_iopp.npy",
                    )[j_abs_inds, 0],
                )

                h1h2_itojpp = ip_tensor(
                    get_h1h2_itojpp(
                        weights, h1_impb, h2_jmpb, phasor_mo, timeshift_exp_bt
                    )
                    / flex_reshape(norm_io, "io", "itojpP")
                )

                overlaps_ij[i_rel_start:i_rel_end, j_rel_start:j_rel_end] = (
                    get_optimal_overlap_ijtopp(
                        h1h1_iopp,
                        h2h2_jpp,
                        h1h2_itojpp,
                        clip=True,
                        normalize=False,
                    )
                    .amax(dim=(-2, -1))
                    .cpu()
                )

        np.savez(
            target_folder / f"{basename}_sub_bank_{sb}.{filetype}",
            overlaps_ij=overlaps_ij,
            i_inds=i_inds,
            j_inds=j_inds,
        )


def calculate_overlap_matrix(
    bank_folder: Union[str, Path],
    target_folder: Union[str, Path] = None,
    i_start: int = 0,
    i_end: int = None,
    h1_blocksize: int = 16,
    h2_blocksize: int = 1024,
    n_t: int = 16,
    n_phi: int = 32,
    wf_dtype: Union[str, torch.dtype] = torch.float64,
    ip_dtype: Union[str, torch.dtype] = torch.float32,
    continue_flag: bool = True,
    mchirp_bins: int = 8,
    chieff_hat_bins: int = 8,
    wf_device: Union[str, Device] = "cpu",
    ip_device: Union[str, Device] = "cpu",
    precompute_blocksize: int = PRECOMPUTE_BLOCKSIZE,
):
    """
    Find and save wavefrom overlap information. This can be used for
    rejection sampling, e.g., if a waveform has low likelihood, all
    similar waveforms can be rejected.
    """
    bank_folder = Path(bank_folder)
    ip_dtype = torch_dtype(ip_dtype)
    wf_dtype = torch_dtype(wf_dtype)
    wf_device, ip_device = get_device_per_dtype(
        [wf_device, ip_device], [wf_dtype, ip_dtype]
    )
    if target_folder is None:
        target_folder = bank_folder / "overlaps"
    else:
        target_folder = Path(target_folder)
    start_time = time.time()

    # TODO: replace printing with logging
    print("Calculating overlap matrix of", bank_folder)
    print("Saving to", target_folder)
    print(f"Starting calculation at {time.ctime(start_time)}")

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        bank_size = bank_config["bank_size"]

    i_end = i_end if i_end else bank_size
    if not target_folder.exists():
        utils.mkdirs(target_folder)

    overlap_config_filename = target_folder / "overlaps_config.json"
    with open(overlap_config_filename, "w", encoding="utf-8") as fp:
        json.dump(
            dict(
                n_t=n_t,
                n_phi=n_phi,
                i_start=i_start,
                i_end=i_end,
                h1_blocksize=h1_blocksize,
                h2_blocksize=h2_blocksize,
                continue_flag=continue_flag,
                wf_device=torch.device(wf_device).type,
                ip_device=torch.device(ip_device).type,
                wf_dtype=str(wf_dtype),
                ip_dtype=str(ip_dtype),
                mchirp_bins=mchirp_bins,
                chieff_hat_bins=chieff_hat_bins,
            ),
            fp,
            indent=4,
            sort_keys=True,
        )

    compute_and_save_hh(
        bank_folder,
        target_folder,
        n_phi,
        blocksize=precompute_blocksize,
        device="cpu",
        wf_dtype=wf_dtype,
        ip_dtype=ip_dtype,
    )

    if continue_flag and overlap_config_filename.exists():
        with open(overlap_config_filename, "r", encoding="utf-8") as fp:
            existing_config = json.load(fp)
            if (
                existing_config["n_t"] != n_t
                or existing_config["n_phi"] != n_phi
                or existing_config["ip_dtype"] != str(ip_dtype)
            ):
                raise ValueError(
                    "Configuration mismatch: Existing configuration does not "
                    + "match the provided parameters."
                )

    _calculate_overlap_matrix_by_subbanks(
        bank_folder=bank_folder,
        target_folder=target_folder,
        i_start=i_start,
        i_end=i_end,
        h1_blocksize=h1_blocksize,
        h2_blocksize=h2_blocksize,
        n_t=n_t,
        n_phi=n_phi,
        ip_dtype=ip_dtype,
        wf_dtype=wf_dtype,
        continue_flag=continue_flag,
        mchirp_bins=mchirp_bins,
        chieff_hat_bins=chieff_hat_bins,
        wf_device=wf_device,
        ip_device=ip_device,
    )
    print("Overlap configuration saved to:", overlap_config_filename)
    end_of_calc_time = time.time()
    print(f"Calculation finished at {time.ctime(end_of_calc_time)}")
    print(f"overall time elapsed: {end_of_calc_time - start_time:.5g} seconds")

    end_time = time.time()
    time_elapsed = end_time - start_time

    print(f"Saving done at {time.ctime(end_time)}")
    print(f"Done! in {time_elapsed:.5g} seconds")


def collect_sub_bank_results(
    target_folder: Union[str, Path], shape: Union[None, Tuple[int, int]] = None
) -> lil_matrix:
    target_folder = Path(target_folder)
    files = sorted(
        target_folder.glob("overlap_matrix_sub_bank_*.npz"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    all_rows, all_cols, all_data = [], [], []

    for file in files:
        # Load the .npz file
        block = np.load(file)

        # Extract indices and values
        i_inds = block["i_inds"]  # Shape (n,)
        j_inds = block["j_inds"]  # Shape (m,)
        values = block["overlaps_ij"]  # Shape (n, m)

        # Ensure dimensions match
        if len(i_inds) != values.shape[0] or len(j_inds) != values.shape[1]:
            raise ValueError(f"Mismatch in dimensions for file: {file}")

        # Generate row and column indices for all non-zero elements
        rows = np.repeat(i_inds, len(j_inds))  # Expand rows for all columns
        cols = np.tile(j_inds, len(i_inds))  # Expand columns for all rows
        data = values.ravel()  # Flatten the values

        # Append data to the global lists
        all_rows.append(rows)
        all_cols.append(cols)
        all_data.append(data)

    # Concatenate all the data across files
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)
    shape = shape if shape else (final_rows.max() + 1,) * 2
    # Create a sparse matrix using the collected indices and data
    sparse_matrix = coo_matrix((final_data, (final_rows, final_cols)), shape=shape)

    return sparse_matrix.tolil()


def parse_arguments():
    """Parser for arguments"""
    parser = argparse.ArgumentParser(
        description="Calculate overlap matrix for waveforms."
    )
    parser.add_argument(
        "--bank_folder", type=Path, required=True, help="Waveform bank folder"
    )
    parser.add_argument(
        "--target_folder",
        type=Path,
        required=False,
        default=None,
        help="Folder to save overlap data. Defaults to '<bank_folder>/overlaps'.",
    )
    parser.add_argument(
        "--i_start",
        type=int,
        required=False,
        default=0,
        help="Starting index for waveforms (rows).",
    )
    parser.add_argument(
        "--i_end",
        type=int,
        required=False,
        default=None,  # Placeholder: Defaults to the size of the bank.
        help="Ending index for waveforms (rows). Defaults to the size of the bank.",
    )
    parser.add_argument(
        "--h1_blocksize",
        type=int,
        required=False,
        default=16,  # Placeholder: Choose appropriate block size.
        help="Block size for h1 waveforms.",
    )
    parser.add_argument(
        "--h2_blocksize",
        type=int,
        required=False,
        default=1024,
        help="Block size for h2 waveforms.",
    )
    parser.add_argument(
        "--n_t",
        type=int,
        required=False,
        default=16,
        help="Number of time shifts to consider.",
    )
    parser.add_argument(
        "--n_phi",
        type=int,
        required=False,
        default=32,
        help="Number of phi_ref points to consider.",
    )
    parser.add_argument(
        "--ip_dtype",
        type=str,
        required=False,
        default="float32",
        help="Data type for the overlap matrix (e.g., 'torch.float32').",
    )
    parser.add_argument(
        "--continue",
        dest="continue_flag",
        action="store_true",
        help="Flag to continue from the last result. Default is True.",
    )
    parser.add_argument(
        "--no-continue",
        dest="continue_flag",
        action="store_false",
        help="Flag to start from scratch, overwriting previous files.",
    )
    parser.add_argument(
        "--mchirp_bins",
        type=int,
        required=False,
        default=8,
        help="Number of bins for chirp mass.",
    )
    parser.add_argument(
        "--chieff_hat_bins",
        type=int,
        required=False,
        default=8,
        help="Number of bins for effective spin.",
    )
    parser.add_argument(
        "--wf_device",
        type=str,
        required=False,
        default="cpu",
        help="Device for waveform generation (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--ip_device",
        type=str,
        required=False,
        default="cpu",
        help="Device for inner product calculations (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--precompute_blocksize",
        type=int,
        required=False,
        default=None,  # Placeholder: Define appropriate default.
        help="Block size for precomputations.",
    )

    parser.set_defaults(continue_flag=True)
    return vars(parser.parse_args())


def profile_and_run(**kwargs):
    bank_folder = Path(kwargs["bank_folder"])
    target_folder = kwargs["target_folder"]
    if target_folder is None:
        target_folder = bank_folder / "overlaps"

    target_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiling_text_path = target_folder / f"profiling_results_{timestamp}.txt"
    profiling_binary_path = target_folder / f"profiling_results_{timestamp}.prof"
    print(f"Starting {__file__}")
    for k, v in kwargs.items():
        print(f"Parameter {k}={v}")
    profiler = cProfile.Profile()
    profiler.enable()

    calculate_overlap_matrix(**kwargs)

    profiler.disable()
    # Save the profiling results
    with open(profiling_text_path, "w", encoding="utf-8") as profile_file:
        stats = pstats.Stats(profiler, stream=profile_file)
        stats.sort_stats(pstats.SortKey.TIME)  # Sort by time spent in each function
        stats.print_stats()
    profiler.dump_stats(profiling_binary_path)  # Save the profile to a file

    print(f"Profiling results saved to: {profiling_binary_path}")


if __name__ == "__main__":
    kwargs = parse_arguments()
    profile_and_run(**kwargs)
