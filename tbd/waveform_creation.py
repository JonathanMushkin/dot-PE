"""
Create linear free waveforms.
"""

import argparse
import logging
import os
import subprocess
import numpy as np
import pandas as pd
import json
import time
from multiprocessing.pool import Pool
from pathlib import Path

from cogwheel import data, waveform, utils
from cogwheel.sampler_free import config
from cogwheel.sampler_free.sampler_free_utils import setup_logger


def get_waveform(wfg, int_dic, fbin, override_dic=None):
    """

    Parameters
    ----------
    wfg : WaveformGenerator object
        waveform generator object
    int_dic : dictionary
        intrinsic parameters
    fbin : array
        frequency bins of relative binning
    override_dic : dictionary, optional
        override the default parameters in int_dic

    Returns
    -------
    amp_mpb : array
        amplitude of the linear free waveform
    phase_mpb : array
        phase of the linear free waveform
    """
    # use default parameters for l1, l2, f_ref, d_luminoisty and phi_ref
    override_dic = config.DEFAULT_PARAMS_DICT if override_dic is None else override_dic
    waveform_dic = int_dic | override_dic if override_dic else int_dic
    h_mpb = wfg.get_hplus_hcross(fbin, waveform_dic, by_m=True)
    amp_mpb = np.abs(h_mpb)
    # unwrap in fine grid, then take sparser fbin points
    phase_mpb = np.unwrap(np.angle(h_mpb), axis=-1)
    # shift forward in time by dt_linfree

    return amp_mpb, phase_mpb


def _gen_waveforms_from_samples_dataframe(wfg, samples, fbin, override_dic=None):
    """
    Generate waveforms for the samples in the dataframe.

    Parameters
    ----------
    samples : Pandas.dataframe
        samples to generate waveforms for
    fs : array
        frequency array to generate the waveform on
    fbin : array
        frequency bins of relative binning
    fbin_inds : array, optional
        indices of the frequency bins
    override_dic : dictionary, optional
        override the default parameters in int_dic. If None,
        the default parameters are used. Pass empty dictionary to
        not override any parameters.
    """
    if override_dic is None:
        override_dic = config.DEFAULT_PARAMS_DICT

    n = samples.shape[0]
    n_modes = len(wfg.m_arr)
    amp_impb = np.zeros((n, n_modes, 2, len(fbin)))
    phase_impb = np.zeros((n, n_modes, 2, len(fbin)))

    for i, sample in samples.iterrows():
        (
            amp_impb[i],
            phase_impb[i],
        ) = get_waveform(wfg, sample.to_dict(), fbin, override_dic=override_dic)

    return amp_impb, phase_impb


def _gen_waveforms_from_index(
    wfg,
    samples,
    i,
    waveform_dir,
    blocksize,
    fbin,
    override_dic=None,
    logger=None,
    t0=None,
):
    """
    Create a dataframe from the index, and genetate waveforms
    for the samples in the dataframe.
    Waveforms are saved to disk, in the arrays, see
    _gen_waveforms_from_samples_dataframe for details and
    `names`, `block_str` for the file names.
    """

    # set up logger if not provided
    logger = logger or logging.getLogger(__name__)
    t0 = t0 or time.time()

    logger.info(f"block {i} started")
    # define the block
    r = range(i * blocksize, (i + 1) * blocksize)
    samples = samples.iloc[r].copy()
    first = samples.index[0]
    last = samples.index[-1]
    logger.info(
        f"{i}:{first}->{last} started. "
        + f"Time passed {time.time() - t0:.3g} seconds."
    )

    # generate waveforms
    samples.index = range(samples.shape[0])
    arrays = _gen_waveforms_from_samples_dataframe(
        wfg, samples, fbin, override_dic=override_dic
    )
    names = ["amplitudes", "phase"]
    block_str = f"_block_{i}"
    for arr, name in zip(arrays, names):
        filename = waveform_dir / (name + block_str + ".npy")
        np.save(file=filename, arr=arr)

    logger.info(
        f"Block {i}: indices {first}->{last} ended. Time passed "
        + f"{time.time() - t0:.3g} seconds."
    )
    return (i, first, last)


def parse_arguments(arguments=None):
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--waveform_dir",
        dest="waveform_dir",
        type=Path,
        default=None,
        help="""where to save the waveforms.""",
    )
    parser.add_argument(
        "--samples_path",
        dest="samples_path",
        type=Path,
        help="""path to the intrinsic samples dataframe.""",
    )
    parser.add_argument(
        "--n_pool",
        dest="n_pool",
        type=int,
        default=1,
        help="""number of processes to use for multiprocessing.""",
    )
    parser.add_argument(
        "--blocksize",
        dest="blocksize",
        type=int,
        default=4096,
        help="""number of samples in each block.""",
    )
    parser.add_argument(
        "--i_start",
        dest="i_start",
        type=int,
        default=0,
        help="""index of the first block to generate.""",
    )
    parser.add_argument(
        "--i_end",
        dest="i_end",
        type=int,
        default=None,
        help="""index of the last block to generate.""",
    )
    parser.add_argument(
        "--approximant",
        dest="approximant",
        type=str,
        default="IMRPhenomXODE",
        help="""Approximant to use for waveform generation.""",
    )
    par_args = (
        parser.parse_args() if arguments is None else parser.parse_args(arguments)
    )
    # if (par_args.n_blocks is None) == (par_args.blocksize is None):
    #     raise ValueError('Either n_blocks or blocksize must be provided.')

    if par_args.waveform_dir is None:
        par_args.waveform_dir = par_args.samples_path.parent / "waveforms"
    return par_args


def _set_blocksize_n_blocks(n_blocks, blocksize, n_samples):
    # -(a//-b) is ceil devision, equivalent to int(math.ceil(a/b))
    if n_blocks is None:
        n_blocks = -(n_samples // -blocksize)
    if blocksize is None:
        blocksize = -(n_samples // -n_blocks)
    return n_blocks, blocksize


def create_waveform_bank_from_samples(
    samples_path,
    bank_config_path=None,
    n_blocks=None,
    waveform_dir=None,
    n_pool=1,
    blocksize=4096,
    i_start=0,
    i_end=None,
    approximant="IMRPhenomXODE",
):
    """
    Load a sample bank from a feather file, generate linear-free time
    convention frequency-domain waveforms, and save them to disk in
    the waveform_dir.

    Parameters
    samples_path : posix path
        path to the intrinsic samples dataframe
    n_blocks : int, optional
        number of blocks to divide the samples into
    waveform_dir : posix path, optional
        path to the directory to save the waveforms
    n_pool : int, optional
        number of processes to use for multiprocessing
    blocksize : int, optional
        number of samples in each block
    i_start : int, optional
        index of the first block to generate
    i_end : int, optional
        index of the last block to generate
    """

    start_time = time.time()
    if not waveform_dir.exists():
        utils.mkdirs(waveform_dir)
    logger = setup_logger(waveform_dir)
    logger.info("%s started at %s", __name__, time.ctime(start_time))

    if bank_config_path is not None:
        with open(bank_config_path, "r", encoding="utf-8") as fp:
            config_dict = json.load(fp)
            fbin = np.array(config_dict["fbin"])
            f_ref = config_dict["f_ref"]
    else:
        fbin = config.DEFAULT_FBIN
        f_ref = config.DEFAULT_F_REF

    intrinsic_samples = pd.read_feather(samples_path)
    event_data = data.EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)

    # access waveform.APPROXIMANTS only after wfg is created, due to
    # importing scheme. May crash otherwise.
    wfg = waveform.WaveformGenerator.from_event_data(event_data, approximant)
    if waveform.APPROXIMANTS[approximant].aligned_spins:
        intrinsic_samples[["s1x_n", "s1y_n", "s2x_n", "s2y_n"]] = 0.0
        intrinsic_samples.to_feather(samples_path)

    n_samples = intrinsic_samples.shape[0]
    n_blocks, blocksize = _set_blocksize_n_blocks(n_blocks, blocksize, n_samples)

    # allow for partial generation of the waveforms, in case
    # part of the waveforms are already done
    if i_end is None:
        i_end = n_blocks
    override_dic = config.DEFAULT_PARAMS_DICT | {"f_ref": f_ref}
    if n_pool and n_pool > 1:
        with Pool(n_pool) as pool:
            _ = pool.starmap(
                _gen_waveforms_from_index,
                [
                    (
                        wfg,
                        intrinsic_samples,
                        i,
                        waveform_dir,
                        blocksize,
                        fbin,
                        override_dic,
                        logger,
                        start_time,
                    )
                    for i in range(i_start, i_end)
                ],
            )
    else:
        for i in range(i_start, i_end):
            _gen_waveforms_from_index(
                wfg,
                intrinsic_samples,
                i,
                waveform_dir,
                blocksize,
                fbin,
                override_dic,
                logger,
                start_time,
            )

    runtime_seconds = time.time() - start_time
    runtime_minutes = runtime_seconds / 60
    logger.info(
        f"{__name__} finished after "
        + f"{runtime_seconds:.3g} seconds "
        + f"({runtime_minutes:.3g} minutes)."
    )
    with open(bank_config_path, "r", encoding="utf-8") as fp:
        config_dict = json.load(fp)
    config_dict["approximant"] = approximant
    config_dict["m_arr"] = wfg.m_arr.tolist()
    config_dict["blocksize"] = blocksize

    with open(bank_config_path, "w", encoding="utf-8") as fp:
        json.dump(config_dict, fp, indent=4)


def submit_to_lsf(
    samples_path,
    waveform_dir=None,
    n_pool=1,
    blocksize=4096,
    i_start=0,
    i_end=None,
    approximant="IMRPhenomXODE",
    cwd=None,
):
    """
    Submit the waveform generation to the LSF cluster.
    """
    script_name = Path(__file__).resolve().as_posix()
    command = [
        "bsub",
        "python",
        script_name,
        "--waveform_dir",
        str(waveform_dir),
        "--samples_path",
        str(samples_path),
        "--n_pool",
        str(n_pool),
        "--blocksize",
        str(blocksize),
        "--i_start",
        str(i_start),
        "--i_end",
        str(i_end),
        "--approximant",
        str(approximant),
    ]
    subprocess.run(command, check=True, cwd=cwd)


if __name__ == "__main__":
    args = parse_arguments()
    if not args.waveform_dir.exists():
        utils.mkdirs(args.waveform_dir)

    create_waveform_bank_from_samples(
        samples_path=args.samples_path,
        waveform_dir=args.waveform_dir,
        n_pool=args.n_pool,
        blocksize=args.blocksize,
        i_start=args.i_start,
        i_end=args.i_end,
        approximant=args.approximant,
    )
