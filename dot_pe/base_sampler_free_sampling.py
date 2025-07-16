"""A module to perform sampler-free Bayesian inference on gravitational wave."""

import logging
import time
from math import ceil
from pathlib import Path

import numpy as np
from numba import njit

from cogwheel import gw_plotting, plotting

_LABELS = gw_plotting._LABELS
_UNITS = gw_plotting._UNITS
_LABELS["ln_posterior"] = r"$\ln \mathcal{P}$"
GB = 2**30

# create LatexLabels object for corner plots
latex_labels = plotting.LatexLabels(_LABELS, _UNITS)


def calculate_optimal_splits(i, e, max_size, size_func, splits_i=1, splits_e=1):
    """
    Given the maximal number of intrinsic and extrinsic samples,
    and the maximal size of the block in bytes, calculate the optimal
    number of splits in the intrinsic and extrinsic dimensions.

    Parameters:
    - i (int): The maximal number of intrinsic samples.
    - e (int): The maximal number of extrinsic samples.
    - max_size (int): The maximal size of the block in bytes.
    - size_func (function): A function that calculates the size of the
        block given the number of intrinsic and extrinsic samples.
    - splits_i (int): The current number of splits in the intrinsic
        dimension.
    - splits_e (int): The current number of splits in the extrinsic
        dimension.

    Returns:
    - splits_i (int): The optimal number of splits in the intrinsic
        dimension.
    - splits_e (int): The optimal number of splits in the extrinsic
        dimension.
    """
    # Calculate the size of the current block
    block_size = size_func(ceil(i / splits_i), ceil(e / splits_e))

    # Base case: if the block size is within the max size,
    # return the current split counts
    if block_size <= max_size:
        return splits_i, splits_e

    # Recursive case: split the longer dimension to make the blocks
    # square-like
    if i / splits_i >= e / splits_e:
        # Split the 'i' dimension
        return calculate_optimal_splits(
            i, e, max_size, size_func, splits_i + 1, splits_e
        )
    # Split the 'e' dimension
    return calculate_optimal_splits(i, e, max_size, size_func, splits_i, splits_e + 1)


@njit
def get_top_n_indices_two_pointer(x, y, N):
    lx, ly = x.size, y.size
    if lx == 0:
        start = max(ly - N, 0)
        out_y = np.arange(start, ly, dtype=np.int64)
        return np.empty(0, np.int64), out_y

    N = min(N, lx + ly)
    ix = np.empty(N, np.int64)
    iy = np.empty(N, np.int64)
    px, py = lx - 1, ly - 1
    cx = cy = 0

    while cx + cy < N and (px >= 0 or py >= 0):
        if px >= 0 and (py < 0 or x[px] >= y[py]):
            ix[cx] = px
            px -= 1
            cx += 1
        else:
            iy[cy] = py
            py -= 1
            cy += 1

    # reverse in-place
    for i in range(cx // 2):
        ix[i], ix[cx - 1 - i] = ix[cx - 1 - i], ix[i]
    for j in range(cy // 2):
        iy[j], iy[cy - 1 - j] = iy[cy - 1 - j], iy[j]

    return ix[:cx], iy[:cy]


def get_n_effective_total_i_e(samples, assume_noramlized=False):
    """
    Calculate the effective number of samples for the total,
    intrinsic, and extrinsic samples.

    Parameters
    ----------
    samples : pandas.DataFrame,
        Has columns 'weights', 'i', 'e'.
    assume_normalized: Boolian,
        if assume_normalized==False, normalization is imposed in
        the code.

    Return
    ------
    n_effective : float,
        effective number of samples (overall).
    n_effective_i : float,
        effective number of intrinsic samples.
    n_effective_e : float,
        effective number of extrinsic samples.
    """

    if samples is None or len(samples) == 0:
        return 0, 0, 0
    if not assume_noramlized:
        samples = samples.copy()
        samples["weights"] /= samples["weights"].sum()

    p_i = samples.groupby("i")["weights"].sum().values
    p_e = samples.groupby("e")["weights"].sum().values
    p = samples["weights"].values

    n_effective = 1 / np.sum(p**2)
    n_effective_i = 1 / np.sum(p_i**2)
    n_effective_e = 1 / np.sum(p_e**2)

    return n_effective, n_effective_i, n_effective_e


class Loggable:
    """
    A class with the ability to record log entries to file.
    """

    def __init__(
        self,
    ):
        self.setup_logger()

    def setup_logger(
        self,
        rundir=None,
        log_filename=None,
        level=logging.INFO,
        print_to_console=True,
        unique_id=None,
    ):
        """
        Create a logger object.
        TODO: make a wrapper around sampler_free_utils.setup_logger

        This function creates a logger object that can be used for logging
        messages.

        Parameters:
        - run_dir (str): The directory where the log file will be saved.
            If None, the log file will be saved in the current directory.
        - log_filename (str): The name of the log file.
            If None, the logger will not save to a file.
        - level (int): The logging level. Default is logging.INFO.
        - print_to_console (bool): Whether to print log messages to the
            console. Default is True.

        """
        if unique_id is None:
            unique_id = time.strftime("%Y%m%d-%H%M%S")
        logger_name = f"{__name__}_{unique_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.handlers = []  # remove any existing handlers

        log_format = "%(asctime)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)

        if log_filename:
            if rundir:
                log_path = Path(rundir) / log_filename
            else:
                log_path = Path.cwd() / log_filename
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if print_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self.logger = logger

    def log(self, msg, level=logging.INFO):
        """log a message, either to file and/or console, according the
        logger settings."""

        if self.logger is not None:
            self.logger.log(level, msg)
