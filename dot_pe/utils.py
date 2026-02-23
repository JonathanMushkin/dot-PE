"""
Utility functions for the sampler_free module.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

from numpy.typing import NDArray

import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.special import logsumexp
from torch.types import Device

from cogwheel.data import EventData
from cogwheel import gw_utils
from cogwheel.utils import read_json


def validate_q_bounds(q_min: float, q_max: float) -> None:
    """Raise ValueError if q_min >= q_max or q_max > 1."""
    if q_min >= q_max:
        raise ValueError(f"q_min ({q_min}) must be strictly less than q_max ({q_max})")
    if q_max > 1.0:
        raise ValueError(f"q_max ({q_max}) must be <= 1.0")


def is_dtype_supported(device: Device, dtype: torch.dtype) -> bool:
    try:
        _ = torch.empty(1, dtype=dtype, device=device)
        return True
    except (RuntimeError, TypeError):
        return False


def get_device_for_dtype(preferred_device: Device, dtype: torch.dtype) -> Device:
    fallback = "cpu"
    return torch.device(
        preferred_device if is_dtype_supported(preferred_device, dtype) else fallback
    )


def get_device_per_dtype(
    preferred_devices: List[Device], dtypes: List[torch.dtype]
) -> List[Device]:
    """ "
    Args:
        preferred_devices (List[Device]): A list of preferred devices.
        dtypes (List[torch.dtype]): A list of data types corresponding
                                    to each device.

    Returns:
        List[Device]: A list of devices set for each dtype based on user
        preference.

    Example:
        get_device_per_dtype(["mps", "mps"], [torch.complex128,
        torch.float32])
        Output: [device(type='mps'), device(type='mps')]
    """

    return [
        get_device_for_dtype(device, dtype)
        for device, dtype in zip(preferred_devices, dtypes)
    ]


def torch_dtype(dtype):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.split(".")[-1])
    return dtype


def safe_move_and_cast(
    tensor: torch.Tensor, dtype: torch.dtype, device: Union[Device, str]
) -> torch.Tensor:
    try:
        output = tensor.to(device=device).to(dtype=dtype)
    except TypeError:
        try:
            output = tensor.to(dtype=dtype).to(device=device)
        except TypeError as e:
            message = (
                "Failed to move tensor "
                + f"from device {tensor.device} and dtype {tensor.dtype}"
                + f"to device {device} and dtype {dtype}: {e}"
            )
            raise RuntimeError(message)

    return output


def clear_cache(device: Device) -> None:
    if hasattr(getattr(torch, device), "empty_cache"):
        getattr(torch, device).empty_cache()
    gc.collect()


def setup_logger(
    log_dir: Union[str, Path],
    log_filename: str = "logging.log",
    level=logging.INFO,
):
    """
    Sets up a logger with both file and console handlers.

    Parameters
    ----------
    log_dir : str or Path
        Directory where the log file will be saved.
    log_filename : str, optional
        Filename for the log file. Default is 'logging.log'.
    level : int, optional
        Logging level. Default is logging.INFO.
    """
    log_path = Path(log_dir) / log_filename
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create and configure file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _find_shape(shape: tuple, origin: str, target: str):
    """
    return tuple of a shape to make an array of dimensions `origin`
    compatible with `target`.
    Example:
    shape = (64, 32)
    origin = "jo"
    target = "ijtob"
    print(_find_shape(shape, origin, target)) # (1, 64, 1, 32, 1, 1)
    """
    return tuple(shape[origin.index(s)] if s in origin else 1 for s in target)


def flex_reshape(arr, origin_string, target_string):
    """Flexible reshape. Reshape an array according to origin & target
    strings of shapes. Use 1 to missing dimensions.
    Example:
    y_itombp = y_itompb / flex_reshape(norm_io, "io", "itompb")
    """
    return arr.reshape(_find_shape(arr.shape, origin_string, target_string))


def get_event_data(event: Union[str, Path, EventData]) -> EventData:
    if isinstance(event, EventData):
        event_data = event
    else:
        try:
            event_data = EventData.from_npz(event)
        except FileNotFoundError:
            event_data = EventData.from_npz(filename=event)
    return event_data


def extract_single_detector_event_data(
    event_data: Union[str, Path, EventData], det_name: str
) -> EventData:
    """
    Extract event data for a single detector from multi-detector event data.

    Parameters
    ----------
    event_data : Union[str, Path, EventData]
        Event data (can be path or EventData object).
    det_name : str
        Detector name to extract.

    Returns
    -------
    EventData
        Event data filtered to contain only the specified detector.
    """
    import copy

    event_data_loaded = get_event_data(event_data)
    event_data_1d = copy.deepcopy(event_data_loaded)
    indices = [event_data_1d.detector_names.index(det) for det in list(det_name)]

    array_attributes = ["strain", "blued_strain", "wht_filter"]
    for attr in array_attributes:
        setattr(event_data_1d, attr, getattr(event_data_1d, attr)[indices])

    tuple_attributes = ["detector_names"]
    for attr in tuple_attributes:
        temp = tuple(np.take(getattr(event_data_1d, attr), indices))
        setattr(event_data_1d, attr, temp)

    if getattr(event_data_1d, "injection", None) is not None:
        event_data_1d.injection["h_h"] = np.take(
            event_data_1d.injection["h_h"], indices
        ).tolist()
        event_data_1d.injection["d_h"] = np.take(
            event_data_1d.injection["d_h"], indices
        ).tolist()

    return event_data_1d


def compute_lnq(m1, m2):
    """Compute log mass ratio: ln(q) = ln(m2/m1)."""
    return np.log(m2 / m1)


def load_intrinsic_samples_from_rundir(rundir: Union[str, Path]) -> pd.DataFrame:
    """
    Load weighted intrinsic samples from a previous inference run.

    Loads the bank and prob_samples from a run directory, aggregates weights
    by intrinsic index 'i', and adds transformed columns (mchirp, lnq, chieff).

    Parameters
    ----------
    rundir : str or Path
        Path to the inference run directory containing run_kwargs.json and
        prob_samples.feather.

    Returns
    -------
    pd.DataFrame
        DataFrame with bank rows corresponding to prob_samples indices,
        including a 'weights' column from aggregated prob_samples, and
        'mchirp', 'lnq', 'chieff' columns computed from the bank parameters.
    """
    rundir = Path(rundir)
    run_kwargs = read_json(rundir / "run_kwargs.json")
    bank_folder = Path(run_kwargs["bank_folder"])

    bank = pd.read_feather(bank_folder / "intrinsic_sample_bank.feather")
    prob_samples = pd.read_feather(rundir / "prob_samples.feather")

    aggregated_weights = prob_samples.groupby("i")["weights"].sum()

    indices = aggregated_weights.index.to_numpy(dtype=int)
    weighted_bank = bank.iloc[indices].copy().reset_index(drop=True)
    weighted_bank["i"] = indices
    weighted_bank["weights"] = aggregated_weights.values

    m1 = weighted_bank["m1"].to_numpy()
    m2 = weighted_bank["m2"].to_numpy()
    s1z = weighted_bank.get("s1z", pd.Series(np.zeros(len(weighted_bank)))).to_numpy()
    s2z = weighted_bank.get("s2z", pd.Series(np.zeros(len(weighted_bank)))).to_numpy()

    weighted_bank["mchirp"] = gw_utils.m1m2_to_mchirp(m1, m2)
    weighted_bank["lnq"] = compute_lnq(m1, m2)
    weighted_bank["chieff"] = gw_utils.chieff(m1, m2, s1z, s2z)

    return weighted_bank


def safe_logsumexp(x):
    """
    Calculate the log of the sum of exponentials of an array.
    Always returns a float64, even if x comes in as object-dtype.
    """
    # coerce to float64 ndarray
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return -np.inf
    # drop NaN/inf
    arr = arr[np.isfinite(arr)]
    return -np.inf if arr.size == 0 else logsumexp(arr)


@njit
def sample_k_from_N(N, k):
    """
    Sample k unique indices from range(N) in O(k) time (plus O(N) allocation).
    Returns a 1D int64 array of length k.
    """
    # Create an array [0, 1, ..., N-1]
    arr = np.arange(N, dtype=np.int64)
    # Perform partial Fisher–Yates shuffle for the first k positions
    for i in range(k):
        j = np.random.randint(i, N)  # random integer in [i, N)
        # swap arr[i] and arr[j]
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
    return arr[:k]


def validate_bank_configs(bank_paths: List[Path]) -> Dict:
    """
    Validate that all bank configs match on critical parameters.

    Parameters
    ----------
    bank_paths : List[Path]
        List of bank folder paths to validate.

    Returns
    -------
    Dict
        Shared bank config dictionary.

    Raises
    ------
    ValueError
        If any critical parameters mismatch across banks.
    """
    if len(bank_paths) == 0:
        raise ValueError("No bank paths provided for validation")

    configs = []
    for bank_path in bank_paths:
        config_path = bank_path / "bank_config.json"
        if not config_path.exists():
            raise ValueError(f"Bank config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            configs.append(json.load(f))

    # Get first config as reference (missing q_max interpreted as 1.0)
    ref_config = configs[0]
    ref_approximant = ref_config["approximant"]
    ref_fbin = np.array(ref_config["fbin"])
    ref_f_ref = ref_config["f_ref"]
    ref_m_arr = np.array(ref_config["m_arr"])
    ref_q_min = ref_config.get("q_min")
    ref_q_max = ref_config.get("q_max", 1.0)

    # Validate against reference
    for i, config in enumerate(configs[1:], start=1):
        errors = []

        if config["approximant"] != ref_approximant:
            errors.append(
                f"approximant mismatch: bank {i} has '{config['approximant']}', "
                f"expected '{ref_approximant}'"
            )

        if not np.array_equal(np.array(config["fbin"]), ref_fbin):
            errors.append(
                f"fbin mismatch: bank {i} has different frequency bins than reference"
            )

        if config["f_ref"] != ref_f_ref:
            errors.append(
                f"f_ref mismatch: bank {i} has {config['f_ref']}, expected {ref_f_ref}"
            )

        if not np.array_equal(np.array(config["m_arr"]), ref_m_arr):
            errors.append(
                f"m_arr mismatch: bank {i} has different m_arr than reference"
            )

        if ref_q_min is not None and config.get("q_min") != ref_q_min:
            errors.append(
                f"q_min mismatch: bank {i} has {config.get('q_min')}, "
                f"expected {ref_q_min}"
            )

        if config.get("q_max", 1.0) != ref_q_max:
            errors.append(
                f"q_max mismatch: bank {i} has {config.get('q_max', 1.0)}, "
                f"expected {ref_q_max}"
            )

        if errors:
            error_msg = "Bank config validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)

    return ref_config


def parse_bank_folders(
    bank_folder: Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]],
) -> Dict[str, Path]:
    """
    Parse bank_folder input into a dict mapping bank_id to bank_path.

    Accepts:
    - Single path (str or Path): treated as one bank
    - List/tuple of paths: multiple banks
    - Comma-separated string: multiple banks

    Returns:
    - Dict[str, Path] mapping bank_id (e.g., "bank_0", "bank_1", ...) to bank_path
    """
    if isinstance(bank_folder, str) and "," in bank_folder:
        bank_paths = [Path(p.strip()) for p in bank_folder.split(",")]
    elif isinstance(bank_folder, (list, tuple)):
        bank_paths = [Path(p) for p in bank_folder]
    else:
        bank_paths = [Path(bank_folder)]

    # Convert to Path and assign sequential IDs
    banks = {}
    for i, bank_path in enumerate(bank_paths):
        bank_path = Path(bank_path)
        if not bank_path.exists():
            raise ValueError(f"Bank folder does not exist: {bank_path}")
        bank_id = f"bank_{i}"
        banks[bank_id] = bank_path

    return banks


def inds_to_blocks(
    indices: NDArray[np.int_], block_size: int
) -> List[NDArray[np.int_]]:
    """Split the indices into blocks of size blocksize (or less)."""
    return [
        indices[i * block_size : (i + 1) * block_size]
        for i in range(-(len(indices) // -block_size))
    ]
