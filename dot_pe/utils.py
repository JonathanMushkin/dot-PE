"""
Utility functions for the sampler_free module.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from numpy.typing import NDArray

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import logsumexp

from cogwheel.data import EventData
from cogwheel import gw_utils
from cogwheel.utils import read_json
from cogwheel.waveform import APPROXIMANTS, WaveformGenerator


def validate_q_bounds(q_min: float, q_max: float) -> None:
    """Raise ValueError if q_min >= q_max or q_max > 1."""
    if q_min >= q_max:
        raise ValueError(f"q_min ({q_min}) must be strictly less than q_max ({q_max})")
    if q_max > 1.0:
        raise ValueError(f"q_max ({q_max}) must be <= 1.0")


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


def normalize_harmonic_modes(
    harmonic_modes: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """Convert JSON-style [[l, m], ...] to list of (l, m) tuples."""
    return [tuple(int(x) for x in mode) for mode in harmonic_modes]


def m_arr_from_harmonic_modes(
    harmonic_modes: Sequence[Tuple[int, int]],
) -> NDArray[np.int_]:
    """
    Unique m values in order of first appearance.

    Order matches cogwheel ``WaveformGenerator``.
    """
    seen = []
    for _l, m in harmonic_modes:
        if m not in seen:
            seen.append(m)
    return np.asarray(seen, dtype=int)


def default_harmonic_modes_for_approximant(
    approximant: str,
) -> List[Tuple[int, int]]:
    """
    Default (l, m) pairs for an approximant.

    Same source as ``WaveformGenerator`` (``waveform.APPROXIMANTS``).
    """
    if approximant == "IMRPhenomXODE":
        # Registers IMRPhenomXODE in APPROXIMANTS (side effect on import).
        from cogwheel.waveform_models import xode as _  # noqa: F401

    if approximant not in APPROXIMANTS:
        raise ValueError(
            f"Unknown approximant {approximant!r}; "
            "add it to cogwheel.waveform.APPROXIMANTS."
        )
    return [tuple(mode) for mode in APPROXIMANTS[approximant].harmonic_modes]


def harmonic_modes_from_config(
    bank_config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """
    Resolve harmonic_modes from bank config.

    If harmonic_modes is present, use it. Otherwise filter the approximant's
    default modes to those with m in bank_config['m_arr'] (legacy banks).
    """
    if "harmonic_modes" in bank_config:
        return normalize_harmonic_modes(bank_config["harmonic_modes"])

    approximant = bank_config["approximant"]
    if "m_arr" not in bank_config:
        raise ValueError("bank_config must contain 'harmonic_modes' or 'm_arr'")
    m_allowed = set(int(m) for m in bank_config["m_arr"])
    return [
        mode
        for mode in default_harmonic_modes_for_approximant(approximant)
        if mode[1] in m_allowed
    ]


def resolve_bank_modes(
    bank_config: Dict[str, Any],
) -> Tuple[List[Tuple[int, int]], NDArray[np.int_]]:
    """
    Return (harmonic_modes, m_arr) for a bank config.

    Legacy (m_arr only): returned m_arr equals config m_arr unchanged.

    New (harmonic_modes): m_arr is derived; if m_arr is also present it must
    match.
    """
    harmonic_modes = harmonic_modes_from_config(bank_config)
    derived_m_arr = m_arr_from_harmonic_modes(harmonic_modes)

    if "harmonic_modes" in bank_config:
        if "m_arr" in bank_config:
            config_m_arr = np.asarray(bank_config["m_arr"], dtype=int)
            if not np.array_equal(config_m_arr, derived_m_arr):
                raise ValueError(
                    "bank_config m_arr does not match m values derived from "
                    f"harmonic_modes: config {config_m_arr.tolist()}, "
                    f"derived {derived_m_arr.tolist()}"
                )
        return harmonic_modes, derived_m_arr

    return harmonic_modes, np.asarray(bank_config["m_arr"], dtype=int)


def harmonic_modes_to_json(
    harmonic_modes: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Serialize harmonic_modes for bank_config.json."""
    normalized = normalize_harmonic_modes(harmonic_modes)
    return [[int(l), int(m)] for l, m in normalized]


def waveform_generator_from_config(
    event_data: EventData,
    bank_config: Dict[str, Any],
    **kwargs,
) -> WaveformGenerator:
    """Build WaveformGenerator using modes resolved from bank_config."""
    harmonic_modes, _ = resolve_bank_modes(bank_config)
    return WaveformGenerator.from_event_data(
        event_data,
        bank_config["approximant"],
        harmonic_modes=harmonic_modes,
        **kwargs,
    )


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
    ref_config.setdefault("q_max", 1.0)
    ref_approximant = ref_config["approximant"]
    ref_fbin = np.array(ref_config["fbin"])
    ref_f_ref = ref_config["f_ref"]
    ref_m_arr = np.array(ref_config["m_arr"])
    ref_has_harmonic_modes = "harmonic_modes" in ref_config
    if ref_has_harmonic_modes:
        ref_harmonic_modes = harmonic_modes_from_config(ref_config)

    # Validate against reference (q_min/q_max are intentionally not checked:
    # regular banks and low-q banks have non-overlapping q ranges by design)
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

        if ref_has_harmonic_modes:
            if "harmonic_modes" not in config:
                errors.append(
                    f"harmonic_modes missing: bank {i} has no harmonic_modes "
                    "but reference bank does"
                )
            elif harmonic_modes_from_config(config) != ref_harmonic_modes:
                errors.append(
                    f"harmonic_modes mismatch: bank {i} has different "
                    "harmonic_modes than reference"
                )
            _, derived_m = resolve_bank_modes(config)
            if not np.array_equal(derived_m, np.array(config["m_arr"])):
                errors.append(f"bank {i}: m_arr inconsistent with harmonic_modes")

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
