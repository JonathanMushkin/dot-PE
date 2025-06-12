"""A module to perform sampler-free Bayesian inference on gravitational wave."""

# Import standard modules
import os
from abc import ABC, abstractmethod
import sys
from pathlib import Path
import json
import textwrap
import inspect
import logging
from cProfile import Profile
import time
from numba import njit


# load modules affected by the environment variable
from math import ceil
import numpy as np

from cogwheel import gw_plotting, plotting
from cogwheel.utils import JSONMixin, DIR_PERMISSIONS, FILE_PERMISSIONS, mkdirs
from tbd import config, evidence_calculator, posterior
from tbd.sampler_free_utils import (
    setup_logger,
    safe_logsumexp,
    get_device_per_dtype,
    torch_dtype,
)
from tbd.marginalization import (
    MarginalizationExtrinsicSamplerFreeLikelihood,
)

_LABELS = gw_plotting._LABELS
_UNITS = gw_plotting._UNITS
# add labels and units here if needed
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


class BaseSamplerFreeSampler(ABC, JSONMixin):
    """
    See https://en.wiktionary.org/wiki/soapless_soap

    A base class that contains the common methods and attributes for
    SamplerFreeSampler. This class should not be instantiated directly.
    Contains methods for setting up the logger, profiling, and saving
    the sampler to a JSON file. Subclasses should implement the _run
    method, and any other methods that are specific to the subclass.
    """

    PROFILING_FILENAME = "profiling"
    JSON_FILENAME = "SamplerFreeSampler.json"

    DEFAULT_LIKELIHOOD_KWARGS = {
        "fbin": config.DEFAULT_FBIN,
        "pn_phase_tol": None,
    }

    DEFAULT_REF_WF_FINDER_KWARGS = {"time_range": (-1e-1, +1e-1)}

    DEFAULT_POSTERIOR_KWARGS = {
        "mchirp_guess": None,
        "likelihood_class": MarginalizationExtrinsicSamplerFreeLikelihood,
        "prior_class": "IntrinsicIASPrior",
        "approximant": "IMRPhenomXODE",
    }

    DEFAULT_LOGGER_KWARGS = {
        "log_filename": "logging.txt",
        "level": logging.INFO,
        "print_to_console": False,
    }
    DEFAULT_PLOTSTYLE_KWARGS = {"smooth": 1, "tail_probability": 1e-3}

    def __init__(
        self,
        intrinsic_bank_file,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood,
        prior,
        seed=None,
        dir_permissions=DIR_PERMISSIONS,
        file_permissions=FILE_PERMISSIONS,
    ):
        self.dir_permissions = dir_permissions
        self.file_permissions = file_permissions
        self.prior = prior
        self.intrinsic_bank_file = Path(intrinsic_bank_file)
        self.waveform_dir = Path(waveform_dir)
        self.likelihood = likelihood
        self.intrinsic_sample_processor = evidence_calculator.IntrinsicSampleProcessor(
            self.likelihood, self.waveform_dir
        )

        self.extrinsic_sample_processor = evidence_calculator.ExtrinsicSampleProcessor(
            self.likelihood.event_data.detector_names
        )

        self.evidence = evidence_calculator.Evidence(n_phi=n_phi, m_arr=np.array(m_arr))

        self.dh_weights_dmpb, self.hh_weights_dmppb = (
            self.intrinsic_sample_processor.get_summary()
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.cur_rundir = None  # current rundir
        self.logger = None

    def get_init_dict(self):
        return {
            "intrinsic_bank_file": str(self.intrinsic_bank_file),
            "waveform_dir": str(self.waveform_dir),
            "n_phi": self.evidence.n_phi,
            "m_arr": self.evidence.m_arr,
            "likelihood": self.likelihood,
            "prior": self.prior,
            "seed": self.seed,
        }

    @classmethod
    def from_posterior(cls, post, **init_kwargs):
        """
        Create a SamplerFreeSampler object from a posterior.Posterior
        object.
        """
        waveform_dir = init_kwargs.get("waveform_dir")

        waveform_dir = Path(waveform_dir)
        bank_config_path = waveform_dir.parent / "bank_config.json"
        if bank_config_path.is_file():
            with open(bank_config_path, "r", encoding="utf-8") as fp:
                f_ref = json.load(fp)["f_ref"]
        else:
            f_ref = config.DEFAULT_F_REF

        prior = post.prior.reinstantiate(f_ref=f_ref)
        likelihood = post.likelihood
        return cls(likelihood=likelihood, prior=prior, **init_kwargs)

    @classmethod
    def from_event(
        cls,
        event,
        posterior_kwargs=None,
        likelihood_kwargs=None,
        ref_wf_finder_kwargs=None,
        **init_kwargs,
    ):
        """
        Create a SamplerFreeSampler object from an data.EventData object.
        """
        post = cls.get_posterior(
            event, posterior_kwargs, likelihood_kwargs, ref_wf_finder_kwargs
        )

        return cls.from_posterior(post, **init_kwargs)

    @classmethod
    def get_posterior(
        cls,
        event,
        posterior_kwargs=None,
        likelihood_kwargs=None,
        ref_wf_finder_kwargs=None,
    ):
        """
        Create a Posterior object from an event.
        """
        # likelihoow_kwargs are ref_wf_finder_kwargs are taken from the
        # following, from highest to lowest priority:
        # 1. explicit likelihood_kwargs and ref_wf_finder_kwargs
        # 2. entries of posterior_kwargs
        # 3. default values

        posterior_kwargs = cls.DEFAULT_POSTERIOR_KWARGS | (posterior_kwargs or {})
        likelihood_kwargs = (
            cls.DEFAULT_LIKELIHOOD_KWARGS
            | posterior_kwargs.get("likelihood_kwargs", {})
            | (likelihood_kwargs or {})
        )
        ref_wf_finder_kwargs = (
            cls.DEFAULT_REF_WF_FINDER_KWARGS
            | posterior_kwargs.get("ref_wf_finder_kwargs", {})
            | (ref_wf_finder_kwargs or {})
        )

        posterior_kwargs["likelihood_kwargs"] = likelihood_kwargs
        posterior_kwargs["ref_wf_finder_kwargs"] = ref_wf_finder_kwargs

        post = posterior.SamplerFreePosterior.from_event(
            event=event, **posterior_kwargs
        )

        return post

    def log(self, msg, level=logging.INFO):
        """log a message, either to file and/or console, according the
        logger settings."""
        if self.logger is not None:
            self.logger.log(level, msg)

    def get_byte_size(self, i, e):
        """
        Calculate the byte-size of arrays within likelihood block.
        """
        p = 2
        m = len(self.evidence.m_arr)
        mm = len(self.evidence.m_inds)
        d = len(self.likelihood.event_data.detector_names)
        b = len(self.likelihood.fbin)
        o = self.evidence.n_phi
        float_bytes = 8
        complex_bytes = 16
        fudge = 1  # steps after dh_ieo, hh_ieo

        # account for sizes of  arrays
        size = (
            d * m * p * b * complex_bytes  # dh_weights_dmpb
            + d * mm * p * p * b * complex_bytes  # hh__weights_dmppb
            + i * m * p * b * complex_bytes  # h_impb
            + d * b * e * complex_bytes  # timeshifts_dpe
            + d * float_bytes  # asd_drift_d
            + d * p * e * float_bytes  # repsonse_dpe
            + i * e * m * complex_bytes  # dh_iem
            + i * e * mm * complex_bytes  # hh_iem
            + i * e * o * float_bytes * (1 + fudge)  # dh_ieo
            + i * e * o * float_bytes * (1 + fudge)
        )  # hh_ieo

        return size

    def _block_name(self, i_block, e_block):
        """Return the name of the block file."""
        return f"block_{i_block}_{e_block}.npz"

    @staticmethod
    def get_n_effective_total_i_e(samples, assume_noramlized=False):
        return get_n_effective_total_i_e(samples, assume_noramlized)

    @abstractmethod
    def _run(self):
        """Implemented in subclasses."""

    @abstractmethod
    def postprocess_rundir(self, rundir):
        """Implemented in subclasses."""

    def run(self, *args, **kwargs):
        """
        Perfoms over-head tasks (path control, logging, profiling) and
        calls the _run method.
        *args, **kwargs should contain the arguments for the _run method,
        **kwargs can also contain `logger_kwargs` for the logger.
        see setup_logger for details.
        """
        rundir = (
            kwargs.get("rundir") if "rundir" in kwargs else args[0] if args else None
        )
        rundir = Path(rundir)
        self.cur_rundir = rundir
        tempdir = rundir / "temp"
        mkdirs(tempdir, self.dir_permissions)
        self.to_json(
            rundir,
            dir_permissions=self.dir_permissions,
            file_permissions=self.file_permissions,
            overwrite=True,
        )

        # create logger
        logger_kwargs = kwargs.pop("logger_kwargs", None)  # remove from _run
        logger_kwargs = self.DEFAULT_LOGGER_KWARGS | (logger_kwargs or {})
        logger_kwargs |= {"rundir": rundir}
        self.logger = setup_logger(**logger_kwargs)
        # save run_kwargs to a file
        sig = inspect.signature(self._run)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, argument in bound.arguments.items():
            if isinstance(argument, Path):
                bound.arguments[name] = str(argument)
            if isinstance(argument, np.ndarray):
                bound.arguments[name] = argument.tolist()
            if isinstance(argument, dict):
                for key, value in argument.items():
                    if isinstance(value, np.ndarray):
                        argument[key] = value.tolist()
        with open(rundir / "_run_kwargs.json", "w", encoding="utf-8") as f:
            json.dump(bound.arguments, f)

        self.log(f"Run started, rundir:\n{rundir}")
        with Profile() as profiler:
            self._run(*args, **kwargs)
        profiler.dump_stats(rundir / self.PROFILING_FILENAME)

        # after the run: change premissions of the files
        for path in rundir.iterdir():
            if path.is_dir():
                path.chmod(self.dir_permissions)
            elif path.is_file():
                path.chmod(self.file_permissions)

        for path in tempdir.iterdir():
            if path.is_dir():
                path.chmod(self.dir_permissions)
            elif path.is_file():
                path.chmod(self.file_permissions)

    def submit_lsf(
        self, rundir, n_hours_limit=48, memory_per_task="32G", resuming=False
    ):
        """
        Parameters
        ----------
        rundir: path of run directory, e.g. from `self.get_rundir`
        n_hours_limit: Number of hours to allocate for the job
        memory_per_task: Determines the memory and number of cpus
        resuming: bool, whether to attempt resuming a previous run if
                  rundir already exists.
        """
        rundir = Path(rundir)
        job_name = "_".join(
            [
                self.__class__.__name__,
                self.prior.__class__.__name__,
                self.likelihood.event_data.eventname,
                rundir.name,
            ]
        )

        stdout_path = rundir.joinpath("output.out").resolve()
        stderr_path = rundir.joinpath("errors.err").resolve()

        self.to_json(rundir, overwrite=resuming)

        package = Path(__file__).parents[1].resolve()
        module = f"cogwheel.{os.path.basename(__file__)}".removesuffix(".py")

        batch_path = rundir / "batchfile"
        with open(batch_path, "w+", encoding="utf-8") as batchfile:
            batchfile.write(
                textwrap.dedent(
                    f"""\
                #BSUB -J {job_name}
                #BSUB -o {stdout_path}
                #BSUB -e {stderr_path}
                #BSUB -M {memory_per_task}
                #BSUB -W {n_hours_limit:02}:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ["CONDA_DEFAULT_ENV"]}

                cd {package}
                {sys.executable} -m {module} {rundir.resolve()}
                """
                )
            )
        batch_path.chmod(0o777)
        os.system(f"bsub < {batch_path.resolve()}")
        print(f"Submitted job {job_name!r}.")
