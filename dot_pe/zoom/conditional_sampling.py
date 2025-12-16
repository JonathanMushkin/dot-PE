"""
Conditional sampling from IntrinsicIASPrior.

This module provides functionality to sample from IntrinsicIASPrior
conditioned on fixed values of f_ref, mchirp, lnq, chieff.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import qmc

from cogwheel.gw_prior import IntrinsicIASPrior
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior


class ConditionalPriorSampler:
    """
    Sample from IntrinsicIASPrior conditioned on fixed parameters.

    This class allows sampling the remaining parameters (cumchidiff, in-plane spins and
    orientation) when f_ref, mchirp, lnq, chieff are fixed.

    Parameters
    ----------
    mchirp_range : tuple[float, float]
        Range of chirp masses (M_sun).
    q_min : float
        Minimum mass ratio.
    f_ref : float
        Reference frequency (Hz).
    aligned_spin : bool, optional
        If True, only sample costheta_jn (iota) and zero out in-plane spins.
        Default is False (precessing mode).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mchirp_range: tuple[float, float],
        q_min: float,
        f_ref: float,
        aligned_spin: bool = False,
        seed: Optional[int] = None,
    ):
        self.mchirp_range = mchirp_range
        self.q_min = q_min
        self.f_ref = f_ref
        self.aligned_spin = aligned_spin
        self.seed = seed

        self.prior = IntrinsicIASPrior(
            mchirp_range=mchirp_range,
            q_min=q_min,
            f_ref=f_ref,
        )

        self._identify_parameter_mapping()

    def _identify_parameter_mapping(self):
        """Identify which subpriors correspond to which standard_params."""
        self.mass_prior = None
        self.aligned_spin_prior = None
        self.inplane_spin_prior = None

        for subprior in self.prior.subpriors:
            if isinstance(subprior, UniformDetectorFrameMassesPrior):
                self.mass_prior = subprior
            elif hasattr(subprior, "standard_params"):
                if (
                    "s1z" in subprior.standard_params
                    and "s2z" in subprior.standard_params
                ):
                    if "chieff" in getattr(subprior, "range_dic", {}):
                        self.aligned_spin_prior = subprior
                elif "iota" in subprior.standard_params:
                    self.inplane_spin_prior = subprior

        if self.mass_prior is None or self.aligned_spin_prior is None:
            raise RuntimeError("Could not identify mass_prior and aligned_spin_prior.")
        if self.inplane_spin_prior is None:
            raise RuntimeError("Could not identify inplane_spin_prior.")

    def _get_fixed_sampled_params(
        self, mchirp: float, lnq: float, chieff: float
    ) -> Dict[str, float]:
        """
        Convert fixed well-measured parameters to sampled_params.

        Parameters
        ----------
        mchirp : float
            Chirp mass (M_sun).
        lnq : float
            Log mass ratio.
        chieff : float
            Effective aligned spin.

        Returns
        -------
        dict
            Dictionary with fixed sampled_params values (mchirp, lnq, chieff).
        """
        return {
            "mchirp": mchirp,
            "lnq": lnq,
            "chieff": chieff,
        }

    def sample(
        self,
        n_samples: int,
        mchirp: float,
        lnq: float,
        chieff: float,
        method: str = "qmc",
    ) -> pd.DataFrame:
        """
        Sample remaining parameters conditioned on fixed well-measured values.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        mchirp : float
            Fixed chirp mass (M_sun).
        lnq : float
            Fixed log mass ratio.
        chieff : float
            Fixed effective aligned spin.
        method : str, optional
            Sampling method: "qmc" (quasi-Monte Carlo) or "mc" (Monte Carlo).
            Default is "qmc".

        Returns
        -------
        pd.DataFrame
            DataFrame with all standard parameters. Fixed parameters (mchirp, lnq,
            chieff, f_ref) are constant across all samples. Sampled parameters
            (cumchidiff, iota, s1x_n, s1y_n, s2x_n, s2y_n) vary.
        """
        fixed_sampled = self._get_fixed_sampled_params(mchirp, lnq, chieff)

        if self.aligned_spin:
            remaining_sampled_params = ["costheta_jn", "cumchidiff"]
        else:
            remaining_sampled_params = [
                "cumchidiff",
                "costheta_jn",
                "phi_jl_hat",
                "phi12",
                "cums1r_s1z",
                "cums2r_s2z",
            ]

        if method == "qmc":
            sampler = qmc.Halton(
                d=len(remaining_sampled_params),
                scramble=True,
                seed=self.seed,
            )
            u = sampler.random(n_samples).T
        else:
            rng = np.random.default_rng(self.seed)
            u = rng.random((len(remaining_sampled_params), n_samples))

        samples_list = []
        for i in range(n_samples):
            sampled_params = fixed_sampled.copy()

            for j, param_name in enumerate(remaining_sampled_params):
                if param_name == "cumchidiff":
                    param_range = self.aligned_spin_prior.range_dic[param_name]
                else:
                    param_range = self.inplane_spin_prior.range_dic[param_name]
                sampled_params[param_name] = param_range[0] + u[j, i] * (
                    param_range[1] - param_range[0]
                )

            if self.aligned_spin:
                sampled_params["phi_jl_hat"] = 0.0
                sampled_params["phi12"] = 0.0
                sampled_params["cums1r_s1z"] = 0.0
                sampled_params["cums2r_s2z"] = 0.0

            standard_params = self.prior.transform(
                **sampled_params,
                f_ref=self.f_ref,
            )

            if self.aligned_spin:
                standard_params["s1x_n"] = 0.0
                standard_params["s1y_n"] = 0.0
                standard_params["s2x_n"] = 0.0
                standard_params["s2y_n"] = 0.0

            samples_list.append(standard_params)

        df = pd.DataFrame(samples_list)
        return df

    def sample_array(
        self,
        n_samples: int,
        mchirp: float,
        lnq: float,
        chieff: float,
        method: str = "qmc",
    ) -> Dict[str, np.ndarray]:
        """
        Sample remaining parameters and return as arrays.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        mchirp : float
            Fixed chirp mass (M_sun).
        lnq : float
            Fixed log mass ratio.
        chieff : float
            Fixed effective aligned spin.
        method : str, optional
            Sampling method: "qmc" or "mc". Default is "qmc".

        Returns
        -------
        dict
            Dictionary with parameter names as keys and numpy arrays as values.
        """
        df = self.sample(n_samples, mchirp, lnq, chieff, method)
        return {col: df[col].values for col in df.columns}

    def sample_vectorized(
        self,
        mchirp: np.ndarray,
        lnq: np.ndarray,
        chieff: np.ndarray,
        method: str = "mc",
    ) -> pd.DataFrame:
        """
        Sample one sample for each set of fixed parameters (vectorized).

        Parameters
        ----------
        mchirp : np.ndarray
            Array of fixed chirp masses (M_sun).
        lnq : np.ndarray
            Array of fixed log mass ratios.
        chieff : np.ndarray
            Array of fixed effective aligned spins.
        method : str, optional
            Sampling method: "qmc" or "mc". Default is "mc".

        Returns
        -------
        pd.DataFrame
            DataFrame with all standard parameters and sampled parameters, one row per input parameter set.
            Standard parameters: m1, m2, s1z, s2z, iota, s1x_n, s1y_n, s2x_n, s2y_n, etc.
            Sampled parameters: mchirp, lnq, chieff, cumchidiff, costheta_jn, phi_jl_hat, phi12, cums1r_s1z, cums2r_s2z.
        """
        n_samples = len(mchirp)
        if len(lnq) != n_samples or len(chieff) != n_samples:
            raise ValueError("All input arrays must have the same length")

        if self.aligned_spin:
            remaining_sampled_params = ["costheta_jn", "cumchidiff"]
        else:
            remaining_sampled_params = [
                "cumchidiff",
                "costheta_jn",
                "phi_jl_hat",
                "phi12",
                "cums1r_s1z",
                "cums2r_s2z",
            ]

        if method == "qmc":
            sampler = qmc.Halton(
                d=len(remaining_sampled_params),
                scramble=True,
                seed=self.seed,
            )
            u = sampler.random(n_samples).T
        else:
            rng = np.random.default_rng(self.seed)
            u = rng.random((len(remaining_sampled_params), n_samples))

        samples_list = []
        for i in range(n_samples):
            fixed_sampled = self._get_fixed_sampled_params(mchirp[i], lnq[i], chieff[i])

            sampled_params = fixed_sampled.copy()
            for j, param_name in enumerate(remaining_sampled_params):
                if param_name == "cumchidiff":
                    param_range = self.aligned_spin_prior.range_dic[param_name]
                else:
                    param_range = self.inplane_spin_prior.range_dic[param_name]
                sampled_params[param_name] = param_range[0] + u[j, i] * (
                    param_range[1] - param_range[0]
                )

            if self.aligned_spin:
                sampled_params["phi_jl_hat"] = 0.0
                sampled_params["phi12"] = 0.0
                sampled_params["cums1r_s1z"] = 0.0
                sampled_params["cums2r_s2z"] = 0.0

            standard_params = self.prior.transform(
                **sampled_params,
                f_ref=self.f_ref,
            )

            if self.aligned_spin:
                standard_params["s1x_n"] = 0.0
                standard_params["s1y_n"] = 0.0
                standard_params["s2x_n"] = 0.0
                standard_params["s2y_n"] = 0.0

            combined_params = {**standard_params, **sampled_params}
            samples_list.append(combined_params)

        return pd.DataFrame(samples_list)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save ConditionalPriorSampler state to JSON file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file to save.
        """
        path = Path(path)
        data = {
            "mchirp_range": list(self.mchirp_range),
            "q_min": self.q_min,
            "f_ref": self.f_ref,
            "aligned_spin": self.aligned_spin,
            "seed": self.seed,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ConditionalPriorSampler":
        """Load ConditionalPriorSampler state from JSON file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file to load.

        Returns
        -------
        ConditionalPriorSampler
            Reconstructed ConditionalPriorSampler instance.
        """
        from cogwheel.utils import read_json

        path = Path(path)
        data = read_json(path)

        return cls(
            mchirp_range=tuple(data["mchirp_range"]),
            q_min=data["q_min"],
            f_ref=data["f_ref"],
            aligned_spin=data.get("aligned_spin", False),
            seed=data.get("seed"),
        )


# ============================================================================
# Working Examples
# ============================================================================
#
# Example 1: Basic conditional sampling
# --------------------------------------
# from conditioned_sampling import ConditionalPriorSampler
#
# # Create sampler with prior configuration
# sampler = ConditionalPriorSampler(
#     mchirp_range=(10, 20),
#     q_min=0.3,
#     f_ref=50,
#     seed=42,
# )
#
# # Sample remaining parameters conditioned on fixed values
# # Fixed: m1=15, m2=12, s1z=0.5, s2z=-0.3
# samples = sampler.sample(
#     n_samples=1000,
#     m1=15.0,
#     m2=12.0,
#     s1z=0.5,
#     s2z=-0.3,
#     method="qmc",
# )
#
# # Check that fixed parameters are constant
# print(f"m1 constant: {samples['m1'].nunique() == 1}")
# print(f"m2 constant: {samples['m2'].nunique() == 1}")
# print(f"s1z constant: {samples['s1z'].nunique() == 1}")
# print(f"s2z constant: {samples['s2z'].nunique() == 1}")
# print(f"f_ref constant: {samples['f_ref'].nunique() == 1}")
#
# # Check that sampled parameters vary
# print(f"iota varies: {samples['iota'].nunique() > 1}")
# print(f"s1x_n varies: {samples['s1x_n'].nunique() > 1}")
#
#
# Example 2: Using sample_array for array output
# -----------------------------------------------
# from conditioned_sampling import ConditionalPriorSampler
# import numpy as np
#
# sampler = ConditionalPriorSampler(
#     mchirp_range=(10, 20),
#     q_min=0.3,
#     f_ref=50,
# )
#
# # Get samples as arrays
# samples_dict = sampler.sample_array(
#     n_samples=100,
#     m1=15.0,
#     m2=12.0,
#     s1z=0.5,
#     s2z=-0.3,
# )
#
# # Access individual parameter arrays
# iota_samples = samples_dict["iota"]
# s1x_n_samples = samples_dict["s1x_n"]
# print(f"iota range: [{iota_samples.min():.3f}, {iota_samples.max():.3f}]")
#
#
# Example 3: Monte Carlo sampling
# --------------------------------
# from conditioned_sampling import ConditionalPriorSampler
#
# sampler = ConditionalPriorSampler(
#     mchirp_range=(10, 20),
#     q_min=0.3,
#     f_ref=50,
#     seed=123,
# )
#
# # Use Monte Carlo instead of QMC
# samples = sampler.sample(
#     n_samples=1000,
#     m1=15.0,
#     m2=12.0,
#     s1z=0.5,
#     s2z=-0.3,
#     method="mc",
# )
#
# print(f"Generated {len(samples)} samples")
