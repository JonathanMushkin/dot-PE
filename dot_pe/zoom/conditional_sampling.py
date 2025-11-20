"""
Conditional sampling from PowerLawIntrinsicIASPrior.

This module provides functionality to sample from PowerLawIntrinsicIASPrior
conditioned on fixed values of f_ref, mchirp, lnq, chieff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Optional, Dict

from dot_pe.power_law_mass_prior import (
    PowerLawIntrinsicIASPrior,
    PowerLawChirpMassPrior,
)


class ConditionalPriorSampler:
    """
    Sample from PowerLawIntrinsicIASPrior conditioned on fixed parameters.

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
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mchirp_range: tuple[float, float],
        q_min: float,
        f_ref: float,
        seed: Optional[int] = None,
    ):
        self.mchirp_range = mchirp_range
        self.q_min = q_min
        self.f_ref = f_ref
        self.seed = seed

        self.prior = PowerLawIntrinsicIASPrior(
            mchirp_range=mchirp_range,
            q_min=q_min,
            f_ref=f_ref,
        )

        self._identify_parameter_mapping()

    def _identify_parameter_mapping(self):
        """Identify which subpriors correspond to which standard_params."""
        self.mass_prior = None
        self.spin_prior = None
        self.inplane_spin_prior = None

        for subprior in self.prior.subpriors:
            if hasattr(subprior, "standard_params"):
                if (
                    "m1" in subprior.standard_params
                    and "m2" in subprior.standard_params
                ):
                    self.mass_prior = subprior
                elif (
                    "s1z" in subprior.standard_params
                    and "s2z" in subprior.standard_params
                ):
                    if "chieff" in getattr(subprior, "range_dic", {}):
                        self.spin_prior = subprior
                elif "iota" in subprior.standard_params:
                    self.inplane_spin_prior = subprior

        if (
            self.mass_prior is None
            or self.spin_prior is None
            or self.inplane_spin_prior is None
        ):
            raise RuntimeError("Could not identify all required prior components.")

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
            Dictionary with fixed sampled_params values (mchirp_p, lnq, chieff).
        """
        mchirp_p = mchirp**PowerLawChirpMassPrior.TRANSFORM_POWER

        return {
            "mchirp_p": mchirp_p,
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
                    param_range = self.spin_prior.range_dic[param_name]
                else:
                    param_range = self.inplane_spin_prior.range_dic[param_name]
                sampled_params[param_name] = param_range[0] + u[j, i] * (
                    param_range[1] - param_range[0]
                )

            standard_params = self.prior.transform(
                **sampled_params,
                f_ref=self.f_ref,
            )

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
