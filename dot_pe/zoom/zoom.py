"""Utilities for fitting and sampling multivariate Gaussian distributions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg, stats


Bounds = Dict[int, Tuple[Optional[float], Optional[float]]]


@dataclass
class Zoomer:
    """Fit an n-dimensional Gaussian and draw constrained QMC samples.

    Example
    -------
    >>> zoomer = Zoomer()
    >>> zoomer.fit(np.random.normal(size=(1_000, 3)), n_sig=2.0)
    >>> draws, pdf = zoomer.sample(32, bounds={0: (-1.0, 1.0)})
    """

    jitter: float = 1e-12
    engine_seed: Optional[int] = None
    mean: Optional[np.ndarray] = field(init=False, default=None)
    cov: Optional[np.ndarray] = field(init=False, default=None)
    bounds: Optional[Bounds] = field(init=False, default=None)
    distribution: Optional[stats._multivariate.multivariate_normal_frozen] = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.engine_seed)

    def fit(
        self,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_sig: Optional[float] = None,
    ) -> None:
        """Fit a Gaussian to the provided samples.

        Parameters
        ----------
        samples:
            Array of shape (n_samples, n_dims).
        weights:
            Optional array of probabilistic weights of shape (n_samples,).
            If provided, must be non-negative and same length as samples.
        n_sig:
            If provided, scales the covariance so all samples lie within ``n_sig``
            Mahalanobis distance from the mean. Values <= 0 are invalid.
        """
        array = np.asarray(samples, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Samples must be a 2D array of shape (n_samples, n_dims).")
        if array.shape[0] < 2:
            raise ValueError(
                "At least two samples are required to estimate covariance."
            )

        n_samples = array.shape[0]

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1:
                raise ValueError("Weights must be a 1D array.")
            if weights.shape[0] != n_samples:
                raise ValueError(
                    f"Weights length {weights.shape[0]} must match samples length {n_samples}."
                )
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative.")
            if np.sum(weights) == 0:
                raise ValueError("At least one weight must be positive.")
            weights = weights / np.sum(weights)
            mean = np.average(array, axis=0, weights=weights)
            cov = np.cov(array, rowvar=False, aweights=weights, bias=False)
        else:
            mean = array.mean(axis=0)
            cov = np.cov(array, rowvar=False, bias=False)

        cov = np.atleast_2d(cov)
        dim = mean.shape[0]

        if cov.shape != (dim, dim):
            raise ValueError("Covariance matrix shape mismatch with mean dimension.")

        cov = cov + np.eye(dim) * self.jitter

        if n_sig is not None:
            if n_sig <= 0:
                raise ValueError("n_sig must be positive when provided.")
            max_dist = self._max_mahalanobis(array, mean, cov)
            if max_dist > n_sig:
                scale = (max_dist / n_sig) ** 2
                cov = cov * scale

        self.mean = mean
        self.cov = cov
        self.distribution = stats.multivariate_normal(
            mean=self.mean, cov=self.cov, allow_singular=True
        )

    def sample(
        self,
        count: int,
        bounds: Optional[Bounds] = None,
        max_iter: Optional[int] = None,
        batch: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw samples and return their pdf values."""
        if count <= 0:
            raise ValueError("count must be positive.")
        if self.distribution is None or self.mean is None or self.cov is None:
            raise RuntimeError("Call fit() before sampling.")

        self.bounds = bounds
        dim = self.mean.shape[0]
        normalized_bounds = self._normalize_bounds(bounds, dim)
        qmc_sampler = self._build_qmc_sampler(dim)
        batch_size = batch or max(count, dim * 2)

        samples = []
        pdfs = []
        iterations = 0
        accepted_total = 0
        while accepted_total < count:
            iterations += 1
            if max_iter is not None and iterations > max_iter:
                raise RuntimeError(
                    "Unable to gather enough samples within the provided max_iter."
                )

            candidates = self._draw_candidates(batch_size, qmc_sampler)
            mask = self._apply_bounds(candidates, normalized_bounds)
            if not mask.any():
                continue

            accepted = candidates[mask]
            accepted_pdf = self.distribution.pdf(accepted)

            samples.append(accepted)
            pdfs.append(accepted_pdf)
            accepted_total += accepted.shape[0]

        stacked_samples = np.vstack(samples)[:count]
        stacked_pdfs = np.concatenate(pdfs)[:count]

        if normalized_bounds is not None:
            Z = self._compute_bounded_cdf(normalized_bounds)
            if Z > 0:
                stacked_pdfs = stacked_pdfs / Z

        return stacked_samples, stacked_pdfs

    @staticmethod
    def _max_mahalanobis(
        samples: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> float:
        """Return the maximum Mahalanobis distance for the provided samples."""
        centered = samples - mean
        if centered.size == 0:
            return 0.0
        chol = linalg.cholesky(cov, lower=True, check_finite=False)
        solved = linalg.solve_triangular(
            chol, centered.T, lower=True, check_finite=False
        )
        distances_sq = np.sum(solved**2, axis=0)
        return float(np.sqrt(distances_sq.max()))

    def _compute_bounded_cdf(
        self, bounds: Tuple[Tuple[Optional[float], Optional[float]], ...]
    ) -> float:
        """Compute the CDF over the bounded region."""
        a = np.array([lower if lower is not None else -np.inf for lower, _ in bounds])
        b = np.array([upper if upper is not None else np.inf for _, upper in bounds])
        return float(
            stats.multivariate_normal.cdf(
                x=b, mean=self.mean, cov=self.cov, lower_limit=a
            )
        )

    def _build_qmc_sampler(self, dim: int) -> Optional[stats.qmc.MultivariateNormalQMC]:
        try:
            engine = stats.qmc.Halton(d=dim, scramble=True, seed=self.engine_seed)
            return stats.qmc.MultivariateNormalQMC(
                mean=self.mean, cov=self.cov, engine=engine
            )
        except Exception:
            return None

    def _draw_candidates(
        self, size: int, qmc_sampler: Optional[stats.qmc.MultivariateNormalQMC]
    ) -> np.ndarray:
        if qmc_sampler is not None:
            return qmc_sampler.random(size)

        return self._rng.multivariate_normal(
            mean=self.mean, cov=self.cov, size=size, check_valid="ignore", method="svd"
        )

    @staticmethod
    def _normalize_bounds(
        bounds: Optional[Bounds], dim: int
    ) -> Optional[Tuple[Tuple[Optional[float], Optional[float]], ...]]:
        if bounds is None:
            return None
        normalized = [(None, None)] * dim
        for idx, (lower, upper) in bounds.items():
            if idx < 0 or idx >= dim:
                raise ValueError(f"Bound index {idx} is outside dimension {dim}.")
            normalized[idx] = (lower, upper)
        return tuple(normalized)

    @staticmethod
    def _apply_bounds(
        samples: np.ndarray,
        bounds: Optional[Tuple[Tuple[Optional[float], Optional[float]], ...]],
    ) -> np.ndarray:
        if bounds is None:
            return np.ones(samples.shape[0], dtype=bool)
        mask = np.ones(samples.shape[0], dtype=bool)
        for dim, (lower, upper) in enumerate(bounds):
            if lower is not None:
                mask &= samples[:, dim] >= lower
            if upper is not None:
                mask &= samples[:, dim] <= upper
        return mask

    def to_json(self, path: Union[str, Path]) -> None:
        """Save Zoomer state to JSON file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file to save.
        """
        path = Path(path)
        data = {
            "jitter": self.jitter,
            "engine_seed": self.engine_seed,
        }

        if self.mean is not None:
            data["mean"] = self.mean.tolist()
        if self.cov is not None:
            data["cov"] = self.cov.tolist()
        if self.bounds is not None:
            data["bounds"] = {str(k): list(v) for k, v in self.bounds.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Zoomer":
        """Load Zoomer state from JSON file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file to load.

        Returns
        -------
        Zoomer
            Reconstructed Zoomer instance.
        """
        from cogwheel.utils import read_json

        path = Path(path)
        data = read_json(path)

        zoomer = cls(
            jitter=data.get("jitter", 1e-12),
            engine_seed=data.get("engine_seed"),
        )

        if "mean" in data:
            zoomer.mean = np.array(data["mean"])
        if "cov" in data:
            zoomer.cov = np.array(data["cov"])
        if zoomer.mean is not None and zoomer.cov is not None:
            zoomer.distribution = stats.multivariate_normal(
                mean=zoomer.mean, cov=zoomer.cov, allow_singular=True
            )
        if "bounds" in data:
            zoomer.bounds = {int(k): tuple(v) for k, v in data["bounds"].items()}

        return zoomer


# ============================================================================
# Working Examples
# ============================================================================
#
# Example 1: Basic usage with and without bounds
# -----------------------------------------------
# import numpy as np
# from dot_pe.zoom import Zoomer
#
# # Generate sample data (3D)
# np.random.seed(42)
# samples = np.random.multivariate_normal(
#     mean=[0, 1, 2],
#     cov=np.eye(3),
#     size=1000
# )
#
# # Create zoomer and fit
# zoomer = Zoomer()
# zoomer.fit(samples)
#
# # Fit with probabilistic weights
# weights = np.random.exponential(scale=1.0, size=1000)
# weights = weights / weights.sum()  # Normalize (optional, fit() does this)
# zoomer.fit(samples, weights=weights)
#
# # Draw samples without bounds
# draws, pdfs = zoomer.sample(100)
# print(f"Drew {len(draws)} samples, shape: {draws.shape}")
#
# # Draw samples with bounds on single dimension
# bounds = {0: (-1.0, 1.0)}
# draws, pdfs = zoomer.sample(100, bounds=bounds)
# print(f"All samples in dim 0 within [-1, 1]: {np.all((draws[:, 0] >= -1.0) & (draws[:, 0] <= 1.0))}")
# # PDFs are normalized by the bounded prior mass Z
#
# # Draw samples with bounds on multiple dimensions
# bounds = {
#     0: (-0.5, 10.0),
#     1: (-3.0, 5.0),
#     2: (0.5, 6.0)
# }
# draws, pdfs = zoomer.sample(100, bounds=bounds)
# print(f"Drew {len(draws)} samples within bounds")
#
# # Draw samples with partial bounds (only lower or upper limit)
# bounds = {0: (-1.0, None)}  # Only lower bound
# draws, pdfs = zoomer.sample(100, bounds=bounds)
# print(f"All samples in dim 0 >= -1.0: {np.all(draws[:, 0] >= -1.0)}")
#
#
# Example 2: Advanced features (n_sig, batch size, max_iter, seeding)
# ---------------------------------------------------------------------
# import numpy as np
# from dot_pe.zoom import Zoomer
#
# # Generate sample data
# np.random.seed(42)
# samples = np.random.multivariate_normal(
#     mean=[0, 1, 2],
#     cov=np.eye(3),
#     size=1000
# )
#
# # Fit with n_sig parameter: scales covariance so all samples lie within n_sig sigma
# zoomer = Zoomer(engine_seed=12345)  # Seed for reproducibility
# zoomer.fit(samples, n_sig=2.0)
#
# # Draw samples with bounds and custom batch size/max_iter
# # batch: number of candidates to draw per iteration
# # max_iter: maximum number of iterations before raising error
# bounds = {0: (-0.1, 0.1), 1: (-2.0, 4.0)}  # Tight bounds on dim 0
# draws, pdfs = zoomer.sample(100, bounds=bounds, batch=500, max_iter=1000)
# print(f"Drew {len(draws)} samples")
#
# # Verify reproducibility with seed
# draws1, pdfs1 = zoomer.sample(100)
# zoomer2 = Zoomer(engine_seed=12345)
# zoomer2.fit(samples, n_sig=2.0)
# draws2, pdfs2 = zoomer2.sample(100)
# print(f"Samples are reproducible: {np.allclose(draws1, draws2)}")
