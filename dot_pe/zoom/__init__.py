"""Zoom module for fitting and sampling multivariate Gaussian distributions."""

from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler
from dot_pe.zoom.zoom import Bounds, Zoomer
from dot_pe.zoom.zoom_iteration import (
    draw_from_zoomer,
    fit_zoomer,
    hellinger_distance,
    main,
    parse_args,
)

__all__ = [
    "Bounds",
    "ConditionalPriorSampler",
    "Zoomer",
    "draw_from_zoomer",
    "fit_zoomer",
    "hellinger_distance",
    "main",
    "parse_args",
]
