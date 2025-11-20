"""
Compatibility wrapper for the conditional sampler implementation.

The canonical implementation now lives in ``dot_pe.zoom.conditional_sampling``.
"""

from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler

__all__ = ["ConditionalPriorSampler"]
