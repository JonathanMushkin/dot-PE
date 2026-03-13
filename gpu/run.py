"""
GPU-accelerated entry point for inference.run() / inference.run_and_profile().

Monkey-patches both GPU classes into dot_pe before delegating, so that all
downstream code picks up the GPU implementations transparently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _patch():
    """Replace CPU classes with GPU subclasses in the dot_pe namespace."""
    from gpu.single_detector_gpu import GPUSingleDetectorProcessor
    from gpu.likelihood_calculating_gpu import GPULikelihoodCalculator

    import dot_pe.single_detector as _sd
    import dot_pe.coherent_processing as _cp
    import dot_pe.likelihood_calculating as _lc

    _sd.SingleDetectorProcessor = GPUSingleDetectorProcessor
    _lc.LikelihoodCalculator = GPULikelihoodCalculator
    # coherent_processing imports LikelihoodCalculator at class-body level
    # so we also patch it there
    _cp.likelihood_calculating.LikelihoodCalculator = GPULikelihoodCalculator


def run(**kwargs):
    """GPU-accelerated inference.run()."""
    _patch()
    from dot_pe import inference
    return inference.run(**kwargs)


def run_and_profile(**kwargs):
    """GPU-accelerated inference.run_and_profile()."""
    _patch()
    from dot_pe import inference
    return inference.run_and_profile(**kwargs)
