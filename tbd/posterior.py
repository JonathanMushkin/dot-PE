"""
Versions of Posterior and reference waveform finders that are better suited 
for the sampler-free methods. 

TODO: Check if code actually needed.
"""

from cogwheel import posterior
from cogwheel.likelihood import (ReferenceWaveformFinder, 
                                 RelativeBinningLikelihood)
from cogwheel import gw_utils
from cogwheel import gw_prior


class SamplerFreePosterior(posterior.Posterior):
    """
    Posterior class that also keep the reference waveform finder.

    """

    def __init__(self, prior, likelihood, ref_wf_finder=None):
        super().__init__(prior, likelihood)
        self.ref_wf_finder = ref_wf_finder

    @classmethod
    def from_event(
            cls, event, mchirp_guess, approximant, prior_class,
            likelihood_class=None, prior_kwargs=None,
            likelihood_kwargs=None, ref_wf_finder_kwargs=None):
        """
        Instantiate a `Posterior` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `data.EventData` or string with event name,
               or path to npz file with `EventData` instance.

        mchirp_guess: float
            Approximate chirp mass (Msun).

        approximant: str
            Approximant name.

        prior_class: string with key from `gw_prior.prior_registry`,
                     or subclass of `prior.Prior`.

        likelihood_class:
            subclass of likelihood.RelativeBinningLikelihood

        prior_kwargs: dict,
            Keyword arguments for `prior_class` constructor.

        likelihood_kwargs: dict
            Keyword arguments for `likelihood_class` constructor.

        Return
        ------
            Instance of `Posterior`.
        """
        prior_kwargs = prior_kwargs or {}
        likelihood_kwargs = likelihood_kwargs or {}
        ref_wf_finder_kwargs = ref_wf_finder_kwargs or {}

        if isinstance(prior_class, str):
            try:
                prior_class = gw_prior.prior_registry[prior_class]
            except KeyError as err:
                raise KeyError('Avaliable priors are: '
                               f'{", ".join(gw_prior.prior_registry)}.'
                              ) from err

        if likelihood_class is None:
            likelihood_class = getattr(prior_class,
                                       'default_likelihood_class',
                                       RelativeBinningLikelihood)

        ref_wf_finder = ReferenceWaveformFinder.from_event(
            event, mchirp_guess, approximant=approximant,
            **ref_wf_finder_kwargs)

        likelihood = likelihood_class.from_reference_waveform_finder(
            ref_wf_finder, approximant=approximant, **likelihood_kwargs)

        prior = prior_class.from_reference_waveform_finder(ref_wf_finder,
                                                           **prior_kwargs)
        return cls(prior, likelihood, ref_wf_finder)
    