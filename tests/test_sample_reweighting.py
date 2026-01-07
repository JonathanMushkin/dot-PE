import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent).resolve()))
from cogwheel.gw_prior import IntrinsicIASPrior
from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior
from cogwheel.utils import exp_normalize
from cogwheel.gw_utils import m1m2_to_mchirp
from cogwheel.prior_ratio import PriorRatio
import matplotlib.pyplot as plt


N = 10**5
mass_range = (15, 30)
q_min = 0.2
f_ref = 0.5

bins = 30
alpha = 0.5

ias_prior = IntrinsicIASPrior(mchirp_range=mass_range, q_min=q_min, f_ref=f_ref)
powerlaw_prior = PowerLawIntrinsicIASPrior(
    mchirp_range=mass_range, q_min=q_min, f_ref=f_ref
)

powerlaw_samples = powerlaw_prior.generate_random_samples(N)
ias_samples = ias_prior.generate_random_samples(N)
powerlaw_samples["mchirp"] = m1m2_to_mchirp(**powerlaw_samples[["m1", "m2"]])

# Evaluate prior ratio directly

powerlaw_lnprior = powerlaw_samples.apply(
    lambda row: powerlaw_prior.lnprior(**row.to_dict()), axis=1
)
ias_lnprior = powerlaw_samples.apply(
    lambda row: ias_prior.lnprior(**row.to_dict()), axis=1
)
powerlaw_ln_jac_det = powerlaw_prior.subpriors[1].ln_jacobian_determinant(
    **powerlaw_samples[["m1", "m2"]]
)
ias_ln_jac_det = ias_prior.subpriors[1].ln_jacobian_determinant(
    **powerlaw_samples[["m1", "m2"]]
)
log_weights = ias_lnprior + ias_ln_jac_det - powerlaw_lnprior - powerlaw_ln_jac_det

weights = exp_normalize((log_weights).values)

hist_kwargs = dict(bins=bins, alpha=alpha, density=True, histtype="step")

counts, bins, patches = plt.hist(
    powerlaw_samples["mchirp"], label="powerlaw", **hist_kwargs
)
plt.plot(
    bins,
    counts[0] * (bins / bins[0]) ** (-1.7),
    label=r"$P(\mathcal{M})\propto\mathcal{M}_c^{-1.7}$",
)
_ = plt.hist(ias_samples["mchirp"], label="ias", **hist_kwargs)

_ = plt.hist(
    powerlaw_samples["mchirp"],
    weights=weights,
    label="powerlaw samples, reweighted",
    **hist_kwargs,
)

# Use PriorRatio method


prior_ratio = PriorRatio(ias_prior, powerlaw_prior)
ln_prior_ratio = powerlaw_samples.apply(
    lambda row: prior_ratio.ln_prior_ratio(**row.to_dict()), axis=1
)

prior_ratios_normalized = exp_normalize(ln_prior_ratio.values)

_ = plt.hist(
    powerlaw_samples["mchirp"],
    weights=prior_ratios_normalized,
    label="powerlaw samples, reweighted (with PriorRatio)",
    **hist_kwargs,
)

plt.xlabel("chirp mass")
plt.ylabel("pdf")


plt.legend()

fig_savepath = (
    Path(__file__).parent
    / "artifacts"
    / "sample_reweighting"
    / "chirp_mass_histogram.png"
)
fig_savepath.parent.mkdir(parents=True, exist_ok=True)

fig = plt.gcf()
fig.savefig(fig_savepath)
