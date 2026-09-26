---
status: accepted
---

# A per-asset moment target centres a net series on its net mean

## Context

Issue #1320, found while fixing #1316. A moment measure reads the net return series
``\mathbf{X}\boldsymbol{w} - F(\boldsymbol{w})`` and centres it on a target. `factory` fills an
empty target slot with the prior's expected returns, so the target is the per-asset form
``\boldsymbol{w}^\intercal\boldsymbol{\mu}``, which is a **gross** expected return. Every deviation
from it carried the mean fee, and the two levels read that shift in different ways.

- **Mean absolute deviation.** The model states the measure as
  ``2\,\mathrm{mean}(\max(0, \mathrm{tgt} - \mathrm{net}))``, which is ``\mathrm{mean}|d|`` only
  when ``\mathrm{mean}(d) = 0``. With a fee ``\mathrm{mean}(d) = -\bar{F}``, so the model reported
  the functor plus the fee: `0.00659207` against `0.00459207` at `Fees(; l = 0.002)`.
- **Kurtosis.** The model reads the cokurtosis matrix of the prior, so no fee enters it. The
  functor centred the net series on the gross target, and read `6.09761e-5` against the model's
  `5.79171e-5`.

So `expected_risk` did not report the number that the optimiser minimised or bounded. The first
lower moment and the second moments agreed, because their model and their functor read the same
shifted deviations. Two fixes were open: keep the gross target and change the two models, or make
the target net at both levels. The second changes the value of the first lower moment and of the
semi moments under a fee, so it was the maintainer's call.

## Decision

1. **A per-asset target is the net mean.** A target that reads the weights, a `VecNum` or the
   vector part of a `VecScalar`, whether stated or filled from the prior, subtracts the mean fee
   per period, ``\boldsymbol{w}^\intercal\boldsymbol{\mu} - \bar{F}(\boldsymbol{w})``. The mean
   fee is the charge the net expected return carries: the per period terms plus the one-off terms
   divided by the observation count, on either clock of `fees.fa`.
2. **A target stated on the net series takes no fee.** The sample mean or median of the net
   series, a `MeanCentering` or `MedianCentering`, and a scalar threshold are already net.
   `weight_independent_target` draws the line, so the rule is one predicate at both levels.
3. **Both levels subtract the same number.** The value level subtracts `moment_target_fees`,
   which reads `term_fees`, in the `calc_deviations_vec` of every moment family and in
   `difference_risk`. The model level subtracts the fee expressions through `add_fees_to_ret!` in
   `calc_risk_constraint_target`. A tracked measure's target reads the weight difference and the
   fee of the portfolio, as its series does.

## Consequences

- A per period fee cancels from every central moment, so the model and the functor of every
  moment measure agree under a fee. With a prior whose `mu` is the sample mean, a filled target
  gives the same value as the sample mean of the net series, on either clock of the one-off terms.
- The value of a moment measure with a per-asset target under a fee changes. Without a fee
  nothing changes. The first lower moment and the semi moments under a fee now read deviations
  below the net mean, not below the gross mean.
- The Kurtosis model still reads no fee. A one-off fee charged on the first observation is not a
  per period shift, so it leaves a small gap between that model and its functor. A per period fee
  and an amortised one-off fee leave none.
