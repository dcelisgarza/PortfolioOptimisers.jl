---
status: accepted
---

# A weighted regression reads the Kish count of its weights

## Context

The fix of #1637 found
[#1640](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1640): a `LinearModel` or a
`GeneralisedLinearModel` fitted under probability weights that sum to one reads one observation.

GLM 1.9.5 reads every weight vector as frequency weights. It takes `nobs` as `sum(w)`, the residual
degree of freedom as `nobs - p`, and, for `LinearModel` and the Normal family, the dispersion as
`Σ w r² / (nobs - p)`. A family with a unit dispersion, such as Bernoulli or Poisson, takes
`vcov = inv(X'WX)`, so the scale of the weights sets its count through `X'WX` directly. The weights that the
library gives a regression through `factory` sum to one: the posterior probabilities of
`EntropyPoolingPrior` and `MeucciEntropyPoolingPrior`, and the pooled weights of
`OpinionPoolingPrior`. So every such fit had `dof_residual = 1 - p`, every p-value was `1.0`, and
the AIC and the BIC read one observation. A `StepwiseRegression` inside any of these priors found no
factor under its threshold, and fitted every asset on factor 1 alone. ADR 0116 saw the same effect
as a moved `:bic` set.

The scale of the weights moves no coefficient, because a weighted fit is invariant to a constant
scale of its weights. It moves only the count of observations that the inference reads. Two counts
were candidates:

- **The row count `T`.** Scale the weights to sum to `T`. Then
  `vcov = inv(X'WX) · Σ w r² / (T - p)`, which is exact when the weights are inverse variances.
- **The Kish count `T_e = (Σ w)² / Σ w²`.** Scale the weights to sum to `T_e`. Then
  `vcov = inv(X'WX) · Σ w r² / (T_e - p)`. For iid rows reweighted by fixed weights, this gives the
  exact variance `σ² / T_e` of a weighted mean, and the dispersion is unbiased. Under equal weights
  on `k` rows and none on the rest it is the fit of those `k` rows.

The weights of the library are scenario probabilities and decay weights on rows drawn alike. They
are not inverse variances.

## The measurement

A null test of the slope of `y = 1 + 0·x + ε`, with `T = 60`, 20 000 draws per scheme, at a nominal
size of 5 %:

| Weights | `T_e` | Sum one (before) | Scale to `T` | Kish |
| --- | --- | --- | --- | --- |
| uniform | 60.0 | 0.0 % | 4.9 % | 4.9 % |
| half-life 20 | 44.9 | 0.0 % | 9.0 % | 4.8 % |
| exponential tilt | 17.8 | 0.0 % | 25.0 % | 2.4 % |
| equal on 12 of 60 rows | 12.0 | 0.0 % | 42.7 % | 4.8 % |

The scale to `T` rejects a true null in up to 43 % of the fits, because it counts rows that carry
almost no weight. The Kish count stays at or below the nominal size.

## Decision

**A library regression target scales its weights to sum to their Kish count before the call to GLM,
and gives them to GLM as `StatsBase.FrequencyWeights`. Frequency weights pass unchanged.**

`glm_weights` in `src/05_Moments/20_Base_Regression.jl` holds the rule:

```julia
glm_weights(w::StatsBase.FrequencyWeights) = w
glm_weights(w::StatsBase.AbstractWeights) = StatsBase.fweights(w .* (sum(w) / sum(abs2, w)))
```

`glm_fit_kwargs` resolves a `DynamicAbstractWeights` against the design and applies `glm_weights`.
The `fit` methods of `LinearModel` and `GeneralisedLinearModel` both call it. An entry under
`weights` that is not an `ObsWeights` passes to GLM unchanged.

`FrequencyWeights` state a count, which is what GLM reads, so they are the door to any other count.
A caller who wants the row count passes `fweights(w .* (T / sum(w)))`.

## Consequences

- Uniform weights now give the inference of the unweighted fit, to rounding. A `StepwiseRegression`
  refitted inside a pooling prior with no views selects the factor sets of the unweighted fit.
- A stepwise set chosen under non-uniform weights moves, under the p-value rule and under `:aic`,
  `:aicc` and `:bic`. The coefficients of a fixed set do not move.
- The iterative fit of `GeneralisedLinearModel` stops when the change of the deviance is below
  `max(rtol · deviance, atol)`, and the deviance scales with the weights. The relative part does
  not move with the scale, but the absolute floor `atol` does not scale. A weighted fit can
  therefore stop at a different iteration than before, within that tolerance.
- The cross-sectional regression reads the coefficients alone, so its fits do not change.
- GLM gives no sandwich variance for probability weights. The Kish count is the correction that the
  model-based variance of GLM can carry. It is exact for an intercept and for equal weights on a
  subset of rows, and conservative on the tilted weights above.

## Alternatives considered

- **The row count `T`.** Rejected: the measurement above.
- **A field on the target that names the count.** Rejected: the kind of the weights already names
  it. `FrequencyWeights` are a count, and every other kind is a relative weight. A field would
  duplicate that choice.
- **The count of non-zero weights, as StatsBase corrects a variance under `ProbabilityWeights`.**
  Rejected: it is the row count whenever no weight is exactly zero, so it fails the same rows of the
  measurement as the scale to `T`.

## Related

- [ADR 0116](0116-a-prior-that-reweights-observations-works-on-the-axis-its-nested-prior-answered.md):
  its Context saw the `:bic` sets move under weights that sum to one.
- [ADR 0127](0127-the-compact-covariance-radius-is-sized-in-family-not-through-the-calibration-channel.md)
  and [ADR 0190](0190-a-regime-statistic-divides-by-the-bias-its-method-reads.md): the Kish count of
  observation weights elsewhere in the library.
