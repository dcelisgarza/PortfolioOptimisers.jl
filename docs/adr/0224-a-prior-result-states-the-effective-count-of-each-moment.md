---
status: accepted
---

# A prior result states the effective count of each moment

## Context

[#1892](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1892) made the denoiser of a
weighted covariance read the effective number of observations of the moment, through the verb
`effective_observation_count(ce, X)`. A weighted moment is worth fewer observations than `X` has
rows, so a denoiser that reads `size(X, 1)` reads a ratio `T / N` that is too large, and it removes
too little noise. [#1893](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1893)
recorded two cases that the same mechanism reaches, and that each needed a decision.

1. **`GerberIQCovariance`.** Its decay gives the old co-movements less weight. But its statistic is
   a ratio of weighted counts of co-movements over thresholds, and not a weighted moment, so no
   count follows from it.
2. **The matrices that a prior assembles.** A factor lift processes `M * f_sigma * M'`, the
   Black-Litterman priors process a posterior, the cross-sectional factor prior processes a
   systematic block, and the high order factor prior processes a lifted cokurtosis. Each denoiser
   read the number of rows of the returns that it was given. Each matrix is worth the count of the
   moments that it is assembled from.

A prior result already carried a count, `ens`, with two meanings (ADR 0138). An entropy pooling
prior and an opinion pooling prior wrote the perplexity `exp(entropy(w))` of their weights. A
Scenario Cap wrote the number of observations of the fit. `stated_sample_size` dispatched on the
weights of the result only to tell the two meanings apart. An `EmpiricalPrior` with a weighted
covariance wrote neither, so the normal uncertainty sets and the calibration rules priced its
moments at the number of rows. The perplexity is also not the Kish count, and the Kish count is
the exact count of the variance of a weighted mean of independent draws.

## Decision

1. **A count rule is an optional slot.** `AbstractObservationCountAlgorithm` is a new public family,
   and `KishObservationCount` is its first member. The verb `weight_observation_count(alg, w)`
   counts the observations that a vector of weights is worth. `GerberIQCovariance` takes
   `oc::Option{<:AbstractObservationCountAlgorithm} = nothing`. `nothing` keeps `size(X, 1)`, the
   behaviour before the slot existed. `KishObservationCount()` gives the Kish count of the temporal
   weights `decay(T, k)`, resolved on the centred returns as the fit resolves them.
2. **A prior result states the count of each moment in `eoc`.** `LowOrderPrior` holds
   `eoc = (T_mu, T_sigma)`. `HighOrderPrior` holds `eoc = (T_sk, T_kt)`, and its property `eoc`
   reads the four counts `(T_mu, T_sigma, T_sk, T_kt)` through a `swap` rule. An entry of `nothing`
   means `size(X, 1)`. `moment_observation_count(pr, k)` reads entry `k`.
3. **Each estimator of a moment states its own count.** `effective_observation_count` gains a
   default on `AbstractExpectedReturnsEstimator`, `CoskewnessEstimator` and `CokurtosisEstimator`,
   and methods on the members that weight their rows. `EmpiricalPrior` writes the counts of `me`
   and of `ce`. The two can carry different weights, so one number cannot hold both. A count equal
   to the rows that the result carries is written as `nothing`.
4. **A Scenario Cap writes its count in `eoc`, and `ens` is a diagnostic of `w` alone.** The cap
   fits every moment over every row, so each entry is the number of observations of the fit.
5. **A pooling prior keeps the counts of its refit.** It refits its prior under its weights `w`, so
   an estimator that reads `w` states the Kish count of `w`. `w` binds `eoc` in `forward_prior`, as
   it binds `ens`, `kld` and `ow`. A wrapper forwards `eoc`.
6. **Each consumer reads the count of the moment that it prices.**
   - The denoiser of a matrix that a prior assembles reads the count of the moments of that matrix,
     through `sample_matrix_processing!`. `factor_lift`, `cross_sectional_lift` and the public verb
     `systematic_processing!` take a positional `T`, with `size(X, 1)` as the default.
   - The default `tau = 1/T` of the four Black-Litterman priors reads the count of the prior mean.
     The augmented prior blends two priors with one `tau`, so it reads the smaller count of the two
     means, and its denoiser the smaller count of the two covariances.
   - A normal uncertainty set reads entry 1 for its mean set and entry 2 for its covariance set. A
     joint set reads each count for its own half.
   - A calibration rule does not know which moment it prices. With no weights it reads the smaller
     of entries 1 and 2, because the sample is worth no more than its less informed moment. It
     reads no `eoc` next to a `w`, as it read no `ens` next to a `w`.

## Considered options

- **A new verb on the prior estimator,** `prior_observation_count(pe, pr, X)`, with a method for
  each prior. Rejected. The count of a pooling prior is known only after the fit, the verb needs
  a method on every wrapper, and the uncertainty sets would still price a weighted `EmpiricalPrior`
  at its rows.
- **`EmpiricalPrior` writes the count of its `ce` in `ens`.** Rejected. One number cannot hold the
  counts of `me` and `ce`, and `ens` keeps a second meaning, the perplexity of the pooling priors.
- **The Kish count of `w` in each entry of a pooling prior.** Rejected. An estimator of the refit
  that reads no `w` does not weight its moment by `w`, and its own count is then the exact count.
  When every estimator reads `w`, the two rules give the same number.
- **A calibration rule reads the count of the covariance.** Rejected for the smallest count, which
  never prices more information than the result holds.

## Consequences

- An unweighted prior with no Scenario Cap states `(nothing, nothing)`, and every count reader gives
  the number it gave before.
- The normal uncertainty sets of an entropy pooling or an opinion pooling prior read the Kish count
  of `w` in place of the perplexity. The Kish count is smaller, so these sets are wider.
- The normal uncertainty sets, the calibration rules, the Black-Litterman `tau` and the denoisers
  of an `EmpiricalPrior` with a weighted `me` or `ce` read the count of its weights.
- `systematic_processing!` gains a positional argument. A user member of `AbstractSystematicRepair`
  must accept it.
- `scenario_ens` is replaced by `scenario_moment_counts`.
