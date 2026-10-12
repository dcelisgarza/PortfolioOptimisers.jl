---
status: accepted
---

# An entropy pooling prior reweights the scenarios of a factor prior, and holds its model

## Context

[#1649](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1649) found that
`EntropyPoolingPrior` and `MeucciEntropyPoolingPrior` over a time-series factor prior do not meet
their own views.

Both priors build their views on the rows `pr.X` of the fit of `pe` under the prior probabilities
`w0`. A `FactorPrior` makes those rows from its regression, `F * M' .+ b'`. After each solve, the
prior refitted `pe` through `factory(pe, w1)`, and `factory` gives the posterior weights to every
weighted field: the factor prior, the regression `re` and the residual variance estimator `ve`.
So the refit fitted a new regression, the rows moved, and the read-out reported the moments of
other scenarios than the views constrained. On the data of the issue, a mean view of 0.004 read
out as 0.004022 over `FactorPrior()` and 0.003954 over a PCA regression. A variance view under
`H2_EntropyPooling` over `FactorPrior()` failed with a `PosDefException`, because the stepwise
regression under the concentrated posterior weights was singular.

A second gap stayed behind the first. Under `rsd = true` the factor prior adds the residual block
`D` to `sigma`, and the rows `pr.X` do not carry it. A variance view reads the rows, so even with
the rows held, the read-out variance was the view plus `D`: 3.72e-5 for a view of 3.0e-5. The
`prior(...)` reference of a variance reads `pr.sigma`, which holds `D`, so `A == prior(A)` asked
the rows for the variance of the model. `fix_sigma!` held the rows at `pr.sigma` in the same way,
and the skewness row read `pr.sigma` as the variance of the rows in its own expansion.

## Decision

**Entropy pooling holds the scenarios and changes only their probabilities** (Meucci, *Fully
Flexible Views*, 2008). For a time-series factor prior the scenarios are the rows of its model,
so the model is held too.

1. Every refit under the posterior weights goes through `scenario_prior`, which fits the
   estimator of `scenario_factory`. That estimator gives the weights to the estimators that read
   the scenarios, and keeps the rest. A `FactorPrior` and a `FactorBlackLittermanPrior` reweight
   `pe` alone, and keep `re` and `ve`. An `AugmentedBlackLittermanPrior` reweights `a_pe` and
   `f_pe`, and keeps `re`. A Black-Litterman wrapper applies the rule to the prior it embeds.
   Any other prior takes `factory`: its rows are the returns, which no weight moves. Each of
   these recurses through `scenario_factory`, never through `factory`.
   A prior that reweights scenarios itself is replaced by its model. A nested entropy pooling
   prior becomes `pe.pe` under its own `w`, in `scenario_factory`, so an entropy pooling prior
   of the factors inside a `FactorPrior` is covered too. An `OpinionPoolingPrior` has no single
   estimator for its model, so `scenario_prior` refits `pe2` on the rows of `pe1`. The
   posterior of such a prior is already the prior probabilities `w0` of the host, so its views
   or opinions are in `w1` too. The refit through `factory` solved them again from `w1`, so the
   read-out reported the moments of other weights than the `w` of the result: on the data of
   #1649, `mu` and `X' * w` differed by 0.0052 for a nested entropy pooling prior, and the view
   of the host read -0.0004 for 0.004.
2. The posterior is then `mu = M f_mu(w1) + b` and `sigma = M f_sigma(w1) M' + D`, with `M`, `b`
   and `D` of the `w0` fit. With the weighted sample mean of the factors, `mu` is
   `pr.X' * w1`, so every mean view holds.
3. A view row that states a moment the read-out reports takes the residual block off its target.
   `scenario_residual_variances` answers `D` from `rr.esigma` of a `Regression`, and zero for
   anything else. The variance row and a covariance or correlation row on one asset subtract it.
   `fix_sigma!` holds the rows at `sigma - D`, the variance of the scenarios. The skewness and
   kurtosis rows read `sigma - D` too, so they state the moments of the scenarios, which is what
   their `prior(...)` references read. A low order factor model states its residual to the second
   moment alone, so no higher moment of the model exists to state.

The decision makes two compositions one model. A mean view on an asset over `FactorPrior` now
equals `FactorPrior` over an entropy pooling prior of the factors with the equivalent factor view,
whose regression and residual block read no posterior weights. On the data of the issue the two
agree to the tolerance of the solver: the weights to 8.9e-11, `mu` to 8.0e-12 and `sigma` to
1.4e-10 relative.

## Alternatives rejected

- **Read the posterior as the moments of `pr.X` under `w1`.** The mean is the same, but `sigma`
  loses the residual block, so it is singular at rank `K` and drops what `rsd = true` asked for.
- **Reweight the residual block under `w1`.** The residuals are not scenarios that the views read,
  so the solve does not constrain them, and the variance of a residual under `w1` is not linear in
  the posterior probabilities, so a variance view cannot take it off its target. It also breaks
  the equality of the two compositions above.
- **Hold the regression and leave the views alone.** The mean views hold, but every variance view
  over `rsd = true` misses by `D`, and `fix_sigma!` moves the variance that it is meant to hold.

## Consequences

- Released numbers move for every entropy pooling prior over a time-series factor prior: the
  read-out, and the posterior weights of every variance, covariance, correlation, skewness or
  kurtosis view over `rsd = true`. They also move for an entropy pooling prior nested in another,
  whose host now meets its own views. An entropy pooling prior over any other prior is unchanged,
  because its `D` is zero and `scenario_factory` is `factory`.
- The old refit rebuilt the host with `w1`, and the constructor of the host normalises `w` in
  place, so the refit normalised the posterior weights as a side effect. Each solve site now
  normalises them itself. Without that step the weights summed to one to the tolerance of the
  solver alone, 1.3e-6 off on the Meucci test, and with it the weights are bit-identical to the
  old ones. Because `scenario_factory` replaces every entropy pooling prior by its model, no
  entropy pooling constructor reads `w1`, and the solve site pays the normalisation once. The
  old refit paid it twice or more when an entropy pooling prior sat inside a factor prior or a
  Black-Litterman prior, and there it also solved the inner views again: on the data of #1649
  a factor view inside a `FactorPrior` moved the view of the host to 0.0039853 for 0.004.
- Under `PinnedChoice` and `BatchChoice` alike, every refit reads the regression of the `w0`
  fit, so the two Choice Rules agree in the batch fit of an entropy pooling prior again. The
  exception that #1641 recorded in ADR 0193 is gone.
- A `CrossSectionalFactorPrior` makes its rows with the idiosyncratic returns in them, so its `D`
  is zero and it takes `factory`.
- A view that only a nested entropy pooling prior states is a belief of the prior of the host,
  not a view of the host. The posterior of the host is the smallest move from it that meets the
  views of the host, so the read-out can leave it, as Meucci's sequential reading states. A
  caller who wants both views held states both in one prior.
- An `OpinionPoolingPrior` reweights scenarios too, and it had the same defect: its opinions
  read `w1` as their prior probabilities and pooled again. On the data of #1649 the view of the
  host read 0.0028 for 0.004. Its refit now holds the rows of `pe1` and refits `pe2` under
  `w1`.
- `HighOrderFactorPriorEstimator` over an entropy pooling prior of a factor prior now lifts its
  co-moments through the loadings that the views read. The residual block that it takes off
  `sigma` is read from `rr.esigma`, the block that the prior added under the weights of its
  fit. It was a variance of the residuals under the declared `ve` before, which missed the block
  by 7.5 % under a caller's `w` and by 1.3 % under an opinion pool. Its factor co-moments still
  read the weights of its own `kte` and `ske`, not `w1`, which
  [#1650](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1650) records.
- An `OpinionPoolingPrior` on the factors inside a `FactorPrior` still re-pools under `w1`,
  because `scenario_factory` sees no data to pool on.
  [#1652](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1652) records it.
