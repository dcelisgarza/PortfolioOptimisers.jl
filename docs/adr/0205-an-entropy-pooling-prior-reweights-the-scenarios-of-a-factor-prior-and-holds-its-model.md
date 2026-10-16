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
   `pe` alone, and keep `re` and `ve`. A `CrossSectionalFactorPrior` reweights `pe` alone, on
   the factor returns that its fit gives `pe`, and keeps its regression, the weights of its
   cross-sectional fit, its residual block and its Return Forecast
   ([#1685](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1685)). An
   `AugmentedBlackLittermanPrior` reweights `a_pe` and `f_pe`, and keeps `re`. A
   Black-Litterman wrapper applies the rule to the prior it embeds.
   Any other prior takes `factory`: its rows are the returns, which no weight moves. Each of
   these recurses through `scenario_factory`, never through `factory`.
   A prior that reweights scenarios itself is replaced by its model. A nested entropy pooling
   prior becomes `pe.pe` under its own `w`, in `scenario_factory`, so an entropy pooling prior
   of the factors inside a `FactorPrior` is covered too. An `OpinionPoolingPrior` has no single
   estimator for its model, because its pooled weights `wp` come from running the pool on its
   rows. So `scenario_factory` takes the data of the fit that it reweights, and each method
   gives a nested prior the rows that its own fit gives it: a factor prior gives `F` to its
   `pe`. A pool then computes `wp` on those rows and becomes `pe2` under `wp`, reweighted under
   `w1`. A pool that the host embeds directly is refitted by `scenario_prior` on the rows of
   `pe1`. A pool with `pe1` inside a factor prior or a Black-Litterman prior reads other rows
   than the rows that the wrapping prior gives it, and no estimator states that model, so the
   constructor of the host refuses it through `assert_scenario_refit`
   ([#1652](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1652)). The
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
   `fix_sigma!` holds the rows at `sigma - D`. The skewness and kurtosis rows read `sigma - D`
   too, so they state the moments of the scenarios, which is what their `prior(...)` references
   read. `sigma - D` is the corrected variance of the scenarios, and the rows state the
   uncorrected moment, so a stage that pins the variance moves it by the correction of the
   stage before. #1741 records it. So the views hold on the scenarios, and the read-out `sigma`
   of a variance view differs from its target by that correction (ADR 0210). A low order factor model states its residual to the second
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
- A `CrossSectionalFactorPrior` takes the rule of a `FactorPrior`
  ([#1685](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1685)). It took
  `factory` before, which gave the posterior weights to every field, and a Scenario Cap in its
  `pe` then failed with a bare `DimensionMismatch`, because its fields read every fitted
  observation. The number of these observations is known only after the regression, because the
  Descriptor warm-up and the exposure lag drop rows, so `scenario_factory` and
  `prior_probability_factory` read the factor returns of `pe`, or their count. The refit reads
  the regression of the fit that the views read from its Result, and
  `prior_probability_factory` counts the rows with the exposure stage alone, so neither fits the
  regression again (ADR 0219). No default field of
  the prior other than `pe` reads weights through `factory`, so on the panel of
  `test_12k_cross_sectional_factor_prior.jl` the posterior without a cap is bit-identical to the
  old one. A field that reads observation weights, such as a `ce` with weights, now keeps the
  weights of the fit that the views read.
- The rows of a `CrossSectionalFactorPrior` carry the idiosyncratic returns, and its `mu` is
  `L f_mu + b`, which sets the mean of the idiosyncratic returns to zero. So a mean view holds on
  `pr.X' * w`, and the read-out `mu` misses it by the posterior mean of the idiosyncratic part of
  the rows. On the panel of the test, a view of 0.002 reads out as 0.0015321 without a cap, as
  before this decision, and as 0.0037980 with a cap of 30 scenarios. `scenario_residual_variances`
  answers zero for its block, because its rows already carry the residual.
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
  by 7.5 % under a caller's `w` and by 1.3 % under an opinion pool. Its factor co-moments read
  `w1`, and its residual co-moments stay with the model as the residual block does, which
  [ADR 0206](0206-a-high-order-prior-fits-its-co-moments-under-the-probabilities-of-the-scenarios.md)
  states.
- An `OpinionPoolingPrior` on the factors inside a `FactorPrior` re-pooled under `w1` while
  `scenario_factory` saw no data to pool on. On the data of #1652 the view of the host read
  0.0033538 for 0.004, and `mu` missed `X' * w` by 0.0011940. It now meets the view, and the
  same holds for a pool inside a `FactorBlackLittermanPrior`, a Black-Litterman wrapper or an
  `AugmentedBlackLittermanPrior`.
- A pool with `pe1` inside a factor prior or a Black-Litterman prior under an entropy pooling
  prior is refused when the host is built. It was accepted before, and its read-out missed the
  views. A caller sets `pe1 = nothing` in the pool, or gives the pool to the host directly.
