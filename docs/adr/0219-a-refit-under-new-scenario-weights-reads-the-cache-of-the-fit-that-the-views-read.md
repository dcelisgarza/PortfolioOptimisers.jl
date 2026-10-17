---
status: accepted
---

# A refit under new scenario weights reads the cache of the fit that the views read

## Context

[#1751](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1751) found that an
entropy pooling prior over a `CrossSectionalFactorPrior` runs the regression stage of the
cross-sectional fit more times than the result needs. ADR 0205 holds the model of the prior that
an entropy pooling prior wraps: the refit under the posterior weights `w1` reweights the factor
prior `pe` alone, and keeps the regression, the cross-sectional weights, the residual block and
the Return Forecast of the fit under the prior probabilities `w0`. So everything that the
regression stage gives the refit is the output of the fit that the views read, but each refit
ran the stage again. On a panel of 300 observations and 60 assets the stage costs about two
thirds of a full fit. A staged entropy pooling prior refits up to three times, and a caller's
`w` added one more stage, to count the rows that `prior_probability_factory` weighs.

An `OpinionPoolingPrior` had the same repeat. Its refit fits `pe1` again for its rows, and runs
every opinion again for its pooled weights `wp`, which the fit under `w0` already computed.

The data that a refit reads is the data of a Result. An estimator must not hold a Result, and
ADR 0106 admits a partial-fit state alone. So the data cannot travel on the estimator.

## Decision

**The Result of the fit that the views read carries the cache, and the refit reads it from that
Result.**

1. `prior` takes the keyword `cache_reg`, `false` by default. Every `ReturnsResult` method of
   `prior` takes it and forwards it, and every prior that wraps another forwards it in its
   keyword arguments, as it forwards `ne`, `E`, `iv` and `ivpa`. A prior that stores no cache
   ignores it. The keyword is a decision of the call, not configuration, so the estimator that
   the caller gave does not change.
2. Under `cache_reg = true` a `CrossSectionalFactorPrior` stores the active-mask history `amr`
   on its block `CrossSectionalFactorModel`, and the Return Forecast history `hist` when a
   Calibration Slot reads it. Its regression `CrossSectionalRegression` records the live-factor
   mask `lv` on every fit, because the mask is a fact of the regression and costs one entry per
   factor. An `OpinionPoolingPrior` stores the rows of `pe1` and its pooled weights in the new
   field `pc` of `LowOrderPrior`, and passes `cache_reg` to `pe2`. A prior that fits a factor
   prior passes its keywords to that fit and carries its result in `fpr`: a `FactorPrior`, an
   `AugmentedBlackLittermanPrior`, and a `CrossSectionalFactorPrior`, whose `fpr` keeps the
   `pc` of a pool in its factor slot. An `AugmentedBlackLittermanPrior` keeps the `pc` of a
   pool in `a_pe` on its own result.
3. `EntropyPoolingPrior` and `MeucciEntropyPoolingPrior` fit the prior that they wrap with
   `cache_reg = true`, and give the Result `pr0` of that fit to every refit.
   `scenario_prior` and `scenario_factory` take it as a positional argument after `w`, and each
   method gives a nested prior the part of `pr0` that its fit answered: `pr0` itself through an
   entropy pooling prior, a Black-Litterman prior, a pool and the asset prior of an augmented
   Black-Litterman prior, and `pr0.fpr` through every prior that fits a factor prior. `BlackLittermanPrior` and `BayesianBlackLittermanPrior` refit their prior through
   `scenario_prior`, and blend their views into its result with `bl_posterior` and
   `bbl_posterior`, so they reach the cache too.
4. A refit that finds a cache runs only the steps that read the new weights. The refit of a
   `CrossSectionalFactorPrior` fits its factor prior on the factor returns of the block, splits
   the Return Forecast that the block carries, and runs the steps of the assembly that read the
   factor moments, `cross_sectional_moment_prior`. It reads the scenarios `pr0.X` and the
   base-currency returns `pr0.o_X`, which no weight moves. The refit of a pool reads its rows and
   its pooled weights from `pc`, and runs no opinion and no `pe1`. Its result keeps `pc`, and an
   entropy pooling result forwards it, so a nested entropy pooling prior finds it too. The
   refit of a `CrossSectionalFactorPrior` fits the model of a pool in its factor slot, which
   keeps no cache, so its `fpr` keeps the `pc` of the fit for the same reason.
5. A Result with no cache, or `nothing`, changes nothing: the refit reads the data, as it did
   before. A block that a caller builds by hand carries no cache, and a block whose asset axis
   is not the axis of the refit is not read.
6. `prior_probability_factory` reads the data of a prior that reads no Asset Panel only for its
   row count. A method that takes the count `n` in place of the data holds the rules of such a
   prior, and the methods on the data answer through it. A `CrossSectionalFactorPrior` reads `n`
   off its exposure stage, so the count runs no regression.

## Alternatives rejected

- **Pass the regression stage to the refit as an argument.** The stage of a nested prior is
  computed inside the fit of the prior that wraps it, so every `ep_prior` and every wrapper
  would have to return it beside its Result.
- **A field on the estimator.** An estimator that holds the output of a fit holds a Result,
  which ADR 0106 refuses. A Boolean switch on the estimator would also need a verb that sets it
  through every wrapper, and the entropy pooling prior would change the estimator that the
  caller gave it.
- **Store the cache on every fit.** The active-mask history is small, but the Return Forecast
  history and the rows of a pool are full matrices. A plain fit has no refit to serve, so it
  stores nothing new.

## Consequences

- The cached refit is bit-identical to the full refit on the panel of
  `test_12k_cross_sectional_factor_prior.jl`. On the panel of the issue it takes 14.2 ms and
  2.2 MB in place of 42.8 ms and 49.0 MB. The fit of the factor prior under the new weights is
  12.2 ms of it, and that step must run again under every new weight.
- `scenario_prior` and `scenario_factory` take one more positional argument. Both are internal
  verbs.
- `LowOrderPrior` has the field `pc`. Its view on a sub-universe drops it, because its rows hold
  every asset.
- On the panel of the issue, ten compositions give bit-identical results before and after the
  decision: an entropy pooling prior over the Cross-Sectional Factor Prior with a caller's `w`,
  with two stages, with a Family re-basis, with a Return Forecast whose slots read its history,
  nested, under Meucci, and under a Black-Litterman prior, and over a pool, a pool with `pe1`,
  and a factor prior over a pool. The fit with a caller's `w` takes 96 ms and 71 MB in place of
  174 ms and 155 MB, single-threaded.
