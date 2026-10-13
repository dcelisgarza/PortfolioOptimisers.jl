---
status: accepted
---

# A high order prior fits its co-moments under the probabilities of the scenarios

## Context

[#1650](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1650) found that
`HighOrderPriorEstimator` and `HighOrderFactorPriorEstimator` over an `EntropyPoolingPrior` or a
`MeucciEntropyPoolingPrior` fit their co-moments without the posterior weights `w1`. The low order
block carried the views in `mu`, `sigma` and `w`, and the coskewness and the cokurtosis were those
of the prior with no view. A skewness or a kurtosis view changes `w` alone, so it reached no
co-moment. On the data of the issue a skewness view of 0.5 on asset 2 held on the scenarios under
`w1`, and `sk` equalled the `sk` of `HighOrderPriorEstimator()`.

`HighOrderPriorEstimator` fitted `cokurtosis(pe.kte, X, pnl)` and `coskewness(pe.ske, X, pnl)`,
and `HighOrderFactorPriorEstimator` fitted the same verbs on `F`. Neither read `pr.w`.

## Decision

**The co-moments are those of the distribution that the low order block states.** Entropy pooling
holds the scenarios and changes only their probabilities (ADR 0205), so its posterior co-moments
are the co-moments of the same scenarios under `w1`.

1. Both estimators fit their co-moments through `factory(pe.kte, pr.w)` and
   `factory(pe.ske, pr.w)`. `pr.w` replaces the observation weights of the two estimators, and a
   prior result that answers no `w` leaves them as they are. This is the contract that
   `fit_deferred_quantity` already states for a deferred cokurtosis and coskewness, and the rule
   by which entropy pooling refits `mu` and `sigma` under `w1` through `factory`.
2. The rule reads the `w` of every prior result, not only of the priors that reweight scenarios.
   A prior result answers `w` as the probabilities of its scenarios, and every risk measure that
   reads `pr.w` reads it so. `EmpiricalPrior` answers no `w`, so its high order prior does not
   move. A `FactorPrior` answers the `w` of its factor prior, which weighs the same observations.
3. A prior result answers `w` on the rows of its own `X`, which can be the last rows of the
   returns alone: a Scenario Cap keeps the last rows, and a `CrossSectionalFactorPrior` drops the
   rows that its Descriptors warm up over. `scenario_rows` cuts the returns, their Asset Panel and
   the factor returns to the last `length(w)` rows, the rule of step 6 of `FactorPrior` and of
   ADR 0116. It refuses a `w` that is longer than the returns.
4. The residual co-moments of `HighOrderFactorPriorEstimator` read no `w`. The residuals belong
   to the model, which the views do not move, and ADR 0205 holds the residual block of `sigma` at
   the fit of the model for the same reason.
5. The online read of `HighOrderPriorEstimator` refits both co-moments under `pr.w` when the
   embedded result answers one, through `read_comoment`, whether the member folded or not. A state
   folds the rows under equal weights, and it cannot read `w`.

## Alternatives rejected

- **Read `pr.w` only when `kte` or `ske` carries no weights of its own.** This is the rule of the
  risk measures, where `r.w` wins. Here a caller's `w` on a co-moment estimator states other
  probabilities than the ones that the low order block states, so `mu`, `sigma` and the
  co-moments would describe two distributions. The entropy pooling refit replaces the weights of
  `pe.me` and `pe.ce` in the same way.
- **Read `pr.w` only under a prior that reweights scenarios.** The `w` of a `FactorPrior` is the
  `w` of its factor prior, which can be an entropy pooling prior of the factors. A view on a
  factor then reaches no co-moment, and the composition breaks: an entropy pooling prior over a
  `FactorPrior` reaches the factor co-moments, and a `FactorPrior` over an entropy pooling prior
  of the factors does not.
- **Fit the co-moments on `pr.X` under `w`.** `HighOrderPriorEstimator` fits on the caller's
  returns on purpose, because `pr.X` can hold the scenario fill and the model rows of a factor
  prior. The weights move, the rows do not.

## Consequences

- Released numbers move for every high order prior over a prior result that answers a `w` other
  than equal weights: an entropy pooling prior with a view or a tilt, an opinion pooling prior,
  and a factor prior or a Black-Litterman prior over one of them. Under equal weights the
  co-moments equal the old ones to rounding, because a weighted mean under equal weights is the
  plain mean.
- On the data of #1650, the standardised third moment of asset 2 that `sk` holds is 0.5, the
  view, and a kurtosis view of 4.0 on asset 3 reads 4.0 off `kt`. A skewness view of 0.4 on a
  factor inside a `FactorPrior` reads 0.4 off `fpr.sk`.
- The warning of `HighOrderFactorPriorEstimator` about a Black-Litterman prior stays true. A
  Black-Litterman prior answers the `w` of the prior it wraps, and it has no higher moment update
  to state.
- `EntropyPoolingPrior` with a view over a prior with a Scenario Cap fails before it reaches a
  high order estimator, with a bare `DimensionMismatch`. That is not this decision, and
  [#1667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1667) records it.
