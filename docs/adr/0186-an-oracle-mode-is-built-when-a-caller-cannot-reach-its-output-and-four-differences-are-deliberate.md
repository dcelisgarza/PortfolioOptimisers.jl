---
status: accepted
---

# An oracle mode is built when a caller cannot reach its output, and four differences are deliberate

## Context

[Map #1375](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1375) measures the
cross-sectional factor prior and everything around it against an oracle: an independent
implementation of the same models. The oracle has modes that the library does not port.
[Issue #1392](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1392) listed sixteen of
them and decided each one.

A mode of the oracle can be missing from the library in three ways:

1. The library has no route to the mode's output.
2. The library reaches the same output by its own idiom: a composition, a view, a rebuild through a
   checked constructor, or a keyword of a different estimator.
3. The mode contradicts a rule that the library already records.

## Decision

**The library does at least what the oracle does, and never less.** No decision removes a
capability of the oracle. A mode is built when a caller cannot reach its output today. A mode that
a library route reaches gets that route documented, and the build of a convenience surface is
decided mode by mode. The parity target is the output, not the shape of the oracle's call. Where
the library's own rule is better, the library's rule is the default and the oracle's rule stays one
keyword away.

### The deliberate differences are in a default or a form, not in capability

| Mode of the oracle | The library | Why |
| --- | --- | --- |
| A nearest-correlation repair by clipping the correlation eigenvalues at `1e-13`, with one retry at `1e-12` | `Posdef(Newton)` in `f_mp` and `mp` stays the default. The oracle's repair is `ClippedNearestCorrelation`, an algorithm that `Posdef` takes and that owns its acceptance test (#1412). It differs in three places: alternating projections that have not converged after `iter` iterations go on to the clip where the oracle refuses the input, a last result that fails every acceptance warns as every `posdef!` does, and the symmetry test reads the correlation matrix so that it does not move with the units. | Newton gives the nearest correlation matrix in the Frobenius norm. The oracle's clip is a cheaper approximation of it, and a caller who wants the oracle's numbers selects it. |
| An inactive-cell policy stored on each field: `NaN`, zero, or the value left as it is | Fields stay finite, and every consumer reads the masks. A read of a field takes the policy as an argument, so the caller chooses it for each read. | ADR 0102 rules that a numeric field's values stay finite. Every computed output keeps the oracle's capability: the descriptors write `NaN` on an inactive cell, and the weights are built over the estimation universe. The one view the oracle gives, a field with its inactive cells blanked, becomes a read, not a stored state. |
| A warning when the regime half-life exceeds 138 observations | The `regime_decay` docstring states the threshold `2^(-1/138)` and its effect | The rule of #1282: the docstring states the condition, and the measure does not change its behaviour. The numbers are the same; only the message at run time differs. |
| An integer `cv`, which means K-fold with that many folds | `cv = KFold(; n)` | No field or verb of the library takes an integer as a short form for an estimator. |

The oracle's history cap on its prior truncates the stored histories and the scenarios, and leaves
the moments folded over every observation. The library reaches the scenario cap through
`pe = EmpiricalPrior(; max_scenarios)`, because the cross-sectional prior builds its scenarios from
the last rows of the factor scenarios. That is a route, not a difference. ADR 0136 already
separates a scenario cap from the window of `Online(…; max_history)`.

### The square root of a covariance is a named policy

The library took the square root of a covariance by three unnamed policies: a bare Cholesky that
throws, `covariance_factor` (a Cholesky, else an exact eigen square root of a positive semidefinite
matrix), and `safe_regime_cholesky` (a Cholesky with a ridge that grows tenfold over three tries,
which is the oracle's policy). An algorithm type now names the last two, and a field on
`CrossSectionalFactorPrior` and `FactorPrior` selects one. The field defaults to `nothing`, which
keeps the bare Cholesky that throws. With the default repair on, all three agree, because the
matrix reaches the Cholesky positive definite. They differ only when a caller turns the repair off
or the repair fails.

Issue #1410 gave the field, named `mtx_sqrt`, to every estimator that takes a square root of a
covariance, and removed `covariance_factor`, which `matrix_square_root` supersedes. The owners are
the two factor priors, `Variance`, `StandardDeviation`, `DistributionValueatRisk`,
`UncertaintySetVariance`, `ArithmeticReturn`, `RelaxedRiskBudgeting`, `Kurtosis`,
`NegativeSkewness` and `NormBallUncertaintySetAlgorithm`. A default keeps what the site did before:
`nothing` where it took the bare Cholesky, and `EigenFallbackSquareRoot()` where it took
`covariance_factor` or `sqrt(V)`. A square root that feeds only a second-order cone needs
`G' G = Σ`, so a singular positive semidefinite matrix is valid input there. Four sites keep a bare
Cholesky with a documented refusal, because each needs an inverse, which a singular matrix does
not have: the whitening of the radial tail calibration, the QLIKE of the covariance forecast
evaluation, the geodesic shrinkage target, and the Gram matrix of `GramProjection`.

### Built

The build tickets are children of map #1375: #1396 to #1408, #1411 and #1412. #1409 decides the
route of the prior onto the online seam, and it keeps reachable both the oracle's window (the
sample is cut once, at the first fit, and every later row is folded in) and the library's window
(always the last rows). #1392 holds the table of all sixteen modes.

### A lost mask is caught by a test, not by a `NaN`

The oracle's default policy makes a consumer that forgets the mask give `NaN`, which a test sees at
once. With finite fields the same defect gives a plausible answer. #1398 is a real case: a windowed
wrapper dropped the active mask of an Asset Panel. So a census test (#1411) writes an extreme value
into every inactive cell and asserts that the answer of every consumer of an Asset Panel does not
change. It gives the safety of the oracle's `NaN` with no cost at run time.

## Consequences

- A measure ticket of map #1375 that finds a mode the oracle has and the library lacks applies the
  rule above before it asks the maintainer. A design that leaves an oracle output out of reach
  fails the rule, however simple it is.
- A parity test of the nearest-correlation repair compares the oracle with the algorithm of #1412,
  not with Newton. `test/test_07c_parity_clipped_nearest_correlation.jl` measures it on every exit
  path of the repair, and it is the measure of the last box of #1383.
- The premise of #929, that the oracle persists nothing, was wrong. #1399 builds the round trip
  that parity needs.
