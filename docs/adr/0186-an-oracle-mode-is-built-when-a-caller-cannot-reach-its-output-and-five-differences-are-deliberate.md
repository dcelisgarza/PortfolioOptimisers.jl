---
status: accepted
---

# An oracle mode is built when a caller cannot reach its output, and seven differences are deliberate

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
| A nearest-correlation repair by clipping the correlation eigenvalues at `1e-13`, with one retry at `1e-12` | `Posdef(Newton)` in `f_mp` and `mp` stays the default. The oracle's repair is `ClippedNearestCorrelation`, an algorithm that `Posdef` takes and that owns its acceptance test (#1412). It differs in three places: alternating projections that have not converged after `iter` iterations go on to the clip where the oracle refuses the input, a last result that fails every acceptance is returned when it is positive semidefinite to round-off, where the oracle refuses it, and the symmetry test reads the correlation matrix so that it does not move with the units. | Newton gives the nearest correlation matrix in the Frobenius norm. The oracle's clip is a cheaper approximation of it, and a caller who wants the oracle's numbers selects it. |
| An inactive-cell policy stored on each field: `NaN`, zero, or the value left as it is | Fields stay finite, and every consumer reads the masks. A read of a field takes the policy as an argument, so the caller chooses it for each read. | ADR 0102 rules that a numeric field's values stay finite. Every computed output keeps the oracle's capability: the descriptors write `NaN` on an inactive cell, and the weights are built over the estimation universe. The one view the oracle gives, a field with its inactive cells blanked, becomes a read, not a stored state. |
| A warning when the regime half-life exceeds 138 observations | The `regime_decay` docstring states the threshold `2^(-1/138)` and its effect | The rule of #1282: the docstring states the condition, and the measure does not change its behaviour. The numbers are the same; only the message at run time differs. |
| An integer `cv`, which means K-fold with that many folds | `cv = KFold(; n)` | No field or verb of the library takes an integer as a short form for an estimator. |
| A Spanned Shrinkage `lambda` of one, which keeps the fitted factor mean | `lambda = PrecisionBlend()` is the default, and `lambda = 1` gives the oracle's mean. Every stored parity case of the Cross-Sectional Factor Prior states `lambda = 1`. | The precision blend weighs the factor mean and the spanned part of the Return Forecast by the error of each. On the known truth of [#1480](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1480) and [#1482](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1482) it had a smaller `Σ⁻¹` error of the mean than `lambda = 1` in every cell, and with no Return Forecast it shrinks the factor mean towards zero, the largest gain measured (83.9 to 16.7 at 20 factors and 60 rows). [#1483](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1483) built it. The Orthogonal Forecast Scale `c` keeps the oracle's `1`, because its rule reads a history that costs minutes to make. |
| A Neutralisation of the Descriptor scores with no intercept | `DescriptorScores` defaults to `cre = CrossSectionalLinearRegression(; intercept = true)`, and `intercept = false` gives the oracle's rule. The parity cases of the Return Forecasts state `intercept = false`. | Without an intercept the residual is orthogonal to each target in the uncentred product alone, so the neutralised score still correlates with a target set that does not span the constant. With an intercept it is uncorrelated with each target under every design, and where the targets span the constant the two rules give the same residual. [#1521](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1521) built it, and it reverses #950. |
| A fitted Return Forecast read as it stands inside the prior | `CrossSectionalFactorPrior` defaults to `ofit = ScoreNeutralisation()`, and `ofit = UnadjustedForecast()` gives the oracle's form. Every stored parity case with a fitted member states `UnadjustedForecast()`. | A fitted member regresses an idiosyncratic return that the cross-sectional regression makes orthogonal to the loadings, so its orthogonal part is under-scaled by `var(s⊥) / var(s)`. The default neutralises its scores against the estimated factors, so the orthogonal part carries the scale of its fit ([ADR 0194](0194-a-fitted-return-forecast-neutralises-its-scores-so-its-orthogonal-part-carries-the-scale-of-its-fit.md)). |

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
`CrossSectionalFactorPrior` and `FactorPrior` selects one. The field defaults to
`EigenFallbackSquareRoot()`, and `nothing` keeps the bare Cholesky that throws. With the default
repair on, all three agree, because the matrix reaches the Cholesky positive definite. They differ
only when a caller turns the repair off.

Issue #1410 gave the field, named `mtx_sqrt`, to every estimator that takes a square root of a
covariance, and removed `covariance_factor`, which `matrix_square_root` supersedes. The owners are
the two factor priors, `Variance`, `StandardDeviation`, `DistributionValueatRisk`,
`UncertaintySetVariance`, `ArithmeticReturn`, `RelaxedRiskBudgeting`, `Kurtosis`,
`NegativeSkewness` and `NormBallUncertaintySetAlgorithm`. Every one of them defaults to
`EigenFallbackSquareRoot()`, and the oracle's ridge, `RidgeCholeskySquareRoot()`, and the bare
Cholesky, `nothing`, are one keyword away. A square root that feeds only a second-order cone needs
`G' G = Σ`, so a singular positive semidefinite matrix is valid input there.

The eigen square root is the default because it is the only one of the three that is exact on
every valid input. On a positive definite matrix it returns the plain lower Cholesky factor, so no
value changes where the bare Cholesky succeeds. On a positive semidefinite matrix of lower rank it
returns `V max(Λ, 0)^(1/2)`, and `L L' = Σ` holds to round-off. The ridge returns the root of a
different matrix, `Σ + 1e-12 s I`. A bare Cholesky of a singular matrix succeeds only by
round-off, so the pivot of the null direction carries no information. The eigen square root still
refuses an indefinite matrix and a matrix that is not Hermitian, so it hides no data error.

### The calibration warm-up of a fitted forecast is a named rule

`TargetReturnForecast` calibrates on out-of-fold predictions, and a cross-validation estimator
needs two valid samples per fold. Below that count no prediction is out of fold, so no
out-of-fold slope exists. The library states that with a `NaN` slope, `NaNWarmup()`, the default.
The oracle then calibrates on the in-sample predictions of the fitted model. An in-sample slope
is biased upward, because each prediction comes from a model that trained on its own target. But
it is a biased estimate and not a wrong formula, so `warmup = InSampleWarmup()` gives it
(#1512). On the fixture of 9 valid samples under five folds, the two agree to round-off. With an
explicit cross-validation estimator below the count, the oracle refuses. That refusal rejects
valid input, so no keyword reproduces it.

### A repair returns a positive semidefinite matrix or refuses

Every repair of `Posdef` ends with one test: the smallest eigenvalue of the result is at least
`-n eps max|λ|`, the tolerance of the eigen square root. A result that passes returns with no
message, singular or not. A result that fails raises a `PosdefRepairError` that names the
algorithm, the smallest eigenvalue and the tolerance. A repair returns a matrix that a consumer
factorises or optimises over, and a matrix below the tolerance has a negative variance in some
direction, so a warning before the return would leave the defect in the output. A message on a
result that passes is noise: the Newton repair warned on the singular result of a zero-variance
row, and the clip warned on a result whose smallest eigenvalue is a round-off below zero. The
clip makes a positive semidefinite matrix by construction, so its refusal guards against round-off
alone. Four sites keep a bare
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

### Three kept differences that no other record owns

[#1416](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1416) gave a last verdict on
every difference from the oracle. Three kept differences had no ADR, so their reasons are stated
here.

- **A windowed variance series rolls with the observation.** Row `t` of `variance_series` on a
  windowed wrapper is the wrapper fitted on rows `max(1, t - w + 1):t`. The oracle's prior cuts its
  window once, at a first call that holds one row, so its `window_size` has no effect there
  (measured difference exactly 0). A window of `w` is the estimate over the last `w` rows, and the
  oracle's own batch fit keeps that definition, so its prior contradicts its own documented
  formula. The oracle's output is the unwrapped estimator in the variance slot.
- **`ConstantExposure` is `NaN` on an inactive cell.** The exposure matrix `B_t` of an
  observation is defined over the universe of that observation, so the model states no loading for
  an asset outside it, the constant included. Every other exposure is `NaN` there, so a row is
  either wholly in the model or wholly `NaN`, and the universe can be read off any column, also
  off a model whose only exposure is the constant. No fitted value moves. The oracle's matrix is
  `ifelse.(amsk, B, 1)` on the constant column.
- **An attribution defaults to `ppy = 1` and `se = false`.** `ppy` only scales a report, and 1 is
  the identity at every data frequency, where the oracle's 252 assumes daily data. A standard error
  of a realised attribution needs the regression weights and the idiosyncratic variance at each
  observation. A block with static loadings records neither, so `se = true` as the default would
  refuse every `FactorPrior` block. `ppy = 252, se = true` gives every number of the oracle.

## Consequences

- A measure ticket of map #1375 that finds a mode the oracle has and the library lacks applies the
  rule above before it asks the maintainer. A design that leaves an oracle output out of reach
  fails the rule, however simple it is.
- A parity test of the nearest-correlation repair compares the oracle with the algorithm of #1412,
  not with Newton. `test/test_07c_parity_clipped_nearest_correlation.jl` measures it on every exit
  path of the repair, and it is the measure of the last box of #1383.
- The premise of #929, that the oracle persists nothing, was wrong. #1399 builds the round trip
  that parity needs.
