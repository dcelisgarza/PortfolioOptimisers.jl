---
status: accepted
---

# A fitted forecast calibrates on the forecasts it published

## Context

`TargetReturnForecast` fits a regression target on a transformed forward return, so its
prediction has no unit. One exponentially weighted scalar regression of the forward return on the
prediction puts it into return units. The field `cv` chooses the predictions that this regression
reads. Its default was `KFold()` (#1418): five consecutive folds, no purge.

[#1567](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1567) found three faults in
that default.

1. `target_forecast_samples` writes the samples observation by observation, and `KFold` cuts
   consecutive blocks. So the model that predicts an early fold trains on later observations.
   The calibration slope is a measure of skill, and skill is an out-of-time quantity.
2. Under `horizon > 1` the forward windows at the edge of a fold overlap those of the next fold,
   because `purged_size = 0`.
3. A new observation moves every fold edge, so a step of the carry fold of
   `CrossSectionalFactorPrior` cannot fold the member. It must refit every row.

The oracle runs k-fold in its batch fit and predict-then-update in its step. These are two rules,
so its step does not equal its batch fit: on #1567 the calibration coefficient moved by 1.78 times
its own size. The maintainer chose the rule that is "more mathematically correct/justifiable" as
the default.

## Decision

**The default `cv` of `TargetReturnForecast` is `PrequentialCalibration()`.** Under it each matured
observation `t` reads the prediction of the model fitted on the valid samples whose target had
matured at `t`, the observations up to `t - lag - horizon + 1`. That prediction is the forecast the
member publishes at `t`, before its calibration. The batch fit and a step run this one rule.

| `cv` | The predictions that the calibration reads | Folds one row? |
| --- | --- | --- |
| `PrequentialCalibration()`, the default | the forecast of each observation from the targets matured at it | yes |
| a `CrossValidationEstimator`, such as `KFold()` | out of fold, the default before this ADR | no: the carry refits every row, which is exact |
| `nothing` | in sample | no |

**Why it is correct.** No prediction reads a later observation. The forward window of the last
training row, `t - lag - horizon + 1`, ends at `t`, so no training target overlaps the target of
the row it predicts. The rule is the one under which the forecast is used.

**The warm-up.** An observation predicts `NaN` while no valid sample matured before it. This is the
warm-up of the member itself: its forecast is `NaN` until the target of one observation has
matured. The calibration counts its own warm-up in `min_obs`. `warmup` (`NaNWarmup`,
`InSampleWarmup`) still acts under a cross-validation estimator alone.

**The model.** `LinearModel()`, a `LinearModel` with no keyword argument, folds its fit. The member
adds the valid samples of each observation to `X'X` and `X'y` through `normal_equations_add!`, one
sample at a time in the order of the samples, and solves them through a Cholesky factorisation with
pivots. A collinear column takes a zero coefficient, as `GLM.LinearModel` gives it under its
default `dropcollinear = true`. The model of the latest observation is the exported Result
`NormalEquationsFit`, the end of the same fold, which carries `XtX`, `Xty`, `n` and `coef`. A sum
over many ranges, taken one range after the other, is the sum over their union to the last bit,
so a fold that continues from the carried sums equals the batch fit.

Every other regression target, `LinearModel` with keyword arguments among them, fits again on the
matured samples at each observation that adds one. This is exact. It costs one fit for each
observation, where the k-fold rule cost five, so a slow target on a long panel can state
`cv = KFold()`. Under `KFold()` and `nothing` the model stays the fit of the target through
`StatsAPI.fit`, as before.

**Naming.** The type names the rule, `PrequentialCalibration`, after the prequential principle of
statistical forecasting: a forecaster is judged on the forecasts it issued, each from the data
before it. The field keeps its name `cv`, because a cross-validation estimator stays one of its
values.

## Consequences

- Every stored output of a calibrated `TargetReturnForecast` that took the default moves. Each
  oracle case states `cv = KFold()` now, because the oracle calibrates out of fold in its batch
  fit: the cases of `test/test_12z_parity_return_forecasts.jl`, the grid cases `FcTarget` and
  `FcTargetIntercept` of `test/parity_grid.jl`, and the warm-up case of
  `test/test_08x_return_forecasts.jl`. The uncalibrated cases keep the default, so the oracle
  checks `NormalEquationsFit` against its own least squares fit.
- The calibration coefficient can change sign. On the panel of the leverage-one case of
  `test/test_12zb_parity_cs_prior_carry_fold.jl` (1200 rows, 40 assets, two `Passthrough` scores),
  `KFold()` gives 1.874 in the return unit and 0.445 in the Sharpe unit. The prequential rule gives
  -2.020 and -2.069. Out of time, the forecast of those scores has the wrong sign over the
  recent rows that the calibration weighs most. The k-fold rule also trains each fold on the rows
  after it, and its slope is positive.
- `rf.model` of the default is a `NormalEquationsFit`, not a `GLM.LinearModel`. It answers
  `StatsAPI.coef` and `StatsAPI.predict`. Measured on a random design of 1500 samples and 3
  columns, its coefficients equal those of `GLM.LinearModel` to 1.2e-15 relative, and a collinear
  column takes the same zero coefficient.
- `target_forecast_uncalibrated` takes the number of assets as a last argument.
- One function, `return_forecast_step`, runs the rule (#1581). The batch fit runs it over every
  observation from an empty state. A step of the carry fold of `CrossSectionalFactorPrior` runs it
  over the new observations and the observations whose target matures at the step, from the state
  that the carry keeps: the normal equations, the model, the two calibration regressions, and the
  coefficients that predicted the observations whose target has not matured. So the carry keeps
  the panel rows of the look-back of the Descriptors alone, and the step and the batch fit do the
  same arithmetic. Under `whole_history` with `horizon > 1`, a row of the Descriptor warm-up of
  the prior trains the fit, and the carry keeps no such row, so the member refits over every row
  there (`folds_forecast_rows` answers `false`).
- The forecast that the member publishes at an observation is the forecast of a fit through that
  observation. So the carry gives a slot that reads the Return Forecast history the rows of the
  fold, and fits no member again for it.
- [ADR 0186](0186-an-oracle-mode-is-built-when-a-caller-cannot-reach-its-output-and-five-differences-are-deliberate.md)
  states the warm-up under a cross-validation estimator, and
  [ADR 0194](0194-a-fitted-return-forecast-neutralises-its-scores-so-its-orthogonal-part-carries-the-scale-of-its-fit.md)
  reads `κ⊥` off the same predictions as `κ`, prequential by default.
