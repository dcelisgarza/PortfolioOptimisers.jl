---
status: accepted
---

# A fitted Return Forecast neutralises its scores, so its orthogonal part carries the scale of its fit

## Context

`CrossSectionalFactorPrior` splits its Return Forecast `α` against the latest exposures of the
estimated factors, under the regression weights `rw[T]` of the latest fit:
`α = L g + α⊥`. The spanned part `g` blends into the factor mean under the Spanned Shrinkage
`lambda`, and the orthogonal part `α⊥` enters the mean under the Orthogonal Forecast Scale `c`
([ADR 0095](0095-the-calibration-channel-is-parallel-and-the-rule-is-in-the-bound.md), amendment of
2026-10-05).

Two members of the Return Forecast family fit their forecast: `TargetReturnForecast` and
`ExpWeightedReturnForecast`. Each regresses the forward idiosyncratic return `ε̄` of the block on
its Descriptor scores. The normal equations of the cross-sectional regression make every `ε_t`
orthogonal to the loadings under `rw[t]`. By the Frisch–Waugh theorem, the only coefficient that
such a member can estimate is the one on the part `s⊥` of its score that the loadings do not span.
The coefficient on the whole score is attenuated by `var(s⊥) / var(s)`, so `α⊥` is under-scaled by
that factor, and `g` has no evidence from the fit.

[#1485](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1485) measured it out of
sample, on 500 assets × 2520 observations with one score `s1 = β ℓ + u` and `ε ⟂ L`:

| Member | `β` | Scores neutralised | slope of `ε̄` on `α⊥` | `var(α)/var(α⊥)` |
| --- | ---: | --- | ---: | ---: |
| `TargetReturnForecast` | 1 | no | 2.045 ± 0.089 | 2.016 |
| `ExpWeightedReturnForecast` | 1 | no | 1.869 ± 0.063 | 1.979 |
| `TargetReturnForecast` | 1 | yes | 0.966 ± 0.042 | 1.000 |
| `ExpWeightedReturnForecast` | 1 | yes | 0.953 ± 0.032 | 1.000 |
| control: `TargetReturnForecast` | 0 | no | 0.975 ± 0.043 | 1.002 |

`FixedWeightedReturnForecast` has the same ratio of variances, but it fits nothing, so it has no
fitted scale to repair. The independent implementation that map
[#1375](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1375) measures the library
against reads the fitted member as it stands, and the stored parity cases pin that form.

## Decision

**The prior has a field `ofit`, the Orthogonal Forecast Fit.** It takes one of three singleton
types of `AbstractOrthogonalForecastFit`:

| Rule | What it does |
| --- | --- |
| `ScoreNeutralisation()`, the default | The prior neutralises the scores of a member that answers `true` to `fits_idiosyncratic_target` against every estimated factor of the block, after the Neutralisation names of the caller, under `BlockRegressionWeights()`. |
| `OrthogonalPartCalibration()` | The calibration of `TargetReturnForecast` also fits `κ⊥` on the orthogonal part of each row of its out-of-fold prediction. The member still publishes `α = κ p`, its Result carries `κ⊥` in `ocalib`, and the prior keeps `g` and scales `α⊥` by `κ⊥ / κ`. |
| `UnadjustedForecast()` | The prior reads the member as it stands. This is the form of the independent implementation, one keyword away as [ADR 0186](0186-an-oracle-mode-is-built-when-a-caller-cannot-reach-its-output-and-five-differences-are-deliberate.md) requires. |

**The default output of the prior differs from the independent implementation on purpose.** A
fitted member under the default gives a forecast whose `α⊥` carries the scale of its fit and whose
`g` is about zero. Under a linear member the forecast is orthogonal to the loadings under `rw[T]`
to round-off, so `α⊥ = α`.

**The weights are `rw[t]`.** They are the inner product of the split at `T`, and they read no
future row. `DescriptorScores` has a public field `nw` for the weights of its Neutralisation:
`EstimationMaskWeights()` by default, which keeps parity and every standalone result, or
`BlockRegressionWeights()`. The scoring step after the regression reads the same weights, so a
standardised score stays orthogonal when the loadings span the constant. A standalone caller can
make the same repair.

**Only the fitted members, by a trait.** `fits_idiosyncratic_target` answers `true` for
`TargetReturnForecast` and `ExpWeightedReturnForecast`, and `false` by default.
`FixedWeightedReturnForecast` and `CustomValueReturnForecast` keep the forecast of the caller,
because a spanned part can be the view of a factor that the caller intends. A caller's fitted
type opts in with one method, and holds its `DescriptorScores` in `scores`.

**The observed factors are not neutralised.** The regression does not estimate them, so `ε` is not
orthogonal to their exposures, and the split leaves them out.

**`κ⊥` is paid only where it is asked for.** `κ` lives in the calibration of
`TargetReturnForecast` under `calibrate = true`, so `κ⊥` is resolved there, through a
four-argument method of `return_forecast` that takes the Cross-Sectional Regression Estimator of
the prior. The default rule, the unadjusted form and the standalone member never pay for it. The
calibration reuses the out-of-fold prediction of `κ`, so it runs no second cross-validation. A
scale of the whole prediction by `κ⊥` was refused: it makes `g` about twice as large with no
evidence, and `PrecisionBlend` reads `g`. The constructor of the prior refuses
`OrthogonalPartCalibration()` with a member that answers `false` to `calibrates_orthogonal_part`,
and the message names the default rule and `ForecastCalibrationSlope`, which fits the same slope on
the forecast history. A ratio `κ⊥ / κ` that is not finite is the warm-up of a slope, and it gives
the zero split of a forecast that is not finite.

**The rule holds on every path.** The prior reaches its forecast through
`cross_sectional_return_forecast` alone, which the batch fit, the online refit and the carry fold
([ADR 0193](0193-the-cross-sectional-factor-prior-refits-online-first-and-folds-as-a-host-next.md))
all call. The Return Forecast history that a slot reads is the history of the member that the rule
fits.

**The parity cases state `ofit = UnadjustedForecast()`** wherever the prior holds a fitted member.
The standalone member tests do not change.

## Consequences

- A neutralised score is `NaN` before the block. Under `whole_history = true` a
  `TargetReturnForecast` then trains only on the last `lag + horizon - 1` rows before the block,
  the rows whose forward window reaches into the block.
- A rule in `c` that fits the slope on the history of a neutralised member measures about one. Under
  `OrthogonalPartCalibration()` the history is the history of the member as it stands, which `κ⊥`
  does not scale, so such a rule repairs the under-scale a second time.
- The constructors of `DescriptorScores`, `TargetReturnForecastResult` and
  `CrossSectionalFactorPrior` each take one more positional argument. A method that binds a type
  parameter of the prior by its position counts the new one.

## Alternatives refused

- **Leave the member as it stands and let `c` repair it.** `ForecastCalibrationSlope` repairs the
  scale, but it needs the forecast history, which costs 11 to 14 minutes at 500 assets × 2520
  observations for `TargetReturnForecast`. A repair inside the member costs nothing.
- **Neutralise under the estimation mask.** It is not the inner product of the split, so it repairs
  the scale only in part when `rw` is a power of the capitalisation.
- **Change the standalone member.** It stands outside a prior, and it keeps parity with the
  independent implementation (#1386).
