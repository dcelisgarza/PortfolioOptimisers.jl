---
status: accepted
---

# A Return Forecast Estimator scores the whole carrier, and answers on the block's rows

## Context

A `CrossSectionalFactorPrior` drops the leading observations its Descriptors warm up over, and
fits the factor-model block on the rows that remain. Every history on the block lives on that
axis: `Ms`, `vs`, `rw`, `bw`, `csr.eps` and `rf.hist`. The prior's own result is on it too: `o_X`,
`X` and `pnl` are cut to the fitted rows.

A Return Forecast Estimator has Descriptors of its own, and they warm up too. Since
[#739](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/739) the prior hands the
estimator the carrier restricted to the fitted rows, in
`prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult)`. So the forecast's Descriptors warm up a
second time, inside the fitted window. A panel of 300 rows, a factor model whose Descriptors need
60, and a forecast momentum with a half-life of 60: the momentum sees rows 61 to 300 and is cold
for 60 of them. When the forecast's Descriptors have no warm-up the two designs agree, which is
why the stored parity cases of #739 are unaffected.

The reference implementation does not do this. Its `_compute_alpha`, in
`prior/_characteristics_factor_model.py` at lines 2688 to 2706, pads the four block histories back
onto the whole panel: `NaN` for the exposures, the idiosyncratic returns and the variances, and
zero for the regression weights. It then fits the alpha estimator on the whole panel. Its own note
says why: to maximise data availability and avoid stacking warm-up periods. The padded rows carry
no information of the factor model. They give the alpha estimator's Descriptors more rows to warm
up on. A caller of the reference cannot choose the window: the whole history is the only mode.

**What the padded rows change**, traced through every read of a padded row in the reference's
three estimators:

- The fixed-weighted estimator reads the last row of the alpha only. The padded rows change
  nothing but the Descriptor warm-up.
- The exponentially weighted estimator drops a training row whose variance is not finite, at
  `alpha/_ew_sharpe_optimal_alpha.py:601`. Every padded row is dropped.
- The predictor estimator keeps a training row when its target, its scores and its weight are
  good, at `alpha/_predictor_alpha.py:608`. It does not read the variance in return units. So a
  padded signal row whose forward window reaches into the fitted rows **is** a training row. There
  are at most `lag + horizon - 1` such rows, and only in return units. With a Neutralisation they
  are dropped again, because the reference's neutralisation writes `NaN` where the exposure is
  `NaN`. Its own calibration drops them too, because it reads the variance at the signal row.

Every per-row step is identical on the fitted rows in both designs: the Descriptor values, the
outlier and scoring transforms, the Neutralisation, the composite and the unit conversion. So the
whole history changes the final forecast in exactly two ways: the warm-up of the forecast's
Descriptors, in every case, and that boundary band of training rows for one member in one unit.

Four readers of the family index the block beside the carrier by observation. `neutralise_scores!`
reads `Ms`. `forecast_return_units` reads `vs` under the Sharpe unit and refuses a size that does
not match. The two fitted members of
[#738](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/738) read `csr.eps` for the
forward target and `vs` for the weights. All four assume one axis, and that is what made the
narrow carrier the well-formed call.

## Decision

### The estimator sees the whole carrier

`prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult)` hands `cross_sectional_return_forecast`
the **whole** carrier, not `port_opt_view(rd, rw[r], :)`. A caller who fits a forecast alone on a
stored prior result, through `return_forecast(rfe, rd, pr.rr)`, hands the carrier it fitted the
prior on.

### The block is a suffix of the carrier, found by size

The fitted rows are always the trailing rows of the carrier: the prior drops leading rows and no
others. So the family finds the block's rows by size, as the reference finds its pad. A helper,
`return_forecast_rows(rd, csfm)`, returns the last `Tb` rows of the carrier, where `Tb` is the
observation count of the block's histories, and refuses a carrier with fewer rows than the block
with a `DimensionMismatch`. A carrier with exactly `Tb` rows gives the whole range, which is the
call of #739. No field is added to the block: a field would have to survive every view, and the
size arithmetic already holds on every one.

### The scores are computed on the whole axis and handed with the block's rows

`descriptor_scores(ds, rd, csfm)` computes every Descriptor, the outlier transform and the scoring
transform over the whole carrier, so the Descriptors warm up on every row the panel has. A
Neutralisation regresses the block's rows against `Ms` as it does today, and writes `NaN` on the
rows before the block, where no exposure exists. That is the reference's own answer at those rows.
The verb returns the whole-axis scores together with the block's row range, and each member aligns
once.

### Every Result is on the block's axis

`hist` and `mu` of every Return Forecast Result stay on the block's rows, so `rf.hist` lines up
with `csfm.vs`, `csfm.csr.eps`, `pr.o_X` and `pr.pnl`. The unit conversion runs on the block's rows
only, so `forecast_return_units` keeps its size check.

- `FixedWeightedReturnForecast` cuts the scores to the block's rows and proceeds as today.
- `ExpWeightedReturnForecast` cuts the scores to the block's rows. A row before the block has no
  variance, and the reference drops such a row, so the cut is the same answer with less work.
- `TargetReturnForecast` gains a full-word field, **`whole_history::Bool = true`**. Under `true`
  it places `csr.eps` into a `NaN` matrix of the carrier's length, builds its forward target over
  the whole axis, and trains on every pair with a finite score, a finite target and a positive
  weight. That keeps the boundary band and reproduces the reference in every configuration. Its
  calibration reads the variance at the signal row, as the reference does, so the band enters the
  fit and not the calibration. Under `false` it trains on the block's rows only. The switch is a
  mode the reference does not have, and it removes none.

### What the build corrects

The docstring of `cross_sectional_return_forecast` says the forecast is fitted over the fitted
observations, and the header of `test/test_12k_cross_sectional_factor_prior.jl` records that as
the third departure from the reference. Both are corrected by the build that lands this decision.
A new stored parity case must use a forecast Descriptor **with** a warm-up, because that is the
one case the existing cases cannot see.

## Alternatives rejected

| Option | What the estimator sees | Why it was not taken |
| --- | --- | --- |
| **Cut, and the target member trains on the whole axis by default** | The whole carrier for its Descriptors; the block on the block's rows. | Taken. |
| **Window**: the fitted rows only | The carrier restricted to the block's rows. | The forecast's Descriptors warm up a second time. It does not reproduce the reference when a forecast Descriptor has a warm-up, and it is the one design the reference cannot express. |
| **Pad**: pad the block onto the carrier's axis for the call, as the reference does | The whole carrier and a padded twin of the block. | Exact, and it costs four padded copies per fit, one of them `observations × assets × factors`, and a `NaN`-row rule on every member now and to come. `rf.hist` would be longer than every other history on the block, so every join would need the offset. The library gets the same warm-up with the cut and none of those costs. |
| **Cut with no switch**: cut the scores before any member reads them | The whole carrier for its Descriptors, the block's rows for everything else. | Simplest, and it loses the boundary band for the target member in return units without a Neutralisation, at most `lag + horizon - 1` training rows. The maintainer wants that band available. |
| **Whole-axis block**: store every block history on the carrier's axis | One axis, the carrier's. | Every reader of the block gains a `NaN`-row rule: the nested factor prior on `csr.f`, the lift, the attribution and the diagnostics. The reference does not do this either: its fitted attributes are on the post-warm-up axis, and it pads only for the alpha call. |
| **An offset field on the block** | As taken, with the block's first row recorded on it. | The block is always a suffix, so the size gives the same answer, and a field would have to be carried by every view of the block. |

## Consequences

- A forecast Descriptor with a warm-up now warms up over the whole panel, so the prior reproduces
  the reference in every configuration. The stored parity cases of #737 and #739 use Descriptors
  with no warm-up, and no number in them moves.
- `descriptor_scores` returns two things, the whole-axis scores and the block's rows, so its
  callers change. There are four in the family and none outside it.
- The two fitted members of #738 change in how they read the carrier and where they cut. That work
  runs after #738 lands, so the members change once.
- `TargetReturnForecast` gains one field, and its `arg_dict` key is the one addition to the
  dictionaries.
- `return_forecast(rfe, rd, pr.rr)` on a stored prior result now takes the full carrier, and a
  caller who hands the already-narrowed carrier gets today's answer, because the range is then the
  whole of it.
- A carrier of a different window with the same length as the block is not detected. That is the
  contract every same-axis read in the family already had, and the reference's own.
