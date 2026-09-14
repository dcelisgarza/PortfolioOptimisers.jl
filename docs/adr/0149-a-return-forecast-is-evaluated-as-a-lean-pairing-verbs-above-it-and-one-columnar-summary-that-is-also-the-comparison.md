---
status: accepted
---

# A Return Forecast is evaluated as a lean pairing, verbs above it, and one columnar summary that is also the comparison

## Context

[Map #931](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/931) built the
out-of-sample evaluation of a **Return Forecast**: the reading a caller performs *before* an
optimiser sees the forecast, which asks whether the forecast ranks the cross-section, whether the
ranking pays as a book, whether the magnitude is right, how long the edge lasts, and what the
forecast is made of. The map closed on 2026-09-09 with its destination met, through eleven tickets
([#932](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/932) to
[#942](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/942) and
[#965](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/965)), and recorded its
decisions in the tickets' resolution comments alone. This ADR states them in one place, as the
sibling maps of the same release did, so that the code is checked against a document rather than
against a thread. It was written by the PR 625 review of the map (piece Q10, 2026-09-14) and
adds no decision the tickets did not take.

The reference implementation ships the same reading as one entry point of fifteen parameters that
fits the estimator, computes every statistic eagerly, and answers two Result classes — an
evaluation and a comparison — over about twenty-five private helpers, with a free-form parameter
bag and a name on each. The library already held three of its pieces before the map opened:
`forward_mean_returns` (the forward window), the cross-sectional correlation kernels
(`cs_spearman_correlation`, `cs_weighted_correlation`) and the exposure diagnostics
(`exposure_ic`, `exposure_ic_summary`, `exposure_coverage`) of
[map #643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643). Two of the four
shipped Return Forecast members publish a history (`hist`) and two do not.

## Decision

### The evaluation is a three-layer hierarchy, bare arrays first

`forecast_evaluation` takes two matrices, then a fitted Result with a carrier and a block, then an
Estimator with the same. The bare layer makes every statistic testable without a fit and scores a
forecast the library did not produce; the Result layer reads `hist` off a fitted member; the
Estimator layer asks `forecast_history` for one. The two carrier-taking layers share one pairing,
`forecast_evaluation_pairing`, which builds the target once and refuses a carrier the forecast was
not fitted on.

### The Result is lean, and every statistic is a verb over it

`ForecastEvaluationResult` carries the forecast history `alpha`, the **Forward Target** `y`, the
universe mask `umsk`, the evaluation `dates`, and the parameters that produced them (`target`,
`horizon`, `lag`, `step`, `min_count`, `ppy`). It computes no statistic. `forecast_ic`,
`forecast_coverage`, `forecast_portfolio`, `forecast_quantile_spread`, `forecast_calibration`,
`forecast_factor_correlation`, `forecast_holding_period` and `forecast_decay` are verbs over it,
each re-parameterisable without a re-pairing; `min_count` and `ppy` are *carried* by the pairing
and *applied* by the verbs. The reason is cost: producing `alpha` can be a rolling refit, so the
pairing is computed once and everything above it — the plots included — reads that pairing rather
than the block. This is the one place the map departs from the shape of map #643, whose plots take
the block and call the verb themselves because every verb there reads a block that is already
fitted.

The universe is carried for the same reason the pairing is. `forecast_coverage` divides the
assets that carry a finite pair at a date by the estimation universe of that date, which is the
reference's denominator (`n_valid / eligible_count`), and the coverage is read at the Result
layer, where no carrier is in hand. So the pairing cuts the Asset Panel's estimation mask to the
block's rows once, into `umsk`, and every consumer — the coverage, the summary's four coverage
columns, the window tables' `mean_coverage` — reads that one universe; the bare method takes it as
a keyword and defaults it to every asset, and the summary refuses two evaluations whose masks
differ. On a point-in-time panel the mask moves with the listings, so an asset that has not
listed yet, or has delisted, is outside the universe rather than a missed one, and a share of one
means every asset the panel admitted was scored, however few. The count the summary reports
beside the share is what says how few. The alternative — the block-taking method of the coverage
intersecting the weight history with the mask, and the bare and weight-taking methods keeping
every asset — was rejected because it leaves the Result-layer answer wrong by default and grows an
`rd` on every consumer (#1070).

### The evaluation dates bound the sample; they do not filter it

An observation is scorable when some asset carries a finite forecast and a finite target there.
The first and the last such observation bound the evaluation, the dates run between them in
strides of `step`, and a stride of the horizon (the default) gives forward windows that do not
overlap. An unscorable observation inside the bounds is **kept**, with `NaN` statistics, so that
`step` means the same thing everywhere in the sample. The reference drops such a date, and no
statistic reads the difference: every summary of a per-date series — the coefficients through
`exposure_ic_factor_summary`, the books and spreads through `forecast_hit_rate` — reads a `NaN`
date as one at which nothing was measured, so the kept date is in no denominator, and how often a
date was silenced is the coverage's question.

### The forecast is written onto the universe at the pairing

The reference masks the forecast by the panel's estimation mask before every statistic. A member
of this family standardises its Descriptors over the estimation universe but writes a score for
every asset whose fields are finite, so an active asset the estimate never reads carried a finite
forecast and entered every statistic here and none there; on the planted fixture of `test_08x`
that is one asset per row, and it moves a factor correlation by up to `0.098` in a cell. The rule
is applied **once**, on the universe the Result already carries: the bare `forecast_evaluation`
writes `NaN` into `alpha` off `umsk` through `forecast_evaluation_mask` and finds the dates on the
masked forecast, so every verb above the pairing reads `fe.alpha` and inherits the universe from
it, and no verb carries a mask of its own. The coverage is the one verb that reads `umsk` itself,
as its denominator. A mask that admits every asset — the bare method's default — hands the
history back as it is, so the pairing carries a caller's matrix rather than a copy; any other mask
answers a copy, and the caller's history is not written to. Issue #1074 measured the divergence,
and this paragraph is its ruling.

### A member that publishes no history is refit along the evaluation grid

`forecast_history` fits the member once and reads its own Result: a `hist` that is given is
returned, and one that is `nothing` is built by refitting the member at every row of a grid
anchored at the block's first observation and striding by `step`. Which path runs is read off the
Result, not off the type, so a member added later needs no method. A refit at observation `tb`
sees the carrier through the row the block's `tb` sits on and the block through its own row `tb`,
cut by `forecast_history_block`, so it reads nothing after the observation it answers for.
`CustomValueReturnForecast` states one cross-section rather than fitting one, so no refit can give
it a path; it is the one permanent refusal, and a caller pairs its stated values through the bare
method. The Estimator layer passes the evaluation's own `step` down, which is what makes every
evaluation date a row that was fitted.

### The Forward Target is a typed family, and the idiosyncratic return is the default

`AbstractForecastTarget` names which history the forward window is taken over:
`IdiosyncraticTarget` (the default — the component a fitted member forecasts inside
`CrossSectionalFactorPrior`), `AssetReturnTarget`, and `PanelFieldTarget(name)`.
`forecast_target_history` is the family's one seam and the one place the two observation axes are
reconciled: the idiosyncratic history lives on the block's rows, the asset returns and the Panel
Fields on the carrier's, and the block is a suffix of the carrier. The horizon and the lag are the
evaluation's parameters, not the target's. The Forecast Unit is not an evaluation parameter,
because every member answers in return units.

### The portfolios reuse `performance_summary`; the spreads do not

`forecast_portfolio` builds the rank-weighted (`:rank`) or z-score-weighted (`:zscore`) book from
the forecast alone — centred and rescaled to 200 % gross by `forecast_centred_weights!`, so every
date is dollar neutral whatever the spread of the forecast — and its return series *is* a
portfolio, so it earns the whole of `performance_summary` (a Sharpe ratio and its standard error,
Sortino, Calmar, the maximum drawdown and the CVaR), which the reference does not report. The gaps
below `min_count` are dropped before that call, as its Precomputed-returns contract prescribes, so
**`max_drawdown` and `calmar` are of the compressed path**; the docstring carries the caveat and no
guard is applied. A quantile spread is a difference of two means and not a book, so it gets the
three annualisation figures and a hit rate through `forecast_series_summary`. The turnover of the
book is `calc_turnover`, which landed in `src/17_NetReturnsDrawdowns.jl` beside `calc_net_returns`
because a caller who holds a weight path wants its turnover whether a forecast produced it or not;
its first observation is `NaN`, because a path cannot say whether it opened from cash.

### The calibration is pooled, has no intercept, and reads no threshold

`forecast_calibration` is the one reading of a forecast that a rescaling moves. The slope is a
weighted regression of the target on the forecast through the origin, pooled over every scorable
pair of every evaluation date; the intercept is refused rather than fitted, because the
cross-sectional mean of the target is what the factor model is for. The curve and the pooled
moments read every pair alike; the weights reach the slope alone. There is no cross-sectional
count to threshold, so the verb takes no `min_count`.

### A contemporaneous statistic is read on every observation, and the grid is opt-in

`forecast_factor_correlation` correlates the cross-section of the forecast against the
cross-section of each exposure **at the same observation**. Nothing in it looks forward, so it has
no window to mature and nothing to keep disjoint: every observation at which the forecast and an
exposure are both written is a sample of it, and the evaluation grid — which exists to keep the
coefficient's forward windows from overlapping — has no meaning for it. The verb therefore reads
the whole observation axis of `fe.alpha` by default, which is what the reference reads, and it is
bit-exact with the reference's kernel and summary on that axis (`test_08x` pins it). A caller who
wants the correlations beside the coefficients of the same dates passes `dates = fe.dates`, and
any other row set is read on the same terms; the row set is a value, not a flag, so the grid is
one spelling of it rather than a second shape. The rows are not filtered: a member that is refit
along the grid carries `NaN` off it, and the whole axis answers `NaN` on every row it was never
asked for, which every figure of the summary reads as an unmeasured row rather than a miss, so
the whole-axis summary of such a member is the summary of the rows it wrote. Under the default
`step = horizon` the grid holds `1 / horizon` of the observations, so the old reading carried a
t-statistic smaller by the root of that ratio for the same forecast; issue #1071 measured it and
this section is its ruling.

### A forward-window table is read on the common dates of its whole grid

`forecast_holding_period` (cumulative windows, `p · h` ahead) and `forecast_decay` (disjoint
windows, `h` ahead at lag `l + (p − 1) h`) rebuild a `ForecastEvaluationResult` per window and read
its row with the coefficient, book and coverage verbs, so no statistic is written twice. A deeper
window matures later, so the dates are intersected across the **whole grid** before any statistic
is taken: a fall down a column is the forecast decaying, not the sample changing under it. The
consequence is that a table read at one depth `n` is internally comparable and is **not**
comparable to a table read at another, and that shortening `n` does not leave the rows that remain.
The table's book columns are annualised at `fe.ppy`, where the reference reports them per period.

### One columnar summary is also the comparison

`forecast_evaluation_summary` answers `ForecastSummaryResult`, thirty columns whose axis is the
forecast — the ten coefficient figures, the five annualised figures of each book, the six
calibration figures and the four coverage figures, plus the quantile block on request — so a single
evaluation is its length-1 case and the length-2 case *is* the comparison. The reference's
comparison class is not carried. The vector method refuses evaluations that are not comparable (a
different target, horizon, lag, step, threshold, annualisation, date set or asset axis), which the
reference does not check. The column names are those of the forward-window tables, so a row of a
table and a row of a summary read on the same terms. Three parts of an evaluation are deliberately
**not** columns, each because its axis is not the forecast: the drawdown family (of the compressed
path), the forward-window tables (axis: the window) and the factor correlations (axis: the factor).

The five hit rates of the summary are taken against **one denominator**, the dates that scored.
Until 2026-09-14 the coefficient hit rates counted against every date, on the reading that a date
the forecast could not rank is a miss, and the book and spread hit rates against the dates that
scored, and the two were kept apart and named apart because `exposure_ic_factor_summary` was
thought to be released surface of map #643. It is not on `main`, and the reading was the odd one
out: it disagreed with the mean, the ratio and the t-statistic printed beside it (over the finite
dates), with `forecast_hit_rate`, with every pairwise kernel of the library, which reads a `NaN`
as unobserved, and with the reference's evaluation `_hit_rate`, which drops it (its exposure
summary counts the `NaN` as a miss, through a `nanmean` of `ic > 0` whose comparison has already
turned the `NaN` into a `false`, and map #643's oracle in `test_08s` records that as a deliberate
divergence now); and once the coverage
divides by the universe (#1070) and the forecast is masked onto it (#1074), the silenced date is
already reported by the coverage, so a hit rate that counted it too counted the same silence
twice. `exposure_ic_factor_summary` now drops a `NaN` from its hit rate, so the five figures of
every summary of the library sit over the same sample, and the caveat that a refit member's
whole-axis factor correlation had to be summarised on its grid is gone with it.

### Every figure takes the Result, never the block

The eleven figures of the StatsPlots extension take a `ForecastEvaluationResult` (or the summary),
because the pairing may have cost a refit and every figure must read the same one. The summary
figure draws ten of the thirty columns, leaving out the counts that would dwarf a ratio on one
axis, and draws the quantile block only when it was asked for.

### Two parameters of the reference have a documented absence

Its free-form parameter bag has no home: every parameter is a typed field and a `@concrete` Result
prints them. Its per-evaluation `name` is not a parameter of the evaluation: the names axis lives
on `ForecastSummaryResult` and is supplied to `forecast_evaluation_summary`. Every other one of its
fifteen parameters has a home, tabulated in #932's resolution.

### What was rejected

- **A comparison class of its own.** Collapsed into the columnar summary (above).
- **Recomputing per plot, as map #643 does.** Refused because `alpha` may cost a rolling refit.
- **A `min_count` keyword on the summary.** It would have reached the coefficients only, because
  the book and the spread read the threshold off the Result, so a row would print under two
  thresholds. The summary reads `min_count` and `ppy` off the Result.
- **A Forecast Unit parameter on the evaluation.** The unit is a fitting detail.
- **Reconciling the two hit-rate denominators.** It would move map #643's released surface.
- **A stored oracle for the books.** They are pinned by their invariants (dollar neutral, 200 %
  gross, a perfect forecast earns a positive mean and its negation the exact opposite), and the
  coefficients, calibration, tables and summary are pinned bit for bit against the reference's own
  methods on two small matrices.

## Consequences

Seventeen names are exported from `src/08_Moments/45_ReturnForecasts/07`–`14`: the three targets,
the two Results, `forecast_evaluation`, `forecast_history`, and the ten level-2 verbs and
summaries; eleven `plot_forecast_*` figures from the extension. `AbstractForecastTarget` is not
exported. `CONTEXT.md` defines **Forecast Evaluation**, **Forward Target** and **Forecast
Calibration**. Example `7_putting_it_together/07_Forecast_Evaluation.jl` is the page that walks the
reading order.

Two members rarely share an evaluation grid — a member that is refit warms up, one that publishes
its history does not — so the summary refuses them, correctly, and aligning two forecasts is today
a hand-written blank through the bare method. The map named it as a fresh effort;
[#1073](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1073) seeds it.

The review that wrote this ADR found three places where the code at the head read less than the
reference and the tickets did not say so, and all three are paid. `forecast_coverage` divided by
every asset on the panel rather than by the estimation mask's count at the date
([#1070](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1070)), and the Result now
carries the universe, as the decision above states. `forecast_factor_correlation` read the
evaluation dates only, where a contemporaneous statistic can read every observation
([#1071](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1071)), and it now reads
them all, as the decision above states; its parity run against the reference measured a fourth
place, the forecast entering every statistic off the estimation mask
([#1074](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1074)), and the pairing now
writes the forecast onto the universe. The reference's two comparison overlays — several
forecasts' cumulative coefficient and cumulative book return on one axis — had no vector method
([#1072](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1072)), and its own commit
paid it.

A Neutralisation does not decorrelate a forecast from its target, because both Neutralisation
sites fit a cross-sectional regression with no intercept; that is
[#950](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/950)'s, settled by letting a
caller override the regression estimator, and the factor-correlation figure states it.
