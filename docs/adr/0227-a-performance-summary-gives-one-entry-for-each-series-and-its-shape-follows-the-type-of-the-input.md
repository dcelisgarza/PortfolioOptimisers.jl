---
status: accepted
---

# A Performance Summary gives one entry for each series, and its shape follows the type of the input

## Context

`performance_summary` summarised one realised return series with twelve fixed figures. It took a
bare series, weights with returns, an `OptimisationResult` with a `ReturnsResult`, a
`PredictionResult` and a `MultiPeriodPredictionResult`. Issue
[#1934](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1934) found five things that
a caller needs from a summary and could not get:

1. A risk measure of the choice of the caller, such as a CVaR at another level or an Ulcer Index.
2. A population of paths. The spread of a figure over the paths of a combinatorial or a
   multiple-randomised scheme is the reason to run such a scheme.
3. Many portfolios side by side, with no hand loop and no hand table.
4. The members of a prediction whose optimiser gives many portfolios (a frontier sweep, ADR 0062).
   The summary read the first member and dropped the others with no warning.
5. A risk-free rate, and a figure of concentration.

The library has two other columnar summaries, `ForecastSummaryResult` and
`CovarianceForecastSummaryResult`. Each gave length-1 vectors for one item.

## Decision

**1. One verb, one type, and the shape follows the type of the input.** `performance_summary`
stays the one verb, and `PerformanceSummaryResult` stays the one type. The type is `@concrete`,
so a field holds a scalar or a vector with no second definition. An input that holds exactly one
series gives scalar fields: a bare `VecNum`, weights with returns, one `OptimisationResult` with a
`ReturnsResult`, and a `PredictionResult` or `MultiPeriodPredictionResult` whose `X` is a `VecNum`.
Every other input gives one vector entry for each series: a vector of inputs, a
`PopulationPredictionResult`, a prediction whose `X` is a `VecVecNum`, and weights in a
`VecVecNum`, which an `OptimisationResult` of a frontier holds. The shape follows the type and
never the count at run time, so a vector of one input gives vectors of length one, and each
method with a concrete input is inferred.

**2. Each entry is one series, and three keys name it.** `names` is the candidate, from the
`names` keyword of a vector method, with the default `"Portfolio k"` of `forecast_summary_names`.
`paths` is the `id` of a path of a population, and the position of a single-fold prediction in a
population, which has no `id`. `members` is the index of a member. A key that does not apply is
`nothing`. The entries are in the order candidate, path, member: one axis with three keys, in long
form. Each entry also carries `n_periods`, `date0` and `date1`, the first and the last timestamp of
its series, or `nothing`. The names `date0` and `date1` follow `brinson_attribution`.

**3. Entries on different rows are permitted.** No figure pairs two series, so nothing is wrong
to compute, and `n_periods`, `date0` and `date1` show the difference. A forecast summary pairs its
forecasts, so it refuses evaluations on different dates. A Performance Summary does not.

**4. The extra risk measures are keyed by name.** The keyword `measures` takes pairs
`name::Symbol => r::AbstractBaseRiskMeasure`. The result holds `measures` and `risk`, two
`NamedTuple`s by name. Each value is `expected_risk_from_returns(r, ret)` on the same net series
as the headline figures, with the sign of the measure, and `bigger_is_better(r)` gives its
orientation. A measure whose `supports_precomputed_returns` is `false` is refused at entry, before
any figure is computed, and the error names each refused measure by its key and its type, and names
`expected_risk` as the route for it. A repeated name, and a name that is a column of the table,
are refused.

**5. A population gives one entry for each path, and the summary does no reduction.** Each path
is one series, so each figure of a path is a point estimate, and the column of a figure is its
distribution over the paths. A caller reduces it with `median(ps.sharpe)` or
`DataFrames.describe(DataFrame(ps))`.

**6. A many-member prediction gives one entry for each member.** Each member reads its own series,
its own benchmark series, its own turnover through `held_path_turnover(pred, j)` and its own
weights. The summary reads the members through a reader of its own, `prediction_entries`.
`regret_series` gives the series as the prediction stores them, and the summary does not read its
first member alone.

**7. The risk-free rate is for each period.** `rf` is at the frequency of the returns, as in
`probabilistic_sharpe_ratio`, `expected_ratio` and `MeanReturnRiskRatio`. It enters `sharpe`,
`sharpe_stderr` and `sortino`, and no other figure. At `rf = 0` no released number moves, because
`m - 0` is `m` exactly. `rf` must be finite.

**8. `n_eff` is the concentration figure.** Weights, and one `OptimisationResult` on its
investable view, give `number_effective_assets` of their weights. A weight path gives the mean over
its rows. A path gives the mean over its folds of the value of each fold target, and a single
prediction the value of its target. A bare series gives `NaN`. The risk contribution of each asset
is not a column, because its axis is the asset, as ADR 0149 leaves three parts out of the forecast
summary for the same reason.

**9. The table.** `DataFrames.DataFrame(ps)` gives one row for each entry, with the columns
`:name`, `:path`, `:member`, `:date0`, `:date1`, `:n_periods`, the twelve released figures in the
order of the struct, `:n_eff`, and one column for each extra measure, in the order of `measures`.
`ppy`, `alpha`, `compound` and `rf` are one value for the whole summary, so they are table-level
metadata.

**10. The two other columnar summaries take the same rule.** `forecast_evaluation_summary(fe)` and
`covariance_forecast_summary(cfer)` give scalar fields, equal to entry 1 of the vector method on
`[fe]` and `[cfer]`. Each computes the vector of one item and reads entry 1 of each column, so each
figure keeps one site. A column whose second axis is the tail fraction or the level becomes a
vector over that axis. `factor_model_summary` summarises one model over a factor axis, so the rule
does not reach it. ADR 0149 takes an amendment.

**11. The plot.** `plot_performance_summary` draws one entry as one bar chart, as before, with one
bar more for each extra measure. It draws many entries as one panel for each figure, with one bar
for each entry, labelled by the keys that are not `nothing`. `plot_forecast_evaluation_summary`
draws a summary with scalar fields.

## Rejected

- **A second summary verb beside `performance_summary`.** Two lists of statistics would overlap.
- **A length-1 vector for one item.** It breaks the common call for one portfolio, and the type of
  the input already tells the shape.
- **A reduction keyword for a population.** The reduction of a ratio is not the ratio of a
  reduction, so the choice is the caller's.
- **A refusal of entries on different rows.** No figure pairs two series.
- **Names taken from the type of a measure.** Two CVaR levels have one type.
- **The mean weight vector for `n_eff` of a path.** When the folds hold different assets, it gives
  a larger number than any fold held.

## Consequences

- `src/18_ExpectedReturns.jl` went over the size threshold, so it became the directory
  `src/18_ExpectedReturns/` by subject, as ADR 0179 states: `01_ExpectedReturns`,
  `02_BrinsonAttribution`, `03_PerformanceSummary` and `04_SharpeRatioInference`.
- `PerformanceSummaryResult` gained nine fields, so a positional call of the old sixteen arguments
  fails. No released field of a one-series input changes value at the defaults, apart from `n_eff`,
  which is new.
- A many-member prediction, which gave scalar fields of its first member, now gives one vector
  entry for each member.
- A summary of one forecast evaluation, and of one covariance forecast evaluation, holds scalars
  where it held length-1 vectors.
- `GLOSSARY.md` defines **Performance Summary**.
