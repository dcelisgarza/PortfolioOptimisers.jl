```@meta
Description = "A performance summary and the series plots of weights or of a result read a gap in the returns as predict does."
```

# [Summaries and plots of weights read a gap as a prediction does](@id migration-next-summary-gap-filter)

## [Results that move](@id migration-next-summary-gap-filter-numbers)

- **`performance_summary` of weights or of an optimisation result reads a gap in the returns as `predict` does.** The previous release multiplied the returns by the weights as they are, and `0 * NaN` is `NaN`. So one asset with a gap made the figures `NaN`, also an asset with a zero weight, and the `cvar` was a finite wrong number. Each return that is not finite is now zero. A gap at a zero weight gives no message. A gap at a weight that is not zero gives a warning that names the asset, and an `ArgumentError` under the new keyword `strict = true`. So `performance_summary(res, rd)` gives the figures of `performance_summary(predict(res, rd))`, except the three excess statistics, which only the prediction takes from the benchmark of `rd`. The figures of returns with no gap do not move. `plot_performance_summary` takes the same keyword.
- **`plot_portfolio_cumulative_returns`, `plot_drawdowns`, `plot_histogram`, `plot_rolling_drawdowns`, `plot_rolling_measure` and `plot_benchmark` of weights or of an optimisation result** read a gap in the same way, with a warning for a gap at a weight that is not zero, or an `ArgumentError` under the new keyword `strict = true`. `plot_portfolio_dashboard` reads a gap in the same way in its panels 2, 3 and 4, and takes the same keyword. `plot_histogram` dropped each row with a gap, and now it keeps the row. `plot_portfolio_cumulative_returns` of weights on a prior with a non-investable asset draws a finite curve. `plot_asset_cumulative_returns` does not change.
