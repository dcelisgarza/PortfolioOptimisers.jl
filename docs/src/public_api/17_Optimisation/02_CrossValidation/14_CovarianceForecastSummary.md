```@meta
Description = "The summary, comparison and re-projection of a covariance forecast evaluation, public API of PortfolioOptimisers.jl: CovarianceForecastSummaryResult, …"
```

# The summary, comparison and re-projection of a covariance forecast evaluation

Three functions read a [`CovarianceForecastEvaluationResult`](@ref). [`covariance_forecast_summary`](@ref) returns a result with one entry per evaluation in each column, so you can put two forecasts side by side. [`covariance_forecast_compare`](@ref) tests the difference between the losses of two forecasts at each step, with a Diebold-Mariano-West statistic. The loss compares a forecast with a proxy of the true covariance, so its level alone does not say how good a forecast is, but the difference between two forecasts does. [`covariance_forecast_portfolio`](@ref) scores the stored forecasts again on a new test portfolio.

```@docs
CovarianceForecastSummaryResult
CovarianceForecastComparisonResult
covariance_forecast_summary
covariance_forecast_compare
covariance_forecast_portfolio
```
