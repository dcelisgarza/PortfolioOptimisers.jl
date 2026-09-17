```@meta
Description = "The summary, comparison and re-projection of a covariance forecast evaluation, public API of PortfolioOptimisers.jl: CovarianceForecastSummaryResult, …"
```

# The summary, comparison and re-projection of a covariance forecast evaluation

Three verbs above [`CovarianceForecastEvaluationResult`](@ref). [`covariance_forecast_summary`](@ref) answers a columnar Result with one entry per evaluation, so a side-by-side of two forecasts is its length-2 case; [`covariance_forecast_compare`](@ref) tests the per-step loss difference of two forecasts with a Diebold–Mariano–West statistic, because the level of a loss on a proxy is not a reading and its difference is; and [`covariance_forecast_portfolio`](@ref) re-projects stored forecasts on a new test portfolio.

```@docs
CovarianceForecastSummaryResult
CovarianceForecastComparisonResult
covariance_forecast_summary
covariance_forecast_compare
covariance_forecast_portfolio
```
