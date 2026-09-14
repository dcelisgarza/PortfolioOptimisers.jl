# The summary, comparison and re-projection of a covariance forecast evaluation

Three verbs above [`CovarianceForecastEvaluationResult`](@ref). [`covariance_forecast_summary`](@ref) answers a columnar Result with one entry per evaluation, so a side-by-side of two forecasts is its length-2 case; [`covariance_forecast_compare`](@ref) tests the per-step loss difference of two forecasts with a Diebold–Mariano–West statistic, because the level of a loss on a proxy is not a reading and its difference is; and [`covariance_forecast_portfolio`](@ref) re-projects stored forecasts on a new test portfolio.

```@docs
CovarianceForecastSummaryResult
covariance_forecast_summary
CovarianceForecastComparisonResult
covariance_forecast_compare
PortfolioOptimisers.newey_west_variance
covariance_forecast_portfolio
```
