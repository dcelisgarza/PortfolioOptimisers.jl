```@meta
Description = "Covariance forecast summary, public API of PortfolioOptimisers.jl: AbstractStepWeighting, DofStepWeighting, EqualStepWeighting, covariance_step_weights, …"
```

# Covariance forecast summary

Three functions read a [`CovarianceForecastEvaluationResult`](@ref). [`covariance_forecast_summary`](@ref) returns a result with one entry per evaluation in each column, so you can put two forecasts side by side. [`covariance_forecast_compare`](@ref) tests the difference between the losses of two forecasts at each step, with a Diebold-Mariano-West statistic. The loss compares a forecast with a proxy of the true covariance, so its level alone does not say how good a forecast is, but the difference between two forecasts does. [`covariance_forecast_portfolio`](@ref) scores the stored forecasts again on a new test portfolio.

The summary and the calibration figure weight each step in the mean of a ratio by an [`AbstractStepWeighting`](@ref). [`DofStepWeighting`](@ref), the default, weights a step by its degrees of freedom, which gives the mean of least variance. [`EqualStepWeighting`](@ref) gives the plain mean over the steps.

```@docs
AbstractStepWeighting
DofStepWeighting
EqualStepWeighting
covariance_step_weights
CovarianceForecastSummaryResult
CovarianceForecastComparisonResult
covariance_forecast_summary
covariance_forecast_compare
covariance_forecast_portfolio
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
