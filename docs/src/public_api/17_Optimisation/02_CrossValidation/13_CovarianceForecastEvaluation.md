```@meta
Description = "The covariance forecast evaluation, public API of PortfolioOptimisers.jl: RealisedCovariance, HorizonReturn, CovarianceForecastEvaluationResult, …"
```

# The covariance forecast evaluation

A covariance forecast is judged on the returns realised after it, step by step over a walk-forward. [`covariance_forecast_evaluation`](@ref) runs one verb through [`PortfolioOptimisers.fold_loop`](@ref): a fold that carries a training window refits the estimator over it, and a fold of the online arm reads the threaded state out, so the batch expanding, batch rolling, online expanding and online rolling forms are the four compositions the walk-forward and [`Online`](@ref) already express. The realised quantity is a typed family, the test rows are centred on the location the forecast is about, and the per-step kernel is a level-1 verb on bare arrays. ADR 0143 records the decision.

```@docs
RealisedCovariance
HorizonReturn
CovarianceForecastEvaluationResult
covariance_forecast_step
PortfolioOptimisers.partial_fit!(ce::PortfolioOptimisers.AbstractCovarianceEstimator, rd::ReturnsResult)
covariance_forecast_evaluation
```
