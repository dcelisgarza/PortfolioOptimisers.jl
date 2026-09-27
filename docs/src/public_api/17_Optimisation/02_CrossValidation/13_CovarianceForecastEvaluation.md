```@meta
Description = "The covariance forecast evaluation, public API of PortfolioOptimisers.jl: RealisedCovariance, HorizonReturn, CovarianceForecastEvaluationResult, …"
```

# The covariance forecast evaluation

[`covariance_forecast_evaluation`](@ref) scores a covariance forecast against the returns that come after it, one step at a time over a walk-forward. It runs through [`PortfolioOptimisers.fold_loop`](@ref). A fold with a training window refits the estimator on that window, and a fold of an online walk-forward updates the estimator and reads its current estimate. The choice of walk-forward and of [`Online`](@ref) gives the four forms, an expanding or a rolling window, fitted in batch or online. A realised-quantity type, one of the types below, sets what the forecast is scored against. The evaluation subtracts from each test return the centre that the estimator used, such as the mean it fitted, and the loss of each step is a function of plain arrays.

```@docs
RealisedCovariance
HorizonReturn
CovarianceForecastEvaluationResult
covariance_forecast_step
PortfolioOptimisers.partial_fit!(ce::PortfolioOptimisers.AbstractCovarianceEstimator, rd::ReturnsResult)
covariance_forecast_evaluation
PortfolioOptimisers.AbstractRealisedTarget
PortfolioOptimisers.realised_target
PortfolioOptimisers.target_dof
PortfolioOptimisers.target_step_dof
```
