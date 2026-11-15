```@meta
Description = "Covariance Forecast Evaluation, private API of PortfolioOptimisers.jl: forecast_location, prior_forecast_location, prior_state_location, carried_location, …"
```

# Covariance Forecast Evaluation: private API

```@docs
PortfolioOptimisers.forecast_location
PortfolioOptimisers.prior_forecast_location
PortfolioOptimisers.prior_state_location
PortfolioOptimisers.carried_location
PortfolioOptimisers.horizon_location
PortfolioOptimisers.finite_column_mean
PortfolioOptimisers.resolve_forecast_weights
PortfolioOptimisers.step_implied_volatility_kwargs
PortfolioOptimisers.is_time_dependent(::Union{PortfolioOptimisers.AbstractCovarianceEstimator, PortfolioOptimisers.AbstractPriorEstimator, Online})
PortfolioOptimisers.online_entry_state(o::Online)
PortfolioOptimisers.advance_previous_fold(::Any, prev, ::Tuple{CovarianceForecastStepResult, Any, Any})
PortfolioOptimisers.forecast_moments
```
