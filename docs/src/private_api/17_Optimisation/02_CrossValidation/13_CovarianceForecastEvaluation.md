```@meta
Description = "The covariance forecast evaluation, private API of PortfolioOptimisers.jl: forecast_location, forecast_state_location, forecast_coverage_policy, …"
```

# The covariance forecast evaluation: private API

```@docs
PortfolioOptimisers.forecast_location
PortfolioOptimisers.forecast_state_location
PortfolioOptimisers.prior_forecast_location
PortfolioOptimisers.prior_state_location
PortfolioOptimisers.carried_location
PortfolioOptimisers.horizon_location
PortfolioOptimisers.forecast_coverage_policy
PortfolioOptimisers.finite_column_mean
PortfolioOptimisers.resolve_forecast_weights
PortfolioOptimisers.is_time_dependent(::Union{<:PortfolioOptimisers.AbstractCovarianceEstimator, <:PortfolioOptimisers.AbstractPriorEstimator, <:Online})
PortfolioOptimisers.online_entry_state(o::Online)
PortfolioOptimisers.advance_previous_fold(::Any, prev, ::NamedTuple)
PortfolioOptimisers.forecast_moments
```
