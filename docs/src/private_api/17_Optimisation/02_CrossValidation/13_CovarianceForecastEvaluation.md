```@meta
Description = "The covariance forecast evaluation, private API of PortfolioOptimisers.jl: AbstractRealisedTarget, realised_target, target_dof, target_step_dof, …"
```

# The covariance forecast evaluation: private API

```@docs
PortfolioOptimisers.AbstractRealisedTarget
PortfolioOptimisers.realised_target
PortfolioOptimisers.target_dof
PortfolioOptimisers.target_step_dof
PortfolioOptimisers.forecast_location
PortfolioOptimisers.forecast_state_location
PortfolioOptimisers.forecast_coverage_policy
PortfolioOptimisers.finite_column_mean
PortfolioOptimisers.resolve_forecast_weights
PortfolioOptimisers.is_time_dependent(::Union{<:PortfolioOptimisers.AbstractCovarianceEstimator, <:PortfolioOptimisers.AbstractPriorEstimator, <:Online})
PortfolioOptimisers.online_entry_state(o::Online)
PortfolioOptimisers.advance_previous_fold(::Any, prev, ::NamedTuple)
PortfolioOptimisers.forecast_moments
```
