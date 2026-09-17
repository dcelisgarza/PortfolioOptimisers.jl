```@meta
Description = "Exponentially Weighted Covariance, private API of PortfolioOptimisers.jl: ExpWeightedCovarianceState, process_observation!, exp_weighted_pass!, …"
```

# Exponentially Weighted Covariance: private API

## Types

```@docs
ExpWeightedCovarianceState
```

## Functions

```@docs
process_observation!(cache::ExpWeightedCovarianceState, ce::ExpWeightedCovariance, X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedCovariance, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedCovarianceState} = nothing)
exp_weighted_pass!(est::ExpWeightedCovariance, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedCovarianceState} = nothing)
exp_weighted_moment(cache::ExpWeightedCovarianceState, est::ExpWeightedCovariance)
gap_fill_value(::ExpWeightedCovariance)
variance_series(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
PortfolioOptimisers.merge_states(a::ExpWeightedCovarianceState, b::ExpWeightedCovarianceState)
Base.copy(x::PortfolioOptimisers.ExpWeightedCovarianceState)
```
