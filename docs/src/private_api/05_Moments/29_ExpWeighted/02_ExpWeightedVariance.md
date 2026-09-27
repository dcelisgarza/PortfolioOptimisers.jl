```@meta
Description = "Exponentially Weighted Variance, private API of PortfolioOptimisers.jl: ExpWeightedVarianceState, process_observation!, exp_weighted_pass!, …"
```

# Exponentially Weighted Variance: private API

## Types

```@docs
ExpWeightedVarianceState
```

## Functions

```@docs
process_observation!(cache::ExpWeightedVarianceState, ce::ExpWeightedVariance, X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedVariance, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedVarianceState} = nothing)
exp_weighted_pass!(est::ExpWeightedVariance, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedVarianceState} = nothing)
exp_weighted_moment(cache::ExpWeightedVarianceState, est::ExpWeightedVariance)
variance_series(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
Base.copy(x::PortfolioOptimisers.ExpWeightedVarianceState)
exp_weighted_variance_count
```
