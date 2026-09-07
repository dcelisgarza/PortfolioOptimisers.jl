# Exponentially Weighted Variance

## Types

```@docs
ExpWeightedVariance
ExpWeightedVarianceState
process_observation!(cache::ExpWeightedVarianceState, ce::ExpWeightedVariance,
                              X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedVariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedVarianceState} = nothing)
exp_weighted_moment(cache::ExpWeightedVarianceState, est::ExpWeightedVariance)
var(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
std(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
var(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
std(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
variance_series(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
partial_fit!(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::ExpWeightedVariance, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
var(ce::ExpWeightedVariance; kwargs...)
std(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
std(ce::ExpWeightedVariance; kwargs...)
PortfolioOptimisers.merge_states(a::ExpWeightedVarianceState, b::ExpWeightedVarianceState)
Base.copy(x::PortfolioOptimisers.ExpWeightedVarianceState)
```
