# Exponentially Weighted Expected Returns

## Types

```@docs
ExpWeightedExpectedReturns
ExpWeightedExpectedReturnsState
process_observation!(cache::ExpWeightedExpectedReturnsState,
                              me::ExpWeightedExpectedReturns, X::VecNum,
                              active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedExpectedReturns, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedExpectedReturnsState} = nothing)
exp_weighted_moment(cache::ExpWeightedExpectedReturnsState,
                             est::ExpWeightedExpectedReturns)
mean(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns,
                         state::ExpWeightedExpectedReturnsState; kwargs...)
mean(me::ExpWeightedExpectedReturns; kwargs...)
PortfolioOptimisers.merge_states(a::ExpWeightedExpectedReturnsState, b::ExpWeightedExpectedReturnsState)
Base.copy(x::PortfolioOptimisers.ExpWeightedExpectedReturnsState)
```
