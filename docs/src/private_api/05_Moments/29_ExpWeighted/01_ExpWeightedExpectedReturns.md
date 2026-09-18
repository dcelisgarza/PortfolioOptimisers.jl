```@meta
Description = "Exponentially Weighted Expected Returns, private API of PortfolioOptimisers.jl: ExpWeightedExpectedReturnsState, process_observation!, exp_weighted_pass!, …"
```

# Exponentially Weighted Expected Returns: private API

## Types

```@docs
ExpWeightedExpectedReturnsState
```

## Functions

```@docs
process_observation!(cache::ExpWeightedExpectedReturnsState, me::ExpWeightedExpectedReturns, X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
exp_weighted_pass!(f, est::ExpWeightedExpectedReturns, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedExpectedReturnsState} = nothing)
exp_weighted_pass!(est::ExpWeightedExpectedReturns, X::MatNum, dims::Int, active_mask::Option{<:AbstractMatrix{<:Bool}}, state::Option{<:ExpWeightedExpectedReturnsState} = nothing)
exp_weighted_moment(cache::ExpWeightedExpectedReturnsState, est::ExpWeightedExpectedReturns)
Base.copy(x::PortfolioOptimisers.ExpWeightedExpectedReturnsState)
```
