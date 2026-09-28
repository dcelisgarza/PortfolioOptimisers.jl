```@meta
Description = "Exp weighted expected returns, public API of PortfolioOptimisers.jl: ExpWeightedExpectedReturns, mean, partial_fit!, merge_states."
```

# Exp weighted expected returns

## Types

```@docs
ExpWeightedExpectedReturns
```

## Functions

```@docs
mean(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns, state::ExpWeightedExpectedReturnsState; kwargs...)
mean(me::ExpWeightedExpectedReturns; kwargs...)
merge_states(a::ExpWeightedExpectedReturnsState, b::ExpWeightedExpectedReturnsState)
```
