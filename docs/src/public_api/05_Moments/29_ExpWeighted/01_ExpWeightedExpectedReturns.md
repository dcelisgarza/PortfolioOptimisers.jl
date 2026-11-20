```@meta
Description = "Exp weighted expected returns, public API of PortfolioOptimisers.jl: ZeroStartDebias, ExpWeightedExpectedReturns, zero_start_mean, mean, partial_fit!, …"
```

# Exp weighted expected returns

## Types

```@docs
ZeroStartDebias
ExpWeightedExpectedReturns
```

## Functions

```@docs
zero_start_mean
mean(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(me::ExpWeightedExpectedReturns, x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
mean(me::ExpWeightedExpectedReturns, state::ExpWeightedExpectedReturnsState; kwargs...)
mean(me::ExpWeightedExpectedReturns; kwargs...)
merge_states(a::ExpWeightedExpectedReturnsState, b::ExpWeightedExpectedReturnsState)
```
