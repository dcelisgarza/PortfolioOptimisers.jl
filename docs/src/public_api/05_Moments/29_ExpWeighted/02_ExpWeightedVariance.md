```@meta
Description = "Exponentially Weighted Variance, public API of PortfolioOptimisers.jl: ExpWeightedVariance, var, std, partial_fit!."
```

# Exponentially Weighted Variance

## Types

```@docs
ExpWeightedVariance
```

## Functions

```@docs
var(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
std(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
var(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::ExpWeightedVariance, x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
var(ce::ExpWeightedVariance; kwargs...)
std(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
std(ce::ExpWeightedVariance; kwargs...)
```
