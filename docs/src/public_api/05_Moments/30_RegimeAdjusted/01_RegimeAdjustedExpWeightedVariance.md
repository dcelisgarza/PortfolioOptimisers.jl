```@meta
Description = "Regime Adjusted Exponential Weighted Variance, public API of PortfolioOptimisers.jl: LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted, …"
```

# Regime Adjusted Exponential Weighted Variance

## Types

```@docs
LogRegimeAdjusted
FirstMomentRegimeAdjusted
RootMeanSquaredAdjusted
RegimeAdjustedExpWeightedVariance
```

## Functions

```@docs
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, x::VecNum; estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing, active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
```
