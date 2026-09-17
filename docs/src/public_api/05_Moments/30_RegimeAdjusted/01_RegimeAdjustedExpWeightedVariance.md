```@meta
Description = "Regime Adjusted Exponential Weighted Variance, public API of PortfolioOptimisers.jl: RegimeAdjustedMethod, LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted, regime_multiplier, …"
```

# Regime Adjusted Exponential Weighted Variance

## Types

```@docs
RegimeAdjustedMethod
LogRegimeAdjusted
FirstMomentRegimeAdjusted
RootMeanSquaredAdjusted
RegimeAdjustedExpWeightedVariance
```

## Functions

```@docs
regime_multiplier
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, x::VecNum; estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing, active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
merge_states(a::RegimeAdjustedVarianceState, b::RegimeAdjustedVarianceState)
```
