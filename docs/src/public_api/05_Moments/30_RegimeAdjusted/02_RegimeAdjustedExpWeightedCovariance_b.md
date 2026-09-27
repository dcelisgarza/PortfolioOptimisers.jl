```@meta
Description = "Regime Adjusted Exponential Weighted Covariance (b), public API of PortfolioOptimisers.jl: cov, cor, partial_fit!, var, std, merge_states."
```

# Regime Adjusted Exponential Weighted Covariance (b)

## Functions

```@docs
cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, x::VecNum; estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing, active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)
cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
```
