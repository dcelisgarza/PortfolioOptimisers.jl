```@meta
Description = "Exponentially Weighted Covariance, public API of PortfolioOptimisers.jl: ExpWeightedCovariance, cov, cor, var, std, partial_fit!, merge_states."
```

# Exponentially Weighted Covariance

## Types

```@docs
ExpWeightedCovariance
```

## Functions

```@docs
cov(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cor(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
partial_fit!(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
partial_fit!(ce::ExpWeightedCovariance, x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...)
cov(ce::ExpWeightedCovariance; kwargs...)
cor(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...)
cor(ce::ExpWeightedCovariance; kwargs...)
merge_states(a::ExpWeightedCovarianceState, b::ExpWeightedCovarianceState)
```
