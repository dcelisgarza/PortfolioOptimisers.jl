```@meta
Description = "Regime Adjusted Exponential Weighted Covariance (b), private API of PortfolioOptimisers.jl: gap_fill_value, Base.copy, variance_series."
```

# Regime Adjusted Exponential Weighted Covariance (b): private API

## Functions

```@docs
gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedCovarianceState)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
