```@meta
Description = "Windowed covariance, private API of PortfolioOptimisers.jl: variance_series."
```

# Windowed covariance: private API

## Functions

```@docs
variance_series(ce::WindowedCovariance, X::MatNum; dims::Int = 1, kwargs...)
variance_series(ce::WindowedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
