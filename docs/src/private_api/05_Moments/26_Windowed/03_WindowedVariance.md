```@meta
Description = "Windowed variance, private API of PortfolioOptimisers.jl: variance_series."
```

# Windowed variance: private API

## Functions

```@docs
variance_series(ve::WindowedVariance, X::MatNum; dims::Int = 1, kwargs...)
variance_series(ve::WindowedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
