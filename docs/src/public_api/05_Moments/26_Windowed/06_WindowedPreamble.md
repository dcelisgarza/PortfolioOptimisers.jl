```@meta
Description = "Windowed preamble, public API of PortfolioOptimisers.jl: partial_fit!, mean, cov, var, coskewness, cokurtosis."
```

# Windowed preamble

A windowed estimator under [`SeedWindow`](@ref) folds. These methods fold it and read its estimate out of the fold. The [private page](@ref private-api-windowed-preamble) documents the other names of this topic.

```@docs
partial_fit!(est::PortfolioOptimisers.SeedWindowed, X::PortfolioOptimisers.MatNum; dims::Int = 1, kwargs...)
mean(me::WindowedExpectedReturns{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
cov(ce::WindowedCovariance{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
var(ve::WindowedVariance{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
coskewness(ske::WindowedCoskewness{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
cokurtosis(kte::WindowedCokurtosis{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
```
