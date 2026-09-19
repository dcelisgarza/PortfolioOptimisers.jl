```@meta
Description = "Low-dimension ensemble prior, public API of PortfolioOptimisers.jl: LowDimensionEnsemblePrior, prior."
```

# Low-dimension ensemble prior

The one-step forecast of the price relatives and their predictive covariance, read off the same set of random low-dimensional lagged regressions weighted by their in-sample fit: the prior of the online low-dimension ensemble method, and a mean and covariance any programme may hold as a consistent pair.

```@docs
LowDimensionEnsemblePrior
prior(pe::LowDimensionEnsemblePrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
