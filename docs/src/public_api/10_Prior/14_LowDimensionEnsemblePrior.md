```@meta
Description = "Low-dimension ensemble prior, public API of PortfolioOptimisers.jl: LowDimensionEnsemblePrior, prior."
```

# Low-dimension ensemble prior

`LowDimensionEnsemblePrior` forecasts the next price relatives and their covariance. It fits many random low-dimensional regressions of the price relatives on their lagged values, and weights each regression by how well it fits the sample. The mean and the covariance come from the same regressions, so they are a consistent pair. It is the prior of the online low-dimension ensemble method.

```@docs
LowDimensionEnsemblePrior
prior(pe::LowDimensionEnsemblePrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
