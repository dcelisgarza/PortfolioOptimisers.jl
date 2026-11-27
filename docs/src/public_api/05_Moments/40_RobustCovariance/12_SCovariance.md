```@meta
Description = "S covariance, public API of PortfolioOptimisers.jl: AbstractSAlgorithm, DetS, SCovariance, s_starts, s_refine, StatsAPI.fit."
```

# S covariance

```@docs
AbstractSAlgorithm
DetS
SCovariance
PortfolioOptimisers.s_starts
PortfolioOptimisers.s_refine
StatsAPI.fit(ce::SCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
