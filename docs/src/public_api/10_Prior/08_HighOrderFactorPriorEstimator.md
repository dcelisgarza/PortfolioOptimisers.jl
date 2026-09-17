```@meta
Description = "High Order Factor Prior, public API of PortfolioOptimisers.jl: HighOrderFactorPriorEstimator, prior."
```

# High Order Factor Prior

```@docs
HighOrderFactorPriorEstimator
prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
