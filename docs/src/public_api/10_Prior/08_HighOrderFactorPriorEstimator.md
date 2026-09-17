```@meta
Description = "High Order Factor Prior, public API of PortfolioOptimisers.jl: AbstractHighOrderPriorEstimator_F, HighOrderFactorPriorEstimator, prior."
```

# High Order Factor Prior

```@docs
AbstractHighOrderPriorEstimator_F
HighOrderFactorPriorEstimator
prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
