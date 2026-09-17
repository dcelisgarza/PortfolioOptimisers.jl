```@meta
Description = "High Order Prior, public API of PortfolioOptimisers.jl: HighOrderPriorEstimator, prior."
```

# High Order Prior

```@docs
HighOrderPriorEstimator
prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
