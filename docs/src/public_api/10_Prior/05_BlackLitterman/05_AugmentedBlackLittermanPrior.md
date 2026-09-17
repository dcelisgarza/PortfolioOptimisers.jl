```@meta
Description = "Augmented Black-Litterman Prior, public API of PortfolioOptimisers.jl: AugmentedBlackLittermanPrior, prior."
```

# Augmented Black-Litterman Prior

```@docs
AugmentedBlackLittermanPrior
prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
