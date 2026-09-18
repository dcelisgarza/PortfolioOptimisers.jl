```@meta
Description = "Black-Litterman Prior, public API of PortfolioOptimisers.jl: BlackLittermanPrior, prior."
```

# Black-Litterman Prior

```@docs
BlackLittermanPrior
prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
