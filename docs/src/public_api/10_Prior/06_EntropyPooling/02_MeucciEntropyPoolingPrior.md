```@meta
Description = "Meucci entropy pooling Prior, public API of PortfolioOptimisers.jl: MeucciEntropyPoolingPrior, prior."
```

# Meucci entropy pooling Prior

```@docs
MeucciEntropyPoolingPrior
prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
