```@meta
Description = "Meucci Entropy Pooling, public API of PortfolioOptimisers.jl: MeucciEntropyPoolingPrior, prior."
```

# Meucci Entropy Pooling

```@docs
MeucciEntropyPoolingPrior
prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
