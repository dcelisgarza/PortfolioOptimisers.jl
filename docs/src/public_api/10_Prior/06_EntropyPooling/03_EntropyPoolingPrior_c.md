```@meta
Description = "Entropy Pooling (c), public API of PortfolioOptimisers.jl: EntropyPoolingPrior, prior."
```

# Entropy Pooling (c)

```@docs
EntropyPoolingPrior
prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
