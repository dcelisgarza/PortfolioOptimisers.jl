```@meta
Description = "Opinion Pooling, public API of PortfolioOptimisers.jl: LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior, prior."
```

# Opinion Pooling

```@docs
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
