```@meta
Description = "Opinion pooling Prior, public API of PortfolioOptimisers.jl: OpinionPoolingAlgorithm, LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior, …"
```

# Opinion pooling Prior

```@docs
OpinionPoolingAlgorithm
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
compute_pooling
prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
