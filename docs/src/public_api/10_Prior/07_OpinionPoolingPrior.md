```@meta
Description = "Opinion Pooling, public API of PortfolioOptimisers.jl: OpinionPoolingAlgorithm, LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior, …"
```

# Opinion Pooling

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
