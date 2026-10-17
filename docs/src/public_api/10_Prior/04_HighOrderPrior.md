```@meta
Description = "High Order Prior, public API of PortfolioOptimisers.jl: HighOrderPriorEstimator, prior, partial_fit!."
```

# High Order Prior

```@docs
HighOrderPriorEstimator
prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
PortfolioOptimisers.partial_fit!(pe::HighOrderPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::HighOrderPriorEstimator; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
