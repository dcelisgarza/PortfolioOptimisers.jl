# Opinion Pooling

```@docs
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
prior(pe::OpinionPoolingPrior, X::NumMat,
               F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
PortfolioOptimisers.OpinionPoolingAlgorithm
PortfolioOptimisers.robust_probabilities
PortfolioOptimisers.compute_pooling
```
