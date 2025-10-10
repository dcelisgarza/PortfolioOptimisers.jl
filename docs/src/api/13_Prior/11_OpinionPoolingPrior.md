# Opinion Pooling

```@docs
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
prior(pe::OpinionPoolingPrior, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
PortfolioOptimisers.OpinionPoolingAlgorithm
PortfolioOptimisers.robust_probabilities
PortfolioOptimisers.compute_pooling
```
