# Opinion Pooling

```@docs
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
factory(pe::OpinionPoolingPrior, w::ObsWeights)
prior(pe::OpinionPoolingPrior, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
prior_view(pr::OpinionPoolingPrior, rd)
OpinionPoolingAlgorithm
robust_probabilities
compute_pooling
```
