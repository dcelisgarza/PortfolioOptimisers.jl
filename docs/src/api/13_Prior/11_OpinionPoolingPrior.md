# Opinion Pooling

```@docs
LinearOpinionPooling
LogarithmicOpinionPooling
OpinionPoolingPrior
prior(pe::OpinionPoolingPrior, X::MatNum,
               F::Option{<:MatNum} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
OpinionPoolingAlgorithm
robust_probabilities
compute_pooling
```
