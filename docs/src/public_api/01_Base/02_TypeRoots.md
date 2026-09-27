```@meta
Description = "Type roots, public API of PortfolioOptimisers.jl: AbstractPartialFitState, DynamicAbstractWeights, Base.iterate, Base.getindex."
```

# Type roots

Every estimator, algorithm and result iterates as a collection of one element, and `x[1]` returns `x`. This lets a function that loops over a vector of estimators accept a single estimator as well.

```@docs
AbstractPartialFitState
DynamicAbstractWeights
Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, state)
Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, i::Int)
```
