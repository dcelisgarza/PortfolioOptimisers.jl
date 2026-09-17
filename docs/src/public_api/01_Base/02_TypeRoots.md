```@meta
Description = "Type roots, public API of PortfolioOptimisers.jl: AbstractPartialFitState, DynamicAbstractWeights, Base.iterate, Base.getindex."
```

# Type roots

Estimators, algorithms, and results behave as length-1 iterables and containers to simplify dispatch and slicing in hierarchical workflows.

```@docs
AbstractPartialFitState
DynamicAbstractWeights
Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, state)
Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, i::Int)
```
