# Type roots

## Base abstract types

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. Enabling us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
CrossValidationEstimator
```

## Utilities

Custom types are the bread and butter of `PorfolioOptimisers.jl`, the following types and utilities are non-specific and used throughout the library.

```@docs
DynamicAbstractWeights
```

## Partial fit

```@docs
PortfolioOptimisers.AbstractPartialFitState
```

## Iteration and indexing

Estimators, algorithms, and results behave as length-1 iterables and containers to simplify dispatch and slicing in hierarchical workflows.

```@docs
Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, state)
Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, i::Int)
```
