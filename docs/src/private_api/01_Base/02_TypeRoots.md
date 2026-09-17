```@meta
Description = "Type roots, private API of PortfolioOptimisers.jl: AbstractEstimator, AbstractAlgorithm, AbstractResult, CrossValidationEstimator."
```

# Type roots: private API

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. Enabling us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

Custom types are the bread and butter of `PortfolioOptimisers.jl`, the following types and utilities are non-specific and used throughout the library.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
CrossValidationEstimator
```
