```@meta
Description = "Type roots, private API of PortfolioOptimisers.jl: AbstractEstimator, AbstractAlgorithm, AbstractResult, CrossValidationEstimator."
```

# Type roots: private API

Most types of `PortfolioOptimisers.jl` are subtypes of the abstract types below. An estimator holds the settings of a computation, an algorithm selects a variant of it, and a result holds its output. So a function can accept any estimator, or any result, with one method.

[`CrossValidationEstimator`](@ref) is the supertype of the cross-validation schemes.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
CrossValidationEstimator
```
