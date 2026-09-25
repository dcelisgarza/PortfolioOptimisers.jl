```@meta
Description = "Vector to scalar measures, private API of PortfolioOptimisers.jl: Num_VecToScaM."
```

# Vector to scalar measures: private API

## Summary statistics

Some estimators and constraints reduce a vector to one number. A field of type [`Num_VecToScaM`](@ref) can hold a fixed number, a `VectorToScalarMeasure`, or a function that does the reduction.

```@docs
Num_VecToScaM
```
