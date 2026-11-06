```@meta
Description = "Vector to scalar measures, private API of PortfolioOptimisers.jl: Num_VecToScaM, finite_vec_to_real_measure."
```

# Vector to scalar measures: private API

## Summary statistics

Some estimators and constraints reduce a vector to one number. A field of type [`Num_VecToScaM`](@ref) can hold a fixed number, a `VectorToScalarMeasure`, or a function that does the reduction.

```@docs
Num_VecToScaM
```

A summary of a correlation column skips the correlations that the data do not define.

```@docs
finite_vec_to_real_measure
```
