```@meta
Description = "Operators, private API of PortfolioOptimisers.jl: :⊗, :⊙, :⊘, :⊕, :⊖, dot_scalar."
```

# Operators: private API

## Mathematical functions

`⊙`, `⊘`, `⊕` and `⊖` multiply, divide, add and subtract element by element, and each accepts a scalar or an array on either side. `⊗` gives the outer product of two arrays as a matrix. [`dot_scalar`](@ref) takes the dot product of two vectors. When one side is a scalar, it treats that scalar as a vector of equal entries, and the scalar can be a `JuMP` expression.

```@docs
:⊗
:⊙
:⊘
:⊕
:⊖
dot_scalar
```
