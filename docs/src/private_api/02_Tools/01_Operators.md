```@meta
Description = "Operators, private API of PortfolioOptimisers.jl: :⊗, :⊙, :⊘, :⊕, :⊖, dot_scalar, support_product."
```

# [Operators: private API](@id private-api-operators)

## Mathematical functions

`⊙`, `⊘`, `⊕` and `⊖` multiply, divide, add and subtract element by element, and each accepts a scalar or an array on either side. `⊗` gives the outer product of two arrays as a matrix. [`dot_scalar`](@ref) takes the dot product of two vectors. When one side is a scalar, it treats that scalar as a vector of equal entries, and the scalar can be a `JuMP` expression. [`support_product`](@ref) multiplies matrices and reads the right operand only where a row of the left operand has a coefficient that is not zero, so a `NaN` reaches only the rows that read it.

```@docs
:⊗
:⊙
:⊘
:⊕
:⊖
dot_scalar
support_product
```
