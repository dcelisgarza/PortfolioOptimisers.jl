```@meta
Description = "Dimension reduction regression, private API of PortfolioOptimisers.jl: DimensionReductionTarget, ReducedBasisFit, _regression, prep_dim_red_reg, …"
```

# Dimension reduction regression: private API

```@docs
DimensionReductionTarget
ReducedBasisFit
_regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum, sigma::VecNum, x1::MatNum, Vp::MatNum)
prep_dim_red_reg
dimension_reduction_map
pin_regression_choice(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:PinnedChoice, Nothing}, ::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)
reader_frames(re::Union{StepwiseRegression, DimensionReductionRegression}, path::AbstractString, refit::Bool)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
