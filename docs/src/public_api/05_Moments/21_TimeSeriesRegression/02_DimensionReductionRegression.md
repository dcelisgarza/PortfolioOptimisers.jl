```@meta
Description = "Dimension reduction regression, public API of PortfolioOptimisers.jl: PCA, PPCA, DimensionReductionRegression, fit, regression, factory, returns_source."
```

# Dimension reduction regression

```@docs
PCA
PPCA
DimensionReductionRegression
fit(drtgt::PCA, X::MatNum)
fit(drtgt::PPCA, X::MatNum)
regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
factory(drtgt::DimensionReductionTarget, args...; kwargs...)
returns_source(re::Union{StepwiseRegression, DimensionReductionRegression})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
