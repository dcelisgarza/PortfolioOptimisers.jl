```@meta
Description = "Dimensional Reduction Regression, public API of PortfolioOptimisers.jl: PCA, PPCA, DimensionReductionRegression, fit, regression, factory."
```

# Dimensional Reduction Regression

```@docs
PCA
PPCA
DimensionReductionRegression
fit(drtgt::PCA, X::MatNum)
fit(drtgt::PPCA, X::MatNum)
regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
factory(drtgt::DimensionReductionTarget, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
