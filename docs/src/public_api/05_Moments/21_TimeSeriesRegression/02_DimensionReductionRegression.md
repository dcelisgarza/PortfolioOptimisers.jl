```@meta
Description = "Dimension reduction regression, public API of PortfolioOptimisers.jl: PCA, PPCA, DimensionReductionRegression, standardisation_mean, fit, regression, …"
```

# Dimension reduction regression

```@docs
PCA
PPCA
DimensionReductionRegression
standardisation_mean
fit(drtgt::PCA, X::MatNum)
fit(drtgt::PPCA, X::MatNum)
regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
returns_source(re::Union{StepwiseRegression, DimensionReductionRegression})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
