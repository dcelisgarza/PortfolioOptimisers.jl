# Dimensional Reduction Regression

```@docs
PCA
PortfolioOptimisers.fit(drtgt::PCA, X::AbstractMatrix)
PPCA
PortfolioOptimisers.fit(drtgt::PPCA, X::AbstractMatrix)
DimensionReductionRegression
regression(re::DimensionReductionRegression, X::AbstractMatrix, F::AbstractMatrix)
PortfolioOptimisers.DimensionReductionTarget
PortfolioOptimisers._regression(re::DimensionReductionRegression, y::AbstractVector, mu::AbstractVector,
                    sigma::AbstractVector, x1::AbstractMatrix, Vp::AbstractMatrix)
PortfolioOptimisers.prep_dim_red_reg
```
