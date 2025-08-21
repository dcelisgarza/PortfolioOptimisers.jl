# Dimensional Reduction Regression

```@docs
PortfolioOptimisers.DimensionReductionTarget
PCA
PCA()
PortfolioOptimisers.fit(drtgt::PCA, X::AbstractMatrix)
PPCA
PPCA()
PortfolioOptimisers.fit(drtgt::PPCA, X::AbstractMatrix)
DimensionReductionRegression
DimensionReductionRegression()
PortfolioOptimisers.prep_dim_red_reg
regression(retgt::PortfolioOptimisers.AbstractRegressionTarget, y::AbstractVector, mu::AbstractVector, sigma::AbstractVector, x1::AbstractMatrix, Vp::AbstractMatrix)
regression(re::DimensionReductionRegression, X::AbstractMatrix, F::AbstractMatrix)
```
