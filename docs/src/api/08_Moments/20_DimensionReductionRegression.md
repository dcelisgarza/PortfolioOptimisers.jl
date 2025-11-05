# Dimensional Reduction Regression

```@docs
PCA
PortfolioOptimisers.fit(drtgt::PCA, X::NumMat)
PPCA
PortfolioOptimisers.fit(drtgt::PPCA, X::NumMat)
DimensionReductionRegression
regression(re::DimensionReductionRegression, X::NumMat, F::NumMat)
PortfolioOptimisers.DimensionReductionTarget
PortfolioOptimisers._regression(re::DimensionReductionRegression, y::NumVec, mu::NumVec,
                    sigma::NumVec, x1::NumMat, Vp::NumMat)
PortfolioOptimisers.prep_dim_red_reg
```
