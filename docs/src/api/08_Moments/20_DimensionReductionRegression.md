# Dimensional Reduction Regression

```@docs
PCA
PortfolioOptimisers.fit(drtgt::PCA, X::MatNum)
PPCA
PortfolioOptimisers.fit(drtgt::PPCA, X::MatNum)
DimensionReductionRegression
regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
PortfolioOptimisers.DimensionReductionTarget
PortfolioOptimisers._regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
                    sigma::VecNum, x1::MatNum, Vp::MatNum)
PortfolioOptimisers.prep_dim_red_reg
```
