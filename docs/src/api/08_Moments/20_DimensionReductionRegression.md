# Dimensional Reduction Regression

```@docs
PCA
fit(drtgt::PCA, X::MatNum)
PPCA
fit(drtgt::PPCA, X::MatNum)
DimensionReductionRegression
regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
DimensionReductionTarget
_regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
                    sigma::VecNum, x1::MatNum, Vp::MatNum)
prep_dim_red_reg
```
