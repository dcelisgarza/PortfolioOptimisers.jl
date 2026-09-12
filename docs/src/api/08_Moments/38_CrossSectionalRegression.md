# [Cross-Sectional Regression](@id api-cross-sectional-regression)

## Types

```@docs
AbstractCrossSectionalSolveAlgorithm
PseudoInverseFallback
RankDeficiencyRefusal
UncheckedSolve
MinimumNormSolve
CrossSectionalLinearRegression
CrossSectionalTargetRegression
CrossSectionalRegression
```

## Functions

```@docs
cross_sectional_regression
cross_sectional_design_mask
cross_sectional_coefficients
cross_sectional_rank
cross_sectional_solve
cross_sectional_systematic
StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)
cross_sectional_r2
mean_cross_sectional_r2
port_opt_view(csr::CrossSectionalRegression, i, args...)
```
