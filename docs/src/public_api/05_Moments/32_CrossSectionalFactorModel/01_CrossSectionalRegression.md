```@meta
Description = "Cross-Sectional Regression, public API of PortfolioOptimisers.jl: PseudoInverseFallback, RankDeficiencyRefusal, UncheckedSolve, MinimumNormSolve, …"
```

# [Cross-Sectional Regression](@id api-cross-sectional-regression)

## Types

```@docs
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
StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)
cross_sectional_r2
mean_cross_sectional_r2
port_opt_view(csr::CrossSectionalRegression, i, args...)
```
