```@meta
Description = "Cross-Sectional Regression, public API of PortfolioOptimisers.jl: AbstractCrossSectionalSolveAlgorithm, PseudoInverseFallback, RankDeficiencyRefusal, …"
```

# [Cross-Sectional Regression](@id api-cross-sectional-regression)

## Types

```@docs
AbstractCrossSectionalSolveAlgorithm
PseudoInverseFallback
RankDeficiencyRefusal
UncheckedSolve
MinimumNormSolve
DependentColumnDrop
NormalEquationsSolve
CrossSectionalSolveResult
AbstractUnseenMemberRule
ZeroUnseenMember
SolvedUnseenMember
CrossSectionalLinearRegression
CrossSectionalTargetRegression
CrossSectionalRegression
```

## Functions

```@docs
cross_sectional_regression
StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)
port_opt_view(csr::CrossSectionalRegression, i, args...)
cross_sectional_solve
cross_sectional_target_solve
assert_cross_sectional_target_solve
cross_sectional_rank
```
