```@meta
Description = "Cross-Sectional Regression (a), public API of PortfolioOptimisers.jl: AbstractCrossSectionalSolveAlgorithm, PseudoInverseFallback, RankDeficiencyRefusal, …"
```

# [Cross-Sectional Regression (a)](@id api-cross-sectional-regression)

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
port_opt_view(csr::CrossSectionalRegression, i, args...)
cross_sectional_solve
cross_sectional_target_solve
assert_cross_sectional_target_solve
cross_sectional_rank
```
