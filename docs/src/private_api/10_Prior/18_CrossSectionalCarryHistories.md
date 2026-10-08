```@meta
Description = "Cross-Sectional Carry Histories, private API of PortfolioOptimisers.jl: cross_sectional_fold_append, cross_sectional_spare_backing, …"
```

# [Cross-Sectional Carry Histories: private API](@id private-api-cross-sectional-carry-histories)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) appends the rows of each step to its histories in place, into the spare rows of a backing array. The functions below make the append, find the backing that it writes into, and copy a state that a later step passed before the state takes a step.

```@docs
PortfolioOptimisers.cross_sectional_fold_append
PortfolioOptimisers.cross_sectional_spare_backing
PortfolioOptimisers.cross_sectional_carry_own
```
