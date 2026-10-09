```@meta
Description = "Cross-Sectional Carry Histories, private API of PortfolioOptimisers.jl: cross_sectional_fold_append, cross_sectional_spare_backing, …"
```

# [Cross-Sectional Carry Histories: private API](@id private-api-cross-sectional-carry-histories)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) appends the rows of each step to its histories in place, into the spare rows of a backing array. Under observed factors the exposure history and the observed exposures share one backing, so the call with no data reads the joined history with no copy. The functions below make the append, join the two exposure histories in one backing, find the backing that the append writes into, and copy a state that a later step passed before the state takes a step. The last function rebuilds a state with some of its fields replaced.

```@docs
PortfolioOptimisers.cross_sectional_fold_append
PortfolioOptimisers.cross_sectional_fold_join
PortfolioOptimisers.cross_sectional_fold_split
PortfolioOptimisers.cross_sectional_spare_backing
PortfolioOptimisers.cross_sectional_carry_own
PortfolioOptimisers.cross_sectional_carry_with
```
