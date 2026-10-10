```@meta
Description = "Cross-Sectional Carry Histories, private API of PortfolioOptimisers.jl: cross_sectional_fold_append, cross_sectional_fold_join, cross_sectional_fold_split, …"
```

# [Cross-Sectional Carry Histories: private API](@id private-api-cross-sectional-carry-histories)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) appends the rows of each step to its histories in place, into the spare rows of a backing array. Under observed factors the exposure history and the observed exposures share one backing, so the call with no data reads the joined history with no copy. The functions below make the append, join the two exposure histories in one backing, find the backing that the append writes into, and copy a state that a later step passed before the state takes a step. The last function of the first group rebuilds a state with some of its fields replaced.

```@docs
PortfolioOptimisers.cross_sectional_fold_append
PortfolioOptimisers.cross_sectional_fold_join
PortfolioOptimisers.cross_sectional_fold_split
PortfolioOptimisers.cross_sectional_spare_backing
PortfolioOptimisers.cross_sectional_carry_own
PortfolioOptimisers.cross_sectional_carry_with
```

A step that moves the dropped member of a family of factors under a [`BatchChoice`](@ref) rewrites the histories in the new basis, and solves again only the observations that are rank-deficient. The functions below decide whether the move folds, rewrite the basis from its own ratios, mark the Empty Factors in the new basis, solve again each observation whose answer depends on the basis, and fold the factor prior again over the factor returns of the new basis.

```@docs
PortfolioOptimisers.cross_sectional_fold_move
PortfolioOptimisers.cross_sectional_move_basis
PortfolioOptimisers.cross_sectional_rebase_state
PortfolioOptimisers.cross_sectional_rebase
PortfolioOptimisers.cross_sectional_rebase_family
PortfolioOptimisers.cross_sectional_move_marks
PortfolioOptimisers.cross_sectional_move_row_marks
PortfolioOptimisers.cross_sectional_move_solve
PortfolioOptimisers.cross_sectional_refold_factors
```

A step where a factor that was empty at every fitted observation comes alive keeps those observations, because the factor had a return of zero at each of them. The functions below decide whether the regression estimator keeps the answer of an observation whose design gains a zero column, and fold the factor prior over the factor returns of the step, or again over every fitted factor return when a factor comes alive.

```@docs
PortfolioOptimisers.cross_sectional_alive_folds
PortfolioOptimisers.cross_sectional_step_factors
```
