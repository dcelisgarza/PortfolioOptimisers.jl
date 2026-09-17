```@meta
Description = "The coverage policy, public API of PortfolioOptimisers.jl: CoveragePolicy, DecayCoverage, ResetCoverage, ExpireCoverage."
```

# The coverage policy

A moment estimator that carries a [`CoveragePolicy`](@ref) fits each cell of its answer on the observations that cell has, instead of reducing its window to the Coverage Universe. The rule for a delisted asset is an [`PortfolioOptimisers.AbstractCoverageAlgorithm`](@ref), whose two verbs are [`PortfolioOptimisers.fold_inactive!`](@ref) at fold time and [`PortfolioOptimisers.admits`](@ref) at read-out, and the per-cell denominators live in a [`PortfolioOptimisers.CoverageCounts`](@ref) the partial-fit state carries.

```@docs
CoveragePolicy
DecayCoverage
ResetCoverage
ExpireCoverage
```
