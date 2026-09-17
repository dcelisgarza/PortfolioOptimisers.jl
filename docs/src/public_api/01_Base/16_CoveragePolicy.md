```@meta
Description = "The coverage policy, public API of PortfolioOptimisers.jl: CoveragePolicy, AbstractCoverageAlgorithm, DecayCoverage, ResetCoverage, ExpireCoverage, …"
```

# The coverage policy

A moment estimator that carries a [`CoveragePolicy`](@ref) fits each cell of its answer on the observations that cell has, instead of reducing its window to the Coverage Universe. The rule for a delisted asset is an [`AbstractCoverageAlgorithm`](@ref), whose two verbs are [`fold_inactive!`](@ref) at fold time and [`admits`](@ref) at read-out, and the per-cell denominators live in a [`PortfolioOptimisers.CoverageCounts`](@ref) the partial-fit state carries.

```@docs
CoveragePolicy
AbstractCoverageAlgorithm
DecayCoverage
ResetCoverage
ExpireCoverage
fold_inactive!
fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage}, state::PortfolioOptimisers.AbstractPartialFitState, ::AbstractVector{<:Bool})
admits
admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool, ::Integer, min_coverage::Real)
admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer, min_coverage::Real)
```
