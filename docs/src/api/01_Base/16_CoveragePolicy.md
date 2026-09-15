# The coverage policy

## The coverage policy

A moment estimator that carries a [`CoveragePolicy`](@ref) fits each cell of its answer on the observations that cell has, instead of reducing its window to the Coverage Universe. The rule for a delisted asset is an [`PortfolioOptimisers.AbstractCoverageAlgorithm`](@ref), whose two verbs are [`PortfolioOptimisers.fold_inactive!`](@ref) at fold time and [`PortfolioOptimisers.admits`](@ref) at read-out, and the per-cell denominators live in a [`PortfolioOptimisers.CoverageCounts`](@ref) the partial-fit state carries.

```@docs
CoveragePolicy
PortfolioOptimisers.AbstractCoverageAlgorithm
DecayCoverage
ResetCoverage
ExpireCoverage
PortfolioOptimisers.fold_inactive!
PortfolioOptimisers.fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage}, state::PortfolioOptimisers.AbstractPartialFitState, ::AbstractVector{<:Bool})
PortfolioOptimisers.admits
PortfolioOptimisers.admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool, ::Integer, min_coverage::Real)
PortfolioOptimisers.admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer, min_coverage::Real)
PortfolioOptimisers.CoverageCounts
PortfolioOptimisers.coverage_counts_seed
Base.copy(x::PortfolioOptimisers.CoverageCounts)
PortfolioOptimisers.coverage_counts_view
PortfolioOptimisers.coverage_valid
PortfolioOptimisers.coverage_valid_block
PortfolioOptimisers.coverage_step!
PortfolioOptimisers.coverage_merge_stale
PortfolioOptimisers.coverage_reset!
PortfolioOptimisers.coverage_admission
PortfolioOptimisers.coverage_divide
PortfolioOptimisers.coverage_frame
PortfolioOptimisers.coverage_refuse!
PortfolioOptimisers.coverage_refuse_comoment!
```
