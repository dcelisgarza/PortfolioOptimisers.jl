```@meta
Description = "The coverage policy, private API of PortfolioOptimisers.jl: AbstractCoverageAlgorithm, CoverageCounts, fold_inactive!, admits, coverage_counts_seed, …"
```

# The coverage policy: private API

```@docs
PortfolioOptimisers.AbstractCoverageAlgorithm
PortfolioOptimisers.CoverageCounts
PortfolioOptimisers.fold_inactive!
PortfolioOptimisers.fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage}, state::PortfolioOptimisers.AbstractPartialFitState, ::AbstractVector{<:Bool})
PortfolioOptimisers.admits
PortfolioOptimisers.admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool, ::Integer, min_coverage::Real)
PortfolioOptimisers.admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer, min_coverage::Real)
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
