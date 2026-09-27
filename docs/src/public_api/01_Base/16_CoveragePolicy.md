```@meta
Description = "The coverage policy, public API of PortfolioOptimisers.jl: CoveragePolicy, AbstractCoverageAlgorithm, DecayCoverage, ResetCoverage, ExpireCoverage, …"
```

# The coverage policy

By default a moment estimator fits on the assets whose returns are finite and active at every observation of the window, the coverage universe. Put a [`CoveragePolicy`](@ref) in the `cvg` field of the estimator, and it fits each entry of its result on the observations where every asset of that entry is finite and active. An asset with a short history then still gets an estimate. An asset whose share of the observations is below `min_coverage` is `NaN` in the result.

An [`AbstractCoverageAlgorithm`](@ref) decides what happens to a delisted asset. `DecayCoverage` keeps the history of a delisted asset, and drops the asset from the result as soon as it becomes inactive. `ResetCoverage` deletes its history, so a relisted asset starts again from zero. `ExpireCoverage` keeps the asset in the result for `after` more observations. During an incremental fit, [`fold_inactive!`](@ref) applies the rule when an asset becomes inactive, and [`admits`](@ref) decides whether the asset appears in the result when you read the estimate. The observation count of each entry is stored in a [`PortfolioOptimisers.CoverageCounts`](@ref), which the state of the incremental fit carries.

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
