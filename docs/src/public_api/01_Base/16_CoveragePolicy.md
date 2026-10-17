```@meta
Description = "Coverage Policy, public API of PortfolioOptimisers.jl: CoveragePolicy, AbstractCoverageAlgorithm, DecayCoverage, ResetCoverage, ExpireCoverage, …"
```

# Coverage Policy

By default a moment estimator fits on the assets whose returns are finite and active at every observation of the window, the coverage universe. Put a [`CoveragePolicy`](@ref) in the `cvg` field of the estimator, and it fits each entry of its result on the observations where every asset of that entry is finite and active. An asset with a short history then still gets an estimate. An asset whose share of the observations is below `min_coverage` is `NaN` in the result.

An [`AbstractCoverageAlgorithm`](@ref) decides what happens to a delisted asset. `DecayCoverage` keeps the history of a delisted asset, and drops the asset from the result as soon as it becomes inactive. `ResetCoverage` deletes its history, so a relisted asset starts again from zero. `ExpireCoverage` keeps the asset in the result for `after` more observations. During an incremental fit, [`fold_inactive!`](@ref) applies the rule when an asset becomes inactive, and [`admits`](@ref) decides whether the asset appears in the result when you read the estimate. The observation count of each entry is stored in a [`PortfolioOptimisers.CoverageCounts`](@ref), which the state of the incremental fit carries.

Two admitted assets can each have a variance and share too few observations for a covariance. Such a pair is undetermined. An [`AbstractPeel`](@ref) in the `peel` field of the policy removes a set of assets that leaves no undetermined pair, and the covariance warns and names them, or refuses under `strict = true`. `MinimalPeel` removes the fewest assets, `GreedyPeel` removes the asset with the most undetermined pairs first, and `NoPeel` removes none, so the matrix repair refuses the `NaN`. A rule of your own implements [`peel_assets`](@ref).

```@docs
CoveragePolicy
AbstractCoverageAlgorithm
DecayCoverage
ResetCoverage
ExpireCoverage
AbstractPeel
MinimalPeel
GreedyPeel
NoPeel
peel_assets
peel_assets(::NoPeel, ::AbstractMatrix{Bool})
peel_assets(::GreedyPeel, U::AbstractMatrix{Bool})
peel_assets(::MinimalPeel, U::AbstractMatrix{Bool})
fold_inactive!
fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage}, state::PortfolioOptimisers.AbstractPartialFitState, ::AbstractVector{<:Bool})
admits
admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool, ::Integer, min_coverage::Real)
admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer, min_coverage::Real)
```
