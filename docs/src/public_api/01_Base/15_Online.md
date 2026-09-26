```@meta
Description = "The online step, public API of PortfolioOptimisers.jl: Online, AbstractAllocationSet, AbstractProgrammeAllocationSet, partial_fit!, port_opt_view, …"
```

# The online step

Some estimators have no exact incremental fit. Such an estimator stores the observations it has seen in a [`PortfolioOptimisers.SampleBufferState`](@ref), and runs its ordinary batch fit over those rows when you read the estimate. Wrap the estimator in [`Online`](@ref) to give it a buffer. The `max_history` keyword of `Online` caps the buffer, and the estimate is then the batch fit over the last `max_history` observations.

[`PortfolioOptimisers.update_online_estimator`](@ref) replaces each `Online` with the estimator it wraps, which now carries an empty buffer. It runs once, before the first block of observations, so no `Online` remains in the estimator that you update. The buffer also stores the per-observation masks of a [`CoveragePolicy`](@ref) and the factor returns of a factor prior next to the asset returns. The first block of observations fixes which of them the buffer stores. A prior holds its rows in one buffer, and every member of the prior that has no incremental fit refits from those rows.

The page also has the supertypes of the allocation sets. An allocation set is the set of allowed weights that an online portfolio selection rule projects its step onto.

```@docs
Online
PortfolioOptimisers.AbstractAllocationSet
PortfolioOptimisers.AbstractProgrammeAllocationSet
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, X::PortfolioOptimisers.MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.MatNum} = nothing; dims::Int = 1)
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, x::PortfolioOptimisers.VecNum, f::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum} = nothing)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.SampleBufferState, i, args...)
PortfolioOptimisers.partial_fit!(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, X::PortfolioOptimisers.VecNum_MatNum; dims::Int = 1)
merge_states(a::PortfolioOptimisers.SampleBufferState, b::PortfolioOptimisers.SampleBufferState)
```
