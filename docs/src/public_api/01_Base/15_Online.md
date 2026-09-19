```@meta
Description = "The online step, public API of PortfolioOptimisers.jl: Online, AbstractAllocationSet, AbstractProgrammeAllocationSet, partial_fit!, port_opt_view, …"
```

# The online step

An estimator with no exact incremental fold keeps the observations it has seen in a [`PortfolioOptimisers.SampleBufferState`](@ref), and [`Online`](@ref) is the configuration that seeds one. The wrapper is transient: [`PortfolioOptimisers.update_online_estimator`](@ref) resolves it at warm-up, so no wrapper survives into the run. The buffer carries the per-observation masks of a [`CoveragePolicy`](@ref) and the factor observations of a factor prior beside its rows, each fixed by the first append, so one buffer serves every estimator that refits.

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
