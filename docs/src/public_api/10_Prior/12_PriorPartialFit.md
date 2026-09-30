```@meta
Description = "Prior partial fit, public API of PortfolioOptimisers.jl: partial_fit!, port_opt_view, merge_states, prior."
```

# Prior partial fit

A prior updates with `partial_fit!` in one of two ways, and the type of the state it carries shows which.

A prior that updates its moments exactly carries a [`PortfolioOptimisers.PriorCarryState`](@ref). Its moments come from the incremental fits of its members. It stores the rows only because a [`LowOrderPrior`](@ref) holds the returns `X` for the scenario risk measures, and it reads no row to compute its result. A prior with no exact update carries a [`PortfolioOptimisers.SampleBufferState`](@ref), which [`Online`](@ref) gives it, and it computes its result with its batch fit over the rows that the buffer kept. The same buffer stores the factor returns. [`PortfolioOptimisers.needs_factor_returns`](@ref) reads the members of the prior to decide whether the update stores them, so the update takes the same arguments as the batch fit.

A prior that stores the observations updates each member that has an incremental fit, and refits each member that has none over its own rows. You write the same estimator as for a batch fit.

`partial_fit!(pe, rd)` updates a prior from a [`ReturnsResult`](@ref), as `prior(pe, rd)` fits it. It reads the returns, the factor returns and the two masks of the panel of `rd`. It also reads the per-asset data of that panel when the prior reads them, as [`PortfolioOptimisers.reads_panel_fields`](@ref) answers. A prior with no exact update stores all of these, so it takes a narrower estimation universe and the per-asset data. A prior that updates its moments exactly takes the active mask alone, so it refuses an estimation mask that differs from the active mask.

```@docs
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PriorCarryState, x::PortfolioOptimisers.VecNum)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.PriorCarryState, i, args...)
merge_states(a::PortfolioOptimisers.PriorCarryState, b::PortfolioOptimisers.PriorCarryState)
prior(pe::PortfolioOptimisers.AbstractPriorEstimator; kwargs...)
PortfolioOptimisers.partial_fit!(pe::PortfolioOptimisers.AbstractPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; dims::Int = 1, active_mask = nothing, estimation_mask = nothing)
PortfolioOptimisers.partial_fit!(pe::PortfolioOptimisers.AbstractPriorEstimator, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}, x::PortfolioOptimisers.VecNum, ::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
PortfolioOptimisers.partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}, rd::ReturnsResult)
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}; strict::Bool = false, kwargs...)
PortfolioOptimisers.partial_fit!(pe::HighOrderPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::HighOrderPriorEstimator; kwargs...)
PortfolioOptimisers.partial_fit!(pe::BlackLittermanPrior, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
PortfolioOptimisers.partial_fit!(pe::HighOrderPriorEstimator, rd::ReturnsResult)
prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
```
