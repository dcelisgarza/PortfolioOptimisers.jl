```@meta
Description = "The prior family on the partial-fit seam, public API of PortfolioOptimisers.jl: partial_fit!, port_opt_view, merge_states, prior."
```

# The prior family on the partial-fit seam

A prior updates with `partial_fit!` in one of two ways, and the type of the state it carries shows which.

A prior that updates its moments exactly carries a [`PortfolioOptimisers.PriorCarryState`](@ref). Its moments come from the incremental fits of its members. It stores the rows only because a [`LowOrderPrior`](@ref) holds the returns `X` for the scenario risk measures, and it reads no row to compute its result. A prior with no exact update carries a [`PortfolioOptimisers.SampleBufferState`](@ref), which [`Online`](@ref) gives it, and it computes its result with its batch fit over the rows that the buffer kept. The same buffer stores the factor returns. [`PortfolioOptimisers.needs_factor_returns`](@ref) reads the members of the prior to decide whether the update stores them, so the update takes the same arguments as the batch fit.

A prior that stores the observations updates each member that has an incremental fit, and refits each member that has none over its own rows. You write the same estimator as for a batch fit.

```@docs
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PriorCarryState, x::PortfolioOptimisers.VecNum)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.PriorCarryState, i, args...)
merge_states(a::PortfolioOptimisers.PriorCarryState, b::PortfolioOptimisers.PriorCarryState)
prior(pe::PortfolioOptimisers.AbstractPriorEstimator; kwargs...)
PortfolioOptimisers.partial_fit!(pe::PortfolioOptimisers.AbstractPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; dims::Int = 1, active_mask = nothing, estimation_mask = nothing)
PortfolioOptimisers.partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}, x::PortfolioOptimisers.VecNum, ::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}; strict::Bool = false, kwargs...)
PortfolioOptimisers.partial_fit!(pe::HighOrderPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::HighOrderPriorEstimator; kwargs...)
PortfolioOptimisers.partial_fit!(pe::BlackLittermanPrior, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
```
