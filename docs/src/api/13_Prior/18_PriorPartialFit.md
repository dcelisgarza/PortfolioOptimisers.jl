# The prior family on the partial-fit seam

A prior takes the online step by one of two routes, and the type of the state it carries **is** the route.

A prior that folds its moments exactly carries a [`PortfolioOptimisers.PriorCarryState`](@ref): the moments come off its members' own folds and the rows are kept only because a [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, so a read-out never reads them. A prior that has no recursion carries a [`PortfolioOptimisers.SampleBufferState`](@ref), which [`Online`](@ref) seeds, and its read-out is the batch verb over the rows the buffer kept. The factor observations ride inside that same buffer, and whether the fold records them is decided by the estimator tree through [`PortfolioOptimisers.needs_factor_returns`](@ref), so the fold mirrors the batch verb's arity.

A host that carries the observations folds every member that folds and runs the batch verb over its own rows for every member that does not, so a caller writes the estimator they would write in batch. ADR 0136 records the decision.

```@docs
PortfolioOptimisers.PriorCarryState
PortfolioOptimisers.sample_buffer(state::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PriorCarryState, x::PortfolioOptimisers.VecNum)
PortfolioOptimisers.fold_carry
PortfolioOptimisers.merge_states(a::PortfolioOptimisers.PriorCarryState, b::PortfolioOptimisers.PriorCarryState)
Base.copy(x::PortfolioOptimisers.PriorCarryState)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.PriorCarryState, i, args...)
prior(pe::PortfolioOptimisers.AbstractPriorEstimator; kwargs...)
PortfolioOptimisers.needs_factor_returns
PortfolioOptimisers.combine_factor_answers
PortfolioOptimisers.assert_factor_returns
PortfolioOptimisers.partial_fit!(pe::PortfolioOptimisers.AbstractPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; dims::Int = 1, active_mask = nothing, estimation_mask = nothing)
PortfolioOptimisers.fold_factor_argument
PortfolioOptimisers.fold_member
PortfolioOptimisers.read_member
PortfolioOptimisers.partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}, x::PortfolioOptimisers.VecNum, ::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any, <:Option{<:PortfolioOptimisers.PriorCarryState}}; strict::Bool = false, kwargs...)
PortfolioOptimisers.partial_fit!(pe::HighOrderPriorEstimator, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::HighOrderPriorEstimator; kwargs...)
PortfolioOptimisers.partial_fit!(pe::BlackLittermanPrior, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.Option{<:PortfolioOptimisers.VecNum_MatNum} = nothing; kwargs...)
prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
PortfolioOptimisers.update_online_estimator(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})
```
