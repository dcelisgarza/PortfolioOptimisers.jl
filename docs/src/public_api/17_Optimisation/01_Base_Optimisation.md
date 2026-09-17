```@meta
Description = "Base optimisation, public API of PortfolioOptimisers.jl: OptimisationSuccess, OptimisationFailure, RelativeErrorWeightFinaliser, …"
```

# Base optimisation

All optimisers are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [Public API → Aliases](../23_Aliases.md).

```@docs
OptimisationSuccess
OptimisationFailure
RelativeErrorWeightFinaliser
SquaredRelativeErrorWeightFinaliser
AbsoluteErrorWeightFinaliser
SquaredAbsoluteErrorWeightFinaliser
IterativeWeightFinaliser
JuMPWeightFinaliser
TimeDependent
TimeDependentContext
PreviousWeightsFunction
NoDefault
TimeDependentDefaultError
optimise(opt::OptimisationResult, args...; kwargs...)
optimise(opt::OptimisationEstimator, args...; kwargs...)
calc_net_returns(res::OptimisationResult, X::MatNum, fees::Option{<:Fees} = nothing, wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
factory(td::TimeDependent, args...)
optimise(td::TD_OptE_Opt, args...; kwargs...)
port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})
factory(opt::OptE_Opt, ::Any)
```
