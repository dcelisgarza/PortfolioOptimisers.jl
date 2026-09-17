```@meta
Description = "Base optimisation, public API of PortfolioOptimisers.jl: AbstractOptimisationEstimator, OptimisationEstimator, NonFiniteAllocationOptimisationEstimator, …"
```

# Base optimisation

All optimisers are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [Public API → Aliases](../23_Aliases.md).

```@docs
AbstractOptimisationEstimator
OptimisationEstimator
NonFiniteAllocationOptimisationEstimator
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
_optimise
calc_net_returns(res::OptimisationResult, X::MatNum, fees::Option{<:Fees} = nothing, wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
factory(td::TimeDependent, args...)
optimise(td::TD_OptE_Opt, args...; kwargs...)
port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})
factory(opt::OptE_Opt, ::Any)
BaseOptimisationEstimator
OptimisationAlgorithm
OptimisationResult
NonFiniteAllocationOptimisationResult
OptimisationReturnCode
OptimisationModelResult
JuMPWeightFinaliserFormulation
WeightFinaliser
TimeDependentCallable
TimeDependentConstraintCallable
TimeDependentOptimiserCallable
needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm, <: AbstractResult}})
needs_previous_weights(::OptE_Opt)
needs_previous_weights(opt::VecOptE_Opt)
needs_previous_weights(td::TimeDependent)
needs_previous_weights(opt::VecOptE_Opt_TD)
time_dependent_field_defaults
set_clustering_weight_finaliser_alg!
opt_weight_bounds
```
