# Base optimisation

All optimisers are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/24_Aliases).

```@docs
AbstractOptimisationEstimator
BaseOptimisationEstimator
OptimisationEstimator
NonFiniteAllocationOptimisationEstimator
OptimisationAlgorithm
OptimisationResult
NonFiniteAllocationOptimisationResult
OptimisationReturnCode
VecOptRetCode
OptRetCode_VecOptRetCode
OptimisationModelResult
OptimisationSuccess
OptimisationFailure
JuMPWeightFinaliserFormulation
RelativeErrorWeightFinaliser
SquaredRelativeErrorWeightFinaliser
AbsoluteErrorWeightFinaliser
SquaredAbsoluteErrorWeightFinaliser
WeightFinaliser
IterativeWeightFinaliser
JuMPWeightFinaliser
_optimise
optimise(opt::OptimisationResult, args...; kwargs...)
optimise(opt::OptimisationEstimator, args...; kwargs...)
calc_net_returns(res::OptimisationResult, X::MatNum, fees::Option{<:Fees} = nothing)
assert_special_nco_requirements(::OptE_Opt)
assert_special_nco_requirements(opt::VecOptE_Opt)
needs_previous_weights(::Nothing)
needs_previous_weights(::OptE_Opt)
needs_previous_weights(opt::VecOptE_Opt)
needs_previous_weights(td::TimeDependent)
TimeDependent
TimeDependentContext
TimeDependentCallable
PreviousWeightsFunction
VecTd
Td_VecTd
default_time_dependent_target
time_dependent_value
time_dependent_entries
time_dependent_entry_needs_previous_weights
assert_time_dependent_targets
assert_time_dependent_fold_count(::OptE_Opt, ::Integer)
assert_time_dependent_fold_count(opt::VecOptE_Opt, n::Integer)
rebuild_estimator
is_time_dependent(::OptE_Opt)
is_time_dependent(opt::VecOptE_Opt)
update_time_dependent_estimator(opt::OptE_Opt, ::TimeDependentContext)
update_time_dependent_estimator(opt::VecOptE_Opt, ctx::TimeDependentContext)
set_clustering_weight_finaliser_alg!
opt_weight_bounds
finalise_weight_bounds
port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
port_opt_view(opt::VecOptE, ::Any, args...)
assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)
assert_external_optimiser(::NonFiniteAllocationOptimisationResult)
factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})
factory(opt::OptE_Opt, ::Any)
factory(opt::VecOptE_Opt, args...)
OptE_Opt
VecOptE_Opt
VecOpt
VecOptE
extract_fees
extract_pr
```
