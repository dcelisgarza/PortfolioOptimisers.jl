# Base optimisation

All optimisers are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/25_Aliases).

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
needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                 <: AbstractResult}})
needs_previous_weights(::OptE_Opt)
needs_previous_weights(opt::VecOptE_Opt)
needs_previous_weights(td::TimeDependent)
needs_previous_weights(opt::VecOptE_Opt_TD)
TimeDependent
factory(td::TimeDependent, args...)
TimeDependentContext
TimeDependentCallable
TimeDependentOptimiserCallable
PreviousWeightsFunction
NoDefault
TimeDependentDefaultError
TD_Option
TD
TD_OptE_Opt
TDO_Option
OptE_TD
OptE_Opt_TD
VecOptE_Opt_TD
factory(opt::VecOptE_Opt_TD, args...)
TD_VecOptE_Opt
TDO_OptE_Opt
assert_nearest_optimiser_schedule
inner_fold_fields
time_dependent_value
time_dependent_fields
time_dependent_entries
time_dependent_entry_needs_previous_weights
assert_time_dependent_substitution
time_dependent_stand_in
time_dependent_reset_value
assert_time_dependent_optimiser
assert_time_dependent_fold_count(::OptE_Opt, ::Integer, ::Bool = true)
assert_time_dependent_fold_count(td::TDO_OptE_Opt, n::Integer,
                                          all_binds::Bool = true)
assert_time_dependent_fold_count(opt::VecOptE_Opt_TD, n::Integer,
                                          all_binds::Bool = true)
assert_time_dependent_fields_fold_count
rebuild_estimator
is_time_dependent(::OptE_Opt)
is_time_dependent(opt::VecOptE_Opt)
is_time_dependent(opt::BaseOptimisationEstimator)
is_time_dependent(::TimeDependent)
is_time_dependent(opt::VecOptE_Opt_TD)
update_time_dependent_estimator
update_time_dependent_fields
time_dependent_field_defaults
reset_time_dependent_estimator(opt::OptE_Opt)
reset_time_dependent_estimator(opt::BaseOptimisationEstimator)
reset_time_dependent_estimator(td::TD_OptE_Opt)
reset_time_dependent_fields
optimise(td::TD_OptE_Opt, args...; kwargs...)
set_clustering_weight_finaliser_alg!
opt_weight_bounds
finalise_weight_bounds
port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
port_opt_view(opt::VecOptE, ::Any, args...)
port_opt_view(opt::Union{<:VecOptE, <:VecOptE_Opt_TD}, i, args...)
port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)
assert_external_optimiser(::NonFiniteAllocationOptimisationResult)
assert_special_nco_requirements
factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})
factory(opt::OptE_Opt, ::Any)
factory(opt::VecOptE_Opt, args...)
assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)
entitled
OptE_Opt
VecOptE_Opt
VecOpt
VecOptE
extract_fees
extract_pr
```
