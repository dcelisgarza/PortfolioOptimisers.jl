# Base optimisation

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
is_time_dependent(::OptE_Opt)
is_time_dependent(opt::VecOptE_Opt)
update_time_dependent_estimator(opt::OptE_Opt, args...)
update_time_dependent_estimator(opt::VecOptE_Opt, args...)
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
_extract_fees
_extract_pr
```
