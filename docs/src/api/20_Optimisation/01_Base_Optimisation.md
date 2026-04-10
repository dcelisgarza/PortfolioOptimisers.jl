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
optimise
calc_net_returns(res::NonFiniteAllocationOptimisationResult, X::MatNum,
                          fees::Option{<:Fees} = nothing)
assert_special_nco_requirements(::OptE_Opt)
needs_previous_weights(::OptE_Opt)
is_time_dependent(::OptE_Opt)
update_time_dependent_estimator(opt::OptE_Opt, args...)
finalise_weight_bounds
_optimise
opt_view(opt::AbstractOptimisationEstimator, args...)
```
