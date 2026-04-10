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
```
