# Base clustering optimisation

```@docs
BaseClusteringOptimisationEstimator
ClusteringOptimisationEstimator
HierarchicalResult
factory(res::HierarchicalResult, fb::Option{<:OptE_Opt})
HierarchicalOptimiser
needs_previous_weights(opt::HierarchicalOptimiser)
is_time_dependent(opt::HierarchicalOptimiser)
update_time_dependent_estimator(opt::HierarchicalOptimiser, ctx::TimeDependentContext)
is_time_dependent(opt::ClusteringOptimisationEstimator)
update_time_dependent_estimator(opt::ClusteringOptimisationEstimator, ctx::TimeDependentContext)
reset_time_dependent_estimator(opt::ClusteringOptimisationEstimator)
assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
```
