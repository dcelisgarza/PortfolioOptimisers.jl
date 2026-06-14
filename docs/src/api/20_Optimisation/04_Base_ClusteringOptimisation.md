# Base clustering optimisation

```@docs
BaseClusteringOptimisationEstimator
ClusteringOptimisationEstimator
HierarchicalResult
factory(res::HierarchicalResult, fb::Option{<:OptE_Opt})
HierarchicalOptimiser
needs_previous_weights(opt::HierarchicalOptimiser)
factory(opt::HierarchicalOptimiser, w::AbstractVector)
port_opt_view(hco::HierarchicalOptimiser, i, args...)
assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
```
