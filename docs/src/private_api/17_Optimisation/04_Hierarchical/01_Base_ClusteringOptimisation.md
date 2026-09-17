```@meta
Description = "Base clustering optimisation, private API of PortfolioOptimisers.jl: BaseClusteringOptimisationEstimator, ClusteringOptimisationEstimator, …"
```

# Base clustering optimisation: private API

```@docs
BaseClusteringOptimisationEstimator
ClusteringOptimisationEstimator
BaseHierarchicalOptimisationResult
HierarchicalOptimisationResult
assert_clustering_universe
hierarchical_optimiser_td_defaults
needs_previous_weights(opt::HierarchicalOptimiser)
is_time_dependent(opt::ClusteringOptimisationEstimator)
reset_time_dependent_estimator(opt::ClusteringOptimisationEstimator)
assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
```
