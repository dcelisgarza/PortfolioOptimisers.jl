```@meta
Description = "Nested Clustered, private API of PortfolioOptimisers.jl: RiskBudgetingOptimiser, nested_clustered_td_defaults, needs_previous_weights, is_time_dependent, …"
```

# Nested Clustered: private API

```@docs
RiskBudgetingOptimiser
nested_clustered_td_defaults
needs_previous_weights(opt::NestedClustered)
is_time_dependent(opt::NestedClustered)
reset_time_dependent_estimator(opt::NestedClustered)
assert_rc_pl(::Any)
stated_constraint_space_basis
assert_external_lcse
assert_external_optimiser(opt::ClusteringOptimisationEstimator)
assert_rc_variance
_update_asset_sets
```
