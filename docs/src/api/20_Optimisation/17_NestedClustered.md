# Nested Clustered

```@docs
NestedClusteredResult
factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt})
NestedClustered
nested_clustered_td_defaults
factory(nco::NestedClustered, w::AbstractVector)
port_opt_view(nco::NestedClustered, i, X::MatNum, args...)
optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
needs_previous_weights(opt::NestedClustered)
is_time_dependent(opt::NestedClustered)
reset_time_dependent_estimator(opt::NestedClustered)
assert_rc_pl(::Any)
stated_constraint_space_basis
assert_external_lcse
assert_external_optimiser(opt::ClusteringOptimisationEstimator)
RiskBudgetingOptimiser
assert_rc_variance
_update_asset_sets
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
