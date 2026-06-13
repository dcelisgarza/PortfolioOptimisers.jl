# Nested Clustered

```@docs
NestedClusteredResult
factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt})
NestedClustered
factory(nco::NestedClustered, w::AbstractVector)
port_opt_view(nco::NestedClustered, i, X::MatNum)
predict_outer_nco_estimator_returns
optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
needs_previous_weights(opt::NestedClustered)
assert_rc_pl(::Any)
assert_external_optimiser(opt::ClusteringOptimisationEstimator)
RiskBudgetingOptimiser
assert_rc_variance
```
