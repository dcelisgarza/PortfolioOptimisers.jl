# Nested Clustered

```@docs
NestedClusteredResult
factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt})
NestedClustered
factory(nco::NestedClustered, w::AbstractVector)
opt_view(nco::NestedClustered, i, X::MatNum)
predict_outer_nco_estimator_returns
optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
needs_previous_weights(opt::NestedClustered)
assert_rc_pl(::Any)
assert_external_optimiser(opt::ClusteringOptimisationEstimator)
RiskBudgetingOptimiser
outer_optimisation_finaliser(wb::Option{<:WeightBounds}, wf::WeightFinaliser, resi::VecOpt, rcos::AbstractVector{<:OptimisationReturnCode}, ws::VecVecNum, wi::MatNum)
rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes)
assert_rc_variance
```
