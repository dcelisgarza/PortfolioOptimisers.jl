# Grid search cross validation

```@docs
GridSearchCrossValidation
search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                        gscv::GridSearchCrossValidation, rd::ReturnsResult)
 search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation{<:Any,
                                                                 <:CombinatorialCrossValidation},
                                 rd::ReturnsResult)
lens_val_grid(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
```
