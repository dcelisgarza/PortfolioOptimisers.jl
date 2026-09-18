```@meta
Description = "Grid search cross validation, public API of PortfolioOptimisers.jl: GridSearchCrossValidation, search_cross_validation."
```

# Grid search cross validation

```@docs
GridSearchCrossValidation
search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator, gscv::GridSearchCrossValidation, rd::ReturnsResult)
search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator, gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation}, rd::ReturnsResult)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
