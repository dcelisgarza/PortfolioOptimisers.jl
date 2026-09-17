```@meta
Description = "Base Cross Validation, public API of PortfolioOptimisers.jl: PredictionReturnsResult, PredictionResult, MultiPeriodPredictionResult, …"
```

# Base Cross Validation

```@docs
PredictionReturnsResult
PredictionResult
MultiPeriodPredictionResult
PopulationPredictionResult
Base.split(res::CrossValidationResult, args...)
predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
fit_predict(opt::OptE_Opt, rd::ReturnsResult)
sort_by_measure
```
