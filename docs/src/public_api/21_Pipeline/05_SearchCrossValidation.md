```@meta
Description = "Search cross-validation, public API of PortfolioOptimisers.jl: search_cross_validation."
```

# Search cross-validation

A search over the hyperparameters of a pipeline covers the whole workflow. It searches the preprocessing settings, such as the imputation statistic or the threshold for missing data, together with the settings of the prior, the constraints and the optimiser. Each candidate is fitted on each fold, so a fitted preprocessing step never sees the test window. A key names a setting by the name of its step, `"impute.stat"`, by position, where an integer key replaces the whole step, or by property path, `"steps[2].stat"`.

```@docs
search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Prices_RR)
search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation}, data::Prices_RR)
```
