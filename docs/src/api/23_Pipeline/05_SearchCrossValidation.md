# Search cross-validation

Tuning a pipeline widens the search boundary to the entire workflow: preprocessing hyperparameters (imputation statistics, missing-data thresholds) are searched jointly with prior, constraint, and optimiser hyperparameters, and every candidate is fitted per fold so stateful preprocessing never sees the test window. Lens keys address steps by name (`"impute.stat"`), by position (an integer key swaps the whole step), or by raw property path (`"steps[2].stat"`).

```@docs
search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Prices_RR)
fit_and_score(pipe::Pipeline, scv::AbstractSearchCrossValidationEstimator, data::Prices_RR, train_idx::VecInt, test_idx::VecInt)
pipeline_lens
pipeline_lens_val_grid
pipeline_data_view
cv_data_eltype
```
