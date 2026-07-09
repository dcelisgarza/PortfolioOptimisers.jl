# Pipeline

The `Pipeline` estimator reifies an end-to-end workflow — data preparation, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps fitted as a single unit. Computed slots override the terminal optimiser's internal configuration; absent steps fall back to what the optimiser computes internally.

## Pipeline symbols

```@docs
Pipeline
PipelineResult
fit(pipe::Pipeline, data::Prices_RR)
predict(res::PipelineResult, data::AbstractPricesResult, window)
fit_predict(pipe::Pipeline, data::Prices_RR)
```

## Injection

```@docs
inject_context
inject_config
inject_sigma_ucs
constraint_results
maybe_inject_step
```

## Prediction

Predicting with a fitted pipeline replays the fitted preprocessing steps — the training universe, the training imputation parameters, the returns conversion — on an unseen data window, then delegates to the existing weights-level prediction machinery. Cross-validation folds can be computed directly on price-level data ([`Prices_RR`](@ref)), so the whole workflow is fitted per fold with no test-window leakage into stateful preprocessing.

```@docs
apply_fitted_step
apply_fitted_steps
assert_universe_aligned
```
