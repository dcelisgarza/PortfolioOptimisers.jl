# Prediction and cross-validation

Predicting with a fitted pipeline replays the fitted preprocessing steps — the training universe, the training imputation parameters, the returns conversion — on an unseen data window, then delegates to the existing weights-level prediction machinery. Cross-validation folds can be computed directly on price-level data ([`Rd_Pr`](@ref)), so the whole workflow is fitted per fold with no test-window leakage into stateful preprocessing.

## Prediction

```@docs
predict(res::PipelineResult, data::AbstractPricesResult, window)
apply_fitted_step
apply_fitted_steps
assert_universe_aligned
```

## Price-level splitting

```@docs
Rd_Pr
cv_nobs
cv_timestamps
```

## Deliberately unsupported boundaries

Combinatorial and multiple-randomised cross-validation are defined over the returns matrix and cannot drive contiguous input-row windows; wrapping a `Pipeline` inside a meta-optimiser needs an asset view of a universe that is itself fitted state (ADR 0028, "Future expansion"). Each fails with an explanatory error rather than silently doing the wrong thing.

```@docs
Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)
Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)
port_opt_view(pipe::Pipeline, i, args...; kwargs...)
```
