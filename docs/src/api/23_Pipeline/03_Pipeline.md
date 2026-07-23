# PortfolioOptimisers pipeline

The `Pipeline` estimator reifies an end-to-end workflow — data preparation, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps fitted as a single unit. Computed slots override the terminal optimiser's internal configuration; absent steps fall back to what the optimiser computes internally.

## Pipeline symbols

```@docs
Pipeline
PipelineResult
fit(pipe::Pipeline, data::Prices_RR)
StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idx = Colon(), cols = Colon())
fit_predict(pipe::Pipeline, data::Prices_RR)
port_opt_view(::Pipeline, args...; kwargs...)
first_duplicate
```

## Holdout splitting

A [`TrainTestSplit`](@ref) step reserves a held-out test window before any other step runs. It is pinned to the **first** position — a stateful step fitted before it would have seen the held-out rows — and excludes cross-validation, which defines its own train/test windows. `fit_predict(pipe, data)` predicts on the window the split reserved.

```@docs
PortfolioOptimisers.assert_split_position
PortfolioOptimisers.has_split
PortfolioOptimisers.assert_no_holdout
PortfolioOptimisers.holdout_window
```

## Injection

The pipeline resolves its computed slots into [routing targets](@ref PIPELINE_ROUTING_TARGETS) and hands each one to the optimiser, which owns the decision of where it lands. See [`pipe_route`](@ref) for the optimiser-owned half of the seam.

```@docs
inject_context
constraint_results
constraint_targets
maybe_inject_step
pipe_required_targets
assert_routable
```

## Prediction

Predicting with a fitted pipeline replays the fitted preprocessing steps — the training universe, the training imputation parameters, the returns conversion — on an unseen data window, then delegates to the existing weights-level prediction machinery. Cross-validation folds can be computed directly on price-level data ([`Prices_RR`](@ref)), so the whole workflow is fitted per fold with no test-window leakage into stateful preprocessing.

```@docs
apply_fitted_step
apply_fitted_steps
assert_universe_aligned
```
