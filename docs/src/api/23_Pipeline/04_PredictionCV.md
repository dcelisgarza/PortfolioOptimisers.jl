# Pipeline cross-validation

## The price-level restriction (the rolling-window rule)

Combinatorial and multiple-randomised cross-validation recombine non-contiguous groups / resampled paths. A pipeline that *starts from prices* runs a rolling, order-dependent transform (a `PricesToReturns`, or any windowed preprocessing) that needs contiguous input rows, so those schemes are rejected at the price-level `split`. They are supported for a **returns-level** pipeline (below), which has no such transform.

```@docs
port_opt_view(pipe::Pipeline, i, args...; kwargs...)
```

## The pipeline fold loop

`cross_val_predict` over a `Pipeline` fits the whole workflow per fold and predicts on each test window. It is also the fold loop that consumes [`TimeDependent`](@ref) schedules in a pipeline (ADR 0030, "swap, then inject"): schedules are swapped for their per-fold values *before* `fit` runs, so injection never sees a schedule and `fit`/[`run_step`](@ref) never learn about folds.

```@docs
cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)
```

## Combinatorial and asset-resampling over a returns-level pipeline

A returns-level pipeline runs the multi-path schemes like the plain-optimiser loops: combinatorial fits each split on its (possibly non-contiguous) training rows and predicts its test groups; multiple-randomised runs each path's inner walk-forward over an asset-subset view of the input, so the pipeline fits fresh on the sub-universe and never sub-selects fitted state.

```@docs
cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::CombinatorialCrossValidation)
cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised)
PortfolioOptimisers.pipeline_path_fit_and_predict
```

## Time-dependent traits and the swap over steps

The per-step legs of the time-dependent machinery: the traits recurse over a pipeline's steps, the swap maps over them (unwrapping [`PipelineStep`](@ref)-wrapped schedules), the fold-less reset resolves schedule steps to their explicit `default`, and the previous-weights factory delivers `w_prev` to the optimisation steps after the swap.

```@docs
PortfolioOptimisers.pipeline_step_is_time_dependent
is_time_dependent(p::Pipeline)
needs_previous_weights(p::Pipeline)
PortfolioOptimisers.assert_pipeline_step_fold_count
PortfolioOptimisers.update_time_dependent_step
update_time_dependent_estimator(p::Pipeline, ctx::TimeDependentContext, all_binds::Bool = true)
PortfolioOptimisers.reset_time_dependent_step
reset_time_dependent_estimator(p::Pipeline)
PortfolioOptimisers.pipeline_step_factory
factory(p::Pipeline, w::VecNum)
```
