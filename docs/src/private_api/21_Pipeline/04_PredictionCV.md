```@meta
Description = "Pipeline cross-validation, private API of PortfolioOptimisers.jl: Pipeline_OnlPipe, pipeline_cross_val_predict, pipeline_path_fit_and_predict, …"
```

# Pipeline cross-validation: private API

## The pipeline fold loop

`cross_val_predict` over a `Pipeline` fits the whole workflow per fold and predicts on each test window. It is also the fold loop that consumes [`TimeDependent`](@ref) schedules in a pipeline (ADR 0030, "swap, then inject"): schedules are swapped for their per-fold values *before* `fit` runs, so injection never sees a schedule and `fit`/[`run_step`](@ref) never learn about folds.

A scheme that declares a Fold Fit sends the loop down its online arm, where the pipeline is warmed up once, folded fold by fold through [`partial_fit!`](@ref), and read out through `fit(pipe)`; `Online(pipe)` takes the same doors as the declared refit. See [the Pipeline's online step](06_OnlinePipeline.md).

```@docs
PortfolioOptimisers.Pipeline_OnlPipe
PortfolioOptimisers.pipeline_cross_val_predict
```

## Combinatorial and asset-resampling over a returns-level pipeline

A returns-level pipeline runs the multi-path schemes like the plain-optimiser loops: combinatorial fits each split on its (possibly non-contiguous) training rows and predicts its test groups; multiple-randomised runs each path's inner walk-forward over an asset-subset view of the input, so the pipeline fits fresh on the sub-universe and never sub-selects fitted state.

```@docs
PortfolioOptimisers.pipeline_path_fit_and_predict
```

## Time-dependent traits and the swap over steps

The per-step legs of the time-dependent machinery: the traits recurse over a pipeline's steps, the swap maps over them (unwrapping [`PipelineStep`](@ref)-wrapped schedules), the fold-less reset resolves schedule steps to their explicit `default`, and the previous-weights factory delivers `w_prev` to the optimisation steps after the swap.

```@docs
PortfolioOptimisers.pipeline_step_is_time_dependent
is_time_dependent(p::Pipeline)
PortfolioOptimisers.assert_pipeline_step_fold_count
PortfolioOptimisers.update_time_dependent_step
update_time_dependent_estimator(p::Pipeline, ctx::TimeDependentContext, all_binds::Bool = true)
PortfolioOptimisers.reset_time_dependent_step
reset_time_dependent_estimator(p::Pipeline)
PortfolioOptimisers.pipeline_step_factory
```
