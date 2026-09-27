```@meta
Description = "Pipeline cross-validation, private API of PortfolioOptimisers.jl: Pipeline_OnlPipe, pipeline_cross_val_predict, pipeline_path_fit_and_predict, …"
```

# Pipeline cross-validation: private API

## The pipeline fold loop

`cross_val_predict` over a `Pipeline` fits the whole pipeline on each training window and predicts on each test window. It also replaces each [`TimeDependent`](@ref) schedule of the pipeline with its value for the fold, before `fit` runs. So no value that the pipeline passes to the optimiser is a schedule, and neither `fit` nor [`run_step`](@ref) reads the fold.

A scheme built by [`OnlineIndexWalkForward`](@ref) or [`OnlineDateWalkForward`](@ref) is an online scheme. With one, the loop fits the pipeline once on the first training window, adds the rows of each later fold with [`partial_fit!`](@ref), and gets the fitted result from `fit(pipe)` with no data. `Online(pipe)` goes through the same `cross_val_predict` methods, and refits the whole pipeline at each fold on a buffer of the rows so far. The [online pipeline](@ref private-api-the-pipelines-online-step) page documents both.

```@docs
PortfolioOptimisers.Pipeline_OnlPipe
PortfolioOptimisers.pipeline_cross_val_predict
```

## Combinatorial and asset-resampling over a returns-level pipeline

A pipeline over returns runs the schemes with many paths in the same way as a plain optimiser. [`CombinatorialCrossValidation`](@ref) fits each split on its training rows, which need not be contiguous, and predicts its test groups. [`MultipleRandomised`](@ref) runs the inner walk-forward of each path on a view of the input that holds a subset of the assets. The pipeline fits again on that subset, and never takes a subset of a state it already fitted.

```@docs
PortfolioOptimisers.pipeline_path_fit_and_predict
```

## Schedules in the steps of a pipeline

The functions below apply the schedule logic to each step of a pipeline. The check for a schedule looks through every step. So does the replacement of each schedule by its value for the fold, which also unwraps a schedule inside a [`PipelineStep`](@ref). A run with no folds sets each schedule step to its `default`. After the replacement, `pipeline_step_factory` gives the weights of the previous fold, `w_prev`, to the optimisation steps.

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
