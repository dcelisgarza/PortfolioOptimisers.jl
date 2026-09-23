```@meta
Description = "Pipeline cross-validation, public API of PortfolioOptimisers.jl: port_opt_view, needs_previous_weights, cross_val_predict, factory."
```

# Pipeline cross-validation

## Combinatorial cross-validation of a pipeline from prices

Combinatorial cross-validation trains on groups of rows that are not contiguous. A pipeline that starts from prices runs a transform that needs contiguous rows, such as `PricesToReturns`. At each gap between two training groups, that transform computes one false return across the gap, so each training window is approximate. The test groups are contiguous, so the predictions do not change. Multiple randomised cross-validation resamples assets and not rows, so it keeps every window contiguous, and a pipeline from prices runs it exactly. A pipeline that starts from returns has no such transform, and it runs combinatorial cross-validation without the approximation.

```@docs
port_opt_view(pipe::Pipeline, i, args...; kwargs...)
needs_previous_weights(p::Pipeline)
```

## Cross-validation of a pipeline

`cross_val_predict` over a `Pipeline` fits every step of the pipeline on each training window, and predicts on each test window. It also resolves the [`TimeDependent`](@ref) schedules of the pipeline. Before `fit` runs on a fold, it replaces each schedule with its value for that fold. `fit` and [`run_step`](@ref) never see a schedule, and never need to know the fold.

An online walk-forward, such as one that `OnlineIndexWalkForward` builds, fits the pipeline once on the first training window. It then adds the rows of each later fold with [`partial_fit!`](@ref), and reads the result with `fit(pipe)`. A pipeline wrapped as `Online(pipe)`, which refits on the rows it stores, runs through the same calls. See [the online updates of a pipeline](06_OnlinePipeline.md).

```@docs
cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)
```

## Combinatorial and asset resampling schemes over a pipeline from returns

A pipeline that starts from returns runs these schemes as an optimiser does. Combinatorial cross-validation fits each split on its training rows, which need not be contiguous, and predicts its test groups. Multiple randomised cross-validation runs the walk-forward of each path on a subset of the assets. The pipeline then fits again on that subset, and never takes a subset of a fitted state.

```@docs
cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::CombinatorialCrossValidation)
cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised)
```

## Schedules over the steps of a pipeline

These functions apply schedules to the steps of a pipeline. The functions that detect a schedule search every step. The replacement of each schedule by its value for the fold also works step by step, and it unwraps a schedule held in a [`PipelineStep`](@ref). A fit with no fold replaces each schedule with its `default`. After the replacement, `factory` gives the previous weights `w_prev` to the optimisation steps.

```@docs
factory(p::Pipeline, w::VecNum)
```
