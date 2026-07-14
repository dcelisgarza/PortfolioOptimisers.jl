# Pipeline cross-validation

## Deliberately unsupported CV boundaries

Combinatorial and multiple-randomised cross-validation are defined over the returns matrix and cannot drive contiguous input-row windows; wrapping a `Pipeline` inside a meta-optimiser needs an asset view of a universe that is itself fitted state (ADR 0028, "Future expansion"). Each fails with an explanatory error rather than silently doing the wrong thing.

```@docs
Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)
Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)
port_opt_view(pipe::Pipeline, i, args...; kwargs...)
cross_val_predict(::Pipeline, ::AbstractReturnsResult, ::MultipleRandomised; kwargs...)
```

## The pipeline fold loop

`cross_val_predict` over a `Pipeline` fits the whole workflow per fold and predicts on each test window. It is also the fold loop that consumes [`TimeDependent`](@ref) schedules in a pipeline (ADR 0030, "swap, then inject"): schedules are swapped for their per-fold values *before* `fit` runs, so injection never sees a schedule and `fit`/[`run_step`](@ref) never learn about folds.

```@docs
cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)
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
