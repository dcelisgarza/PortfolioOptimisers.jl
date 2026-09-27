```@meta
Description = "PortfolioOptimisers pipeline, public API of PortfolioOptimisers.jl: Pipeline, PipelineResult, fit, StatsAPI.predict, fit_predict, port_opt_view, …"
```

# PortfolioOptimisers pipeline

The `Pipeline` estimator fits a list of steps as one estimator. The steps can be data preparation, prior estimation, phylogeny, uncertainty sets, constraint generation and optimisation. What a step computes replaces the matching setting of the optimiser at the end of the pipeline. For each step that the pipeline does not have, the optimiser computes the value itself, as it does outside a pipeline.

```@docs
Pipeline
PipelineResult
fit(pipe::Pipeline, data::Prices_RR)
StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult, test_idx = Colon(), cols = Colon())
fit_predict(pipe::Pipeline, data::Prices_RR)
port_opt_view(::Pipeline, args...; kwargs...)
implicit_constraint_target
```

## Holdout splitting

A [`TrainTestSplit`](@ref) step keeps a test window aside before any other step runs. It must be the first step, because a fitted step before it would see the rows of the test window. You cannot use it with cross-validation, which makes its own training and test windows. `fit_predict(pipe, data)` predicts on the test window that the split kept aside.

```@docs
PortfolioOptimisers.assert_split_position
PortfolioOptimisers.has_split
PortfolioOptimisers.assert_no_holdout
PortfolioOptimisers.holdout_window
```
