```@meta
Description = "PortfolioOptimisers pipeline, public API of PortfolioOptimisers.jl: Pipeline, PipelineResult, fit, StatsAPI.predict, fit_predict, port_opt_view, …"
```

# PortfolioOptimisers pipeline

The `Pipeline` estimator reifies an end-to-end workflow — data preparation, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps fitted as a single unit. Computed slots override the terminal optimiser's internal configuration; absent steps fall back to what the optimiser computes internally.

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

A [`TrainTestSplit`](@ref) step reserves a held-out test window before any other step runs. It is pinned to the **first** position — a stateful step fitted before it would have seen the held-out rows — and excludes cross-validation, which defines its own train/test windows. `fit_predict(pipe, data)` predicts on the window the split reserved.

```@docs
PortfolioOptimisers.assert_split_position
PortfolioOptimisers.has_split
PortfolioOptimisers.assert_no_holdout
PortfolioOptimisers.holdout_window
```
