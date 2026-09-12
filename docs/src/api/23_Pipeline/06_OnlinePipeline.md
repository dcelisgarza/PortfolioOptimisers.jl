# The Pipeline's online step

A [`Pipeline`](@ref) is a host of the online step (ADR 0142). [`partial_fit!`](@ref) walks the steps in order and folds each block of observations through them into the **row owner** — the prior step, else the optimiser step — and `fit(pipe)` with no data reads the fitted [`PipelineResult`](@ref) out: it reconstitutes the carrier from the owner's rows, refits every universe step over it, views the owner's state to the surviving assets, and runs the tail as batch. A row-local step folds, a universe-only step defers to a view, and a step with no online form is refused at warm-up by name unless the pipeline declares a refit with `Online(pipe)`, whose [`PortfolioOptimisers.PipelineBufferState`](@ref) holds the input carrier and reads out by a batch fit.

## The verbs

```@docs
partial_fit!(pipe::Pipeline{<:Any, <:Any, <:PortfolioOptimisers.Option{<:Union{<:PortfolioOptimisers.PipelineBufferState, <:PortfolioOptimisers.ReturnsBufferState}}}, data::Prices_RR)
PortfolioOptimisers.fold_pipeline
PortfolioOptimisers.fold_pipeline_owner
fit(pipe::Pipeline)
PortfolioOptimisers.readout_pipeline
PortfolioOptimisers.pipeline_returns_result
PortfolioOptimisers.readout_data_step
PortfolioOptimisers.readout_owner
PortfolioOptimisers.view_owner
PortfolioOptimisers.pipeline_fold_fit
```

## The walk and its refusals

```@docs
PortfolioOptimisers.step_estimator
PortfolioOptimisers.rewrap_step
PortfolioOptimisers.is_data_step
PortfolioOptimisers.is_universe_step
PortfolioOptimisers.is_row_owner
PortfolioOptimisers.pipeline_row_owner
PortfolioOptimisers.online_entry_state(p::Pipeline)
PortfolioOptimisers.pipeline_online_member
PortfolioOptimisers.step_online_member
PortfolioOptimisers.assert_online_entry(p::Pipeline)
PortfolioOptimisers.assert_online_owner
PortfolioOptimisers.assert_pipeline_door
```

## The declared refit and the wrappers

```@docs
PortfolioOptimisers.PipelineBufferState
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PipelineBufferState, data::Prices_RR)
PortfolioOptimisers.pipe_writes(o::Online)
PortfolioOptimisers.run_step(o::Online, ::PortfolioOptimisers.PipelineContext)
PortfolioOptimisers.update_online_step
PortfolioOptimisers.update_online_estimator(p::Pipeline)
cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER)
is_time_dependent(o::Online{<:Pipeline})
PortfolioOptimisers.show_fields(p::Pipeline)
```

## The resume

```@docs
PortfolioOptimisers.copy_states(p::Pipeline)
PortfolioOptimisers.copy_step_states
PortfolioOptimisers.held_timestamps(p::Pipeline)
PortfolioOptimisers.pipeline_held_timestamps
PortfolioOptimisers.PipelineResume
cross_val_predict(r::PortfolioOptimisers.PipelineResume, data::Prices_RR, cv::CVER)
```
