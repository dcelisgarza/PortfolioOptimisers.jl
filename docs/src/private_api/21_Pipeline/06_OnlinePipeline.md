```@meta
Description = "The Pipeline's online step, private API of PortfolioOptimisers.jl: PipelineResume, PipelineBufferState, fold_pipeline, fold_pipeline_owner, …"
```

# [The Pipeline's online step: private API](@id private-api-the-pipelines-online-step)

```@docs
PortfolioOptimisers.PipelineResume
PortfolioOptimisers.PipelineBufferState
PortfolioOptimisers.fold_pipeline
PortfolioOptimisers.fold_pipeline_owner
PortfolioOptimisers.readout_pipeline
PortfolioOptimisers.pipeline_returns_result
PortfolioOptimisers.readout_data_step
PortfolioOptimisers.readout_owner
PortfolioOptimisers.view_owner
PortfolioOptimisers.pipeline_fold_fit
PortfolioOptimisers.step_estimator
PortfolioOptimisers.rewrap_step
PortfolioOptimisers.is_data_step
PortfolioOptimisers.is_universe_step
PortfolioOptimisers.is_row_owner
PortfolioOptimisers.pipeline_row_owner
PortfolioOptimisers.online_entry_state(p::Pipeline)
PortfolioOptimisers.pipeline_online_member
PortfolioOptimisers.step_online_member
PortfolioOptimisers.step_online_cap
PortfolioOptimisers.assert_online_entry(p::Pipeline)
PortfolioOptimisers.assert_online_owner
PortfolioOptimisers.assert_pipeline_door
PortfolioOptimisers.pipe_writes(o::Online)
PortfolioOptimisers.run_step(o::Online, ::PortfolioOptimisers.PipelineContext)
PortfolioOptimisers.update_online_step
PortfolioOptimisers.update_online_estimator(p::Pipeline)
is_time_dependent(o::Online{<:Pipeline})
PortfolioOptimisers.show_fields(p::Pipeline)
PortfolioOptimisers.copy_states(p::Pipeline)
PortfolioOptimisers.copy_step_states
PortfolioOptimisers.held_timestamps(p::Pipeline)
PortfolioOptimisers.pipeline_held_timestamps
```
