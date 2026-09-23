```@meta
Description = "The Pipeline's online step, public API of PortfolioOptimisers.jl: partial_fit!, fit, cross_val_predict."
```

# The Pipeline's online step

A [`Pipeline`](@ref) updates incrementally. [`partial_fit!`](@ref) passes each block of observations through the steps in order, up to the step that stores the rows: the prior step, or the optimiser step when the pipeline has no prior. `fit(pipe)` with no data returns the fitted [`PipelineResult`](@ref). It rebuilds the returns from the rows that the step stored, and refits over them every step that chooses assets. It then restricts the state of the step that stores the rows to the assets that remain, and runs the steps after it as a batch fit.

A step that transforms each row on its own updates with each block. A step that only chooses assets waits until you read the result. A step with no incremental form makes the pipeline throw an error that names the step, before the first update. To run such a step, wrap the pipeline as `Online(pipe)`. Its [`PortfolioOptimisers.PipelineBufferState`](@ref) then stores the input data, and reading the result runs a batch fit over it.

```@docs
partial_fit!(pipe::Pipeline{<:Any, <:Any, <:PortfolioOptimisers.Option{<:Union{<:PortfolioOptimisers.PipelineBufferState, <:PortfolioOptimisers.ReturnsBufferState}}}, data::Prices_RR)
fit(pipe::Pipeline)
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PipelineBufferState, data::Prices_RR)
cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER)
cross_val_predict(r::PortfolioOptimisers.PipelineResume, data::Prices_RR, cv::CVER)
```
