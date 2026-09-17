```@meta
Description = "The Pipeline's online step, public API of PortfolioOptimisers.jl: partial_fit!, fit, cross_val_predict."
```

# The Pipeline's online step

A [`Pipeline`](@ref) is a host of the online step (ADR 0142). [`partial_fit!`](@ref) walks the steps in order and folds each block of observations through them into the **row owner** — the prior step, else the optimiser step — and `fit(pipe)` with no data reads the fitted [`PipelineResult`](@ref) out: it reconstitutes the carrier from the owner's rows, refits every universe step over it, views the owner's state to the surviving assets, and runs the tail as batch. A row-local step folds, a universe-only step defers to a view, and a step with no online form is refused at warm-up by name unless the pipeline declares a refit with `Online(pipe)`, whose [`PortfolioOptimisers.PipelineBufferState`](@ref) holds the input carrier and reads out by a batch fit.

```@docs
partial_fit!(pipe::Pipeline{<:Any, <:Any, <:PortfolioOptimisers.Option{<:Union{<:PortfolioOptimisers.PipelineBufferState, <:PortfolioOptimisers.ReturnsBufferState}}}, data::Prices_RR)
fit(pipe::Pipeline)
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.PipelineBufferState, data::Prices_RR)
cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER)
cross_val_predict(r::PortfolioOptimisers.PipelineResume, data::Prices_RR, cv::CVER)
```
