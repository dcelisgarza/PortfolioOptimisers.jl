```@meta
Description = "Higher-Moment Partial Fit, public API of PortfolioOptimisers.jl: port_opt_view, partial_fit."
```

# [Higher-Moment Partial Fit](@id api-higher-moment-partial-fit)

The coskewness and the cokurtosis can take new observations one block at a time. [`partial_fit!`](@ref) adds a block of observations to the state of an estimator, and [`coskewness`](@ref) and [`cokurtosis`](@ref) compute the estimate from that state. Only the `FullMoment` form of each estimator has an incremental fit. `SemiMoment` keeps only the deviations below a centre, and each new observation moves that centre.

## Functions

```@docs
port_opt_view(x::PortfolioOptimisers.CoskewnessPartialFitState, i, args...)
port_opt_view(x::PortfolioOptimisers.CokurtosisPartialFitState, i, args...)
partial_fit(ske::Coskewness{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
partial_fit(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
```
