```@meta
Description = "Higher-Moment Partial Fit, public API of PortfolioOptimisers.jl: port_opt_view, partial_fit."
```

# [Higher-Moment Partial Fit](@id api-higher-moment-partial-fit)

The incremental fit of the third and fourth co-moments. [`partial_fit!`](@ref) folds a block of observations into the state an estimator carries, and [`coskewness`](@ref) and [`cokurtosis`](@ref) read the answer out of it. Only the `FullMoment` arm of each estimator takes part, because `SemiMoment` clips against a centre that a new observation moves.

## Functions

```@docs
port_opt_view(x::PortfolioOptimisers.CoskewnessPartialFitState, i, args...)
port_opt_view(x::PortfolioOptimisers.CokurtosisPartialFitState, i, args...)
partial_fit(ske::Coskewness{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
partial_fit(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
```
