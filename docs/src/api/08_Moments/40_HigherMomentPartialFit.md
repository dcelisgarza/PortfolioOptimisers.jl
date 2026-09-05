# [Higher-Moment Partial Fit](@id api-higher-moment-partial-fit)

The incremental fit of the third and fourth co-moments. [`partial_fit!`](@ref) folds a block of observations into the state an estimator carries, and [`coskewness`](@ref) and [`cokurtosis`](@ref) read the answer out of it. Only the `FullMoment` arm of each estimator takes part, because `SemiMoment` clips against a centre that a new observation moves.

## Types

```@docs
PortfolioOptimisers.CoskewnessPartialFitState
PortfolioOptimisers.CokurtosisPartialFitState
```

## Functions

```@docs
PortfolioOptimisers.comoment_block
PortfolioOptimisers.shift_comoment3
PortfolioOptimisers.shift_comoment4
PortfolioOptimisers.assert_partial_fittable(me::PortfolioOptimisers.AbstractExpectedReturnsEstimator, w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights}, name::AbstractString)
PortfolioOptimisers.assert_partial_fittable(::Nothing, w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights}, name::AbstractString)
Base.copy(x::PortfolioOptimisers.CoskewnessPartialFitState)
Base.copy(x::PortfolioOptimisers.CokurtosisPartialFitState)
port_opt_view(x::PortfolioOptimisers.CoskewnessPartialFitState, i, args...)
port_opt_view(x::PortfolioOptimisers.CokurtosisPartialFitState, i, args...)
partial_fit(ske::Coskewness{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
partial_fit(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
```
