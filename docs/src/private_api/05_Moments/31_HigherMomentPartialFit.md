```@meta
Description = "Higher-Moment Partial Fit, private API of PortfolioOptimisers.jl: CoskewnessPartialFitState, CokurtosisPartialFitState, comoment_block, shift_comoment3, …"
```

# Higher-Moment Partial Fit: private API

The coskewness and cokurtosis estimators can take the observations one block at a time. [`partial_fit!`](@ref) adds a block to the state that the estimator holds, and [`coskewness`](@ref) and [`cokurtosis`](@ref) compute the estimate from that state. Only the `FullMoment` form of the two estimators supports this, because `SemiMoment` clips the deviations at a centre that moves with every new observation.

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
```
