```@meta
Description = "Cross-Sectional Forecast History Carry, public API of PortfolioOptimisers.jl: orthogonal_forecast_step."
```

# [Cross-Sectional Forecast History Carry](@id api-cross-sectional-forecast-history-carry)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) dispatches on its orthogonal forecast fit at one step, and a new rule of [`AbstractOrthogonalForecastFit`](@ref) can add a method for it. The other names of this topic are private, and the [private page](@ref private-api-cross-sectional-forecast-history-carry) documents them.

```@docs
orthogonal_forecast_step
```
