```@meta
Description = "Cross-Sectional Forecast History Carry, private API of PortfolioOptimisers.jl: cross_sectional_forecast_refits, cross_sectional_carries_history, …"
```

# [Cross-Sectional Forecast History Carry: private API](@id private-api-cross-sectional-forecast-history-carry)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) carries the return forecast history when a slot of the prior reads it. The functions below decide when the state carries the history, bring it up to the fitted observations at each step, and give it to the slots.

```@docs
PortfolioOptimisers.cross_sectional_forecast_refits
PortfolioOptimisers.cross_sectional_carries_history
PortfolioOptimisers.cross_sectional_forecast_history
PortfolioOptimisers.cross_sectional_carry_history
```
