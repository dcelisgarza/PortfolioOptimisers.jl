```@meta
Description = "Cross-Sectional Forecast History Carry, private API of PortfolioOptimisers.jl: cross_sectional_forecast_refits, cross_sectional_carries_history, …"
```

# [Cross-Sectional Forecast History Carry: private API](@id private-api-cross-sectional-forecast-history-carry)

The carry fold of a [`CrossSectionalFactorPrior`](@ref) carries the return forecast history when a slot of the prior reads it. The functions below decide when the state carries the history, bring it up to the fitted observations at each step, and give it to the slots. The fifth function brings the standardised idiosyncratic returns, which the call with no data reads, up to the fitted observations by the same rule. The next three carry a Return Forecast that computes its history one observation at a time: its returns data over a set of panel rows, its Descriptor scores, and its rows. The last function brings every output that reads the fit up to the fitted observations.

```@docs
PortfolioOptimisers.cross_sectional_forecast_refits
PortfolioOptimisers.cross_sectional_carries_history
PortfolioOptimisers.cross_sectional_forecast_history
PortfolioOptimisers.cross_sectional_carry_history
PortfolioOptimisers.cross_sectional_carry_standardised
PortfolioOptimisers.cross_sectional_forecast_window
PortfolioOptimisers.cross_sectional_carry_scores
PortfolioOptimisers.cross_sectional_carry_forecast
PortfolioOptimisers.cross_sectional_carry_outputs
```
