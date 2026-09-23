```@meta
Description = "Net returns and drawdowns, public API of PortfolioOptimisers.jl: AbstractPreviousWeightsSource, SelfFinancingDrift, DriftedWeights, HeldWeightsResult, …"
```

# Net returns and drawdowns

The functions below compute the returns of a portfolio net of fees, the returns of each asset position, the turnover, the cumulative returns and the drawdowns. The types below state how the weights move between two rebalances.

```@docs
AbstractPreviousWeightsSource
SelfFinancingDrift
DriftedWeights
HeldWeightsResult
calc_net_returns(w::VecNum, X::MatNum, args...)
calc_net_returns(w::MatNum, X::MatNum, args...)
calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
calc_net_asset_returns
calc_turnover
cumulative_returns
drawdowns
```
