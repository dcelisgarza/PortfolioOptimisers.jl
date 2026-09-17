```@meta
Description = "Net returns and drawdowns, public API of PortfolioOptimisers.jl: SelfFinancingDrift, DriftedWeights, HeldWeightsResult, calc_net_returns, …"
```

# Net returns and drawdowns

Net returns and drawdowns are two of the performance metrics of a portfolio. Here we define functions used to compute portfolio returns and related quantities.

```@docs
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
