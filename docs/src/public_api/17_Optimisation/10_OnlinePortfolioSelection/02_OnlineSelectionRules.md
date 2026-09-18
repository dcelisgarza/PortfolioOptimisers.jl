```@meta
Description = "Online selection rules: the first set, public API of PortfolioOptimisers.jl: BuyAndHold, ConstantRebalancedPortfolio, ExponentiatedGradient, NewtonStep, …"
```

# Online selection rules: the first set

The first algorithm set: the two benchmark rules, exponentiated gradient, the online Newton step, passive aggressive mean reversion with its three slack rules, the passive-aggressive step toward a Price Relative Forecast that the moving-average and robust-median reversions share, and the Expert Mixture whose wealth-weighted form over sampled constant rebalanced portfolios is the universal portfolio.

```@docs
BuyAndHold
ConstantRebalancedPortfolio
ExponentiatedGradient
NewtonStep
PortfolioOptimisers.AbstractPassiveAggressiveSlack
NoSlack
LinearSlack
QuadraticSlack
PortfolioOptimisers.passive_aggressive_step
PassiveAggressiveMeanReversion
ForecastReversion
MovingAverageReversion
RobustMedianReversion
ExpertMixture
UniversalPortfolio
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
