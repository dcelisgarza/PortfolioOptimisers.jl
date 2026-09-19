```@meta
Description = "Online selection rules: the first set, public API of PortfolioOptimisers.jl: BuyAndHold, ConstantRebalancedPortfolio, NewtonStep, …"
```

# Online selection rules: the first set

The first algorithm set: the two benchmark rules, the online Newton step, passive aggressive mean reversion with its three slack rules, and the Expert Mixture whose wealth-weighted form over sampled constant rebalanced portfolios is the universal portfolio.

```@docs
BuyAndHold
ConstantRebalancedPortfolio
NewtonStep
PortfolioOptimisers.AbstractPassiveAggressiveSlack
NoSlack
LinearSlack
QuadraticSlack
PortfolioOptimisers.passive_aggressive_step
PassiveAggressiveMeanReversion
ExpertMixture
UniversalPortfolio
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
