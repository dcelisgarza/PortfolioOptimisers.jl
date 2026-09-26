```@meta
Description = "Online selection rules: the first set, public API of PortfolioOptimisers.jl: BuyAndHold, ConstantRebalancedPortfolio, NewtonStep, …"
```

# Online selection rules: the first set

This page has the two benchmark rules, the online Newton step, passive aggressive mean reversion and the expert mixture. `BuyAndHold` never trades after the start, so its weights drift with the market. `ConstantRebalancedPortfolio` trades back to the same weights after every price move. `NewtonStep` takes a second-order step on the log wealth. `PassiveAggressiveMeanReversion` moves the weights as little as possible, so that their return on the last price relatives is at most `eps`. `NoSlack`, `LinearSlack` and `QuadraticSlack` choose whether and how it caps the size of that step. `ExpertMixture` combines the weights of several rules, the experts, with a weighting that is itself a rule, and `OwnPoint` and `BlendPoint` choose where its first-order experts compute their gradient. `UniversalPortfolio` is the expert mixture that weights sampled constant rebalanced portfolios by their wealth.

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
OwnPoint
BlendPoint
ExpertMixture
UniversalPortfolio
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
