```@meta
Description = "Online selection rules: the meta-learning rows, public API of PortfolioOptimisers.jl: SwitchingWeighting, SwitchingPortfolio, Ader, Sword."
```

# Online selection rules: the meta-learning rows

The rows of the fourth set built on the Expert Mixture's Gradient Point and start over the experts: the switching weighting of Singer (1997) and the switching portfolio it constructs over the single-asset constant rebalanced portfolios, and the constructors of the dynamic-regret mixtures over a geometric grid of first-order experts, the improved Ader of Zhang, Lu and Zhou (2018) and the small-loss Sword of Zhao, Zhang, Zhang and Zhou (2020).

```@docs
SwitchingWeighting
SwitchingPortfolio
Ader
Sword
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
