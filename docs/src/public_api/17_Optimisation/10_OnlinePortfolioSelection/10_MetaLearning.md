```@meta
Description = "Meta-learning, public API of PortfolioOptimisers.jl: SwitchingWeighting, SwitchingPortfolio, Ader, Sword."
```

# Meta-learning

This page has expert mixtures that follow a best portfolio which changes over time. `SwitchingWeighting` is the switching weighting of [singer1997](@citet), and `SwitchingPortfolio` is the expert mixture it builds over the constant rebalanced portfolios of single assets. `Ader` is the improved Ader of [zhang2018ader](@citet), and `Sword` is the small-loss Sword of [zhao2020sword](@citet). Both build mixtures of gradient projection experts over a geometric grid of step sizes. Their experts compute the gradient at the blended weights of the mixture, which the dynamic regret bounds of [zhang2018ader](@cite) and [zhao2020sword](@cite) assume.

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
