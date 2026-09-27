```@meta
Description = "Online selection rules: the meta-learning rows, public API of PortfolioOptimisers.jl: SwitchingWeighting, SwitchingPortfolio, Ader, Sword."
```

# Online selection rules: the meta-learning rows

This page has expert mixtures that follow a best portfolio which changes over time. `SwitchingWeighting` is the switching weighting of Singer (1997), and `SwitchingPortfolio` is the expert mixture it builds over the constant rebalanced portfolios of single assets. `Ader` is the improved Ader of Zhang, Lu and Zhou (2018), and `Sword` is the small-loss Sword of Zhao, Zhang, Zhang and Zhou (2020). Both build mixtures of gradient projection experts over a geometric grid of step sizes. Their experts compute the gradient at the blended weights of the mixture, which the dynamic regret bounds of both papers assume.

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
