```@meta
Description = "Adaptive optimistic, public API of PortfolioOptimisers.jl: AdaptiveSubgradient, LastGradient, MeanGradient, ForecastGradient, predictor_state_seed, …"
```

# Adaptive optimistic

This page has the first-order rule `AdaptiveSubgradient` and the gradient predictors of optimistic mirror descent. `AdaptiveSubgradient` is the diagonal adaptive subgradient method. The step size of each asset follows the gradients that asset has had so far, and the projection uses a norm weighted by those gradients. A `MirrorDescent` rule with a gradient predictor is optimistic mirror descent. It takes two half-steps each period, the second along a guess of the next gradient. The predictor makes the guess: `LastGradient`, `MeanGradient` or `ForecastGradient`. `HintResidualRate` is the adaptive step size that reads how far the guesses were from the real gradients.

```@docs
AdaptiveSubgradient
LastGradient
MeanGradient
ForecastGradient
PortfolioOptimisers.predictor_state_seed
PortfolioOptimisers.predict_gradient!
PortfolioOptimisers.predictor_state_view
PortfolioOptimisers.copy_predictor_state
HintResidualRate
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
