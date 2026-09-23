```@meta
Description = "Online selection rules: the adaptive subgradient and the optimistic step, public API of PortfolioOptimisers.jl: AdaptiveSubgradient, …"
```

# Online selection rules: the adaptive subgradient and the optimistic step

This page has two first-order rules that adapt their steps. `AdaptiveSubgradient` is the diagonal adaptive subgradient method. The step size of each asset follows the gradients that asset has had so far, and the projection uses a norm weighted by those gradients. `OptimisticStep` is optimistic mirror descent. It wraps a `MirrorDescent` rule and takes two half-steps each period, the second along a guess of the next gradient. A gradient predictor makes the guess: `LastGradient`, `MeanGradient` or `ForecastGradient`. `HintResidualRate` is the adaptive step size that reads how far the guesses were from the real gradients.

```@docs
AdaptiveSubgradient
PortfolioOptimisers.AbstractGradientPredictor
LastGradient
MeanGradient
ForecastGradient
PortfolioOptimisers.predictor_state_seed
PortfolioOptimisers.predict_gradient!
HintResidualRate
OptimisticStep
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
