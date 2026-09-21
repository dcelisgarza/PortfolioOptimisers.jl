```@meta
Description = "Online selection rules: the adaptive subgradient and the optimistic step, public API of PortfolioOptimisers.jl: AdaptiveSubgradient, AbstractGradientPredictor, …"
```

# Online selection rules: the adaptive subgradient and the optimistic step

The adaptive and optimistic rows of the family's fourth set: the diagonal adaptive subgradient method, whose per-asset rate is the gradient mass each asset has accrued and whose projection is taken in the norm of that mass, a Projection Geometry of its own; and the optimistic mirror descent, a wrapper over the first-order rule that plays two half-steps a period, the second along a Gradient Predictor's hint, with the adaptive rate that reads the hint residuals.

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
