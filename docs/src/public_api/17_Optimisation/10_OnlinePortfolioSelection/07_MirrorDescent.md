```@meta
Description = "Online selection rules: mirror descent, its schedules and its gradient transforms, public API of PortfolioOptimisers.jl: AbstractGradientTransform, …"
```

# Online selection rules: mirror descent, its schedules and its gradient transforms

The family's first-order rule, one mirror-descent step in the divergence its Projection Geometry holds, with the paper names as constructors: exponentiated gradient on the entropic map, gradient projection on the Euclidean one, and the three momentum variants; the Learning-Rate Schedules a rule may hold on `eta` in place of a number, one of which restarts the rule; and the Gradient Transforms that rescale the exponent before the step; and the objectives the rule steps on, log wealth or a Risk Loss over the head's rows.

```@docs
PortfolioOptimisers.AbstractGradientTransform
InverseSquareRootRate
DoublingTrickRate
SelfConfidentRate
PlainGradient
GradientMomentum
RootMeanSquareGradient
AdaptiveMomentGradient
PortfolioOptimisers.AbstractOnlineObjective
LogWealth
RiskLoss
MirrorDescent
PortfolioOptimisers.gradient_state_seed
PortfolioOptimisers.transform_gradient!
PortfolioOptimisers.gradient_state_view
PortfolioOptimisers.loss_gradient
PortfolioOptimisers.rows_needed(::LogWealth)
ExponentiatedGradient
GradientProjection
EGE
EGR
EGA
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
