```@meta
Description = "Online selection rules: mirror descent, its schedules and its gradient transforms, public API of PortfolioOptimisers.jl: AbstractGradientTransform, …"
```

# Online selection rules: mirror descent, its schedules and its gradient transforms

`MirrorDescent` is the first-order rule. At each period it takes one mirror descent step on the gradient, in the divergence of its projection geometry. Each constructor is named after the method of one paper. `ExponentiatedGradient` uses the entropic geometry, and `GradientProjection` uses the Euclidean one. `EGE`, `EGR` and `EGA` are the exponentiated gradient with momentum, with root-mean-square scaling and with adaptive moments.

In place of a fixed step size `eta`, a rule can hold a learning-rate schedule. `InverseSquareRootRate` and `SelfConfidentRate` set the step size at each period. `DoublingTrickRate` restarts the rule at each stage. `WindowedBestRate` runs the exponentiated gradient at each rate of a set, and takes the rate with the most wealth over a window. `MAEG` and `AEG` are the two adaptive strategies of the paper that defines `WindowedBestRate`. A gradient transform, such as `GradientMomentum`, rescales the gradient before the step. The rule steps on log wealth, `LogWealth`, by default, or on a risk measure over the last rows, `RiskLoss`.

```@docs
PortfolioOptimisers.AbstractGradientTransform
InverseSquareRootRate
DoublingTrickRate
SelfConfidentRate
WindowedBestRate
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
MAEG
AEG
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
