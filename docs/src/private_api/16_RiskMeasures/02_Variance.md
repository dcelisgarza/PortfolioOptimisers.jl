```@meta
Description = "Variance, private API of PortfolioOptimisers.jl: NSkeQuadFormulations, QuadSecondMomentFormulations, SecondMomentFormulation, VarianceFormulation, …"
```

# Variance: private API

```@docs
NSkeQuadFormulations
QuadSecondMomentFormulations
SecondMomentFormulation
VarianceFormulation
assert_lift_corner
resolve_deferred_quantities(r::Variance, pr::AbstractPriorResult)
resolve_deferred_quantities(r::StandardDeviation, pr::AbstractPriorResult)
resolve_deferred_quantities(r::UncertaintySetVariance, pr::AbstractPriorResult)
ucs_risk_measure
ucs_variance
PortfolioOptimisers.reader_frames(::UncertaintySetVariance, ::AbstractString, ::Bool)
```
