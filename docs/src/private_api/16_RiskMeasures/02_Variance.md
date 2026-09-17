```@meta
Description = "Variance, private API of PortfolioOptimisers.jl: NSkeQuadFormulations, QuadSecondMomentFormulations, SecondMomentFormulation, VarianceFormulation, …"
```

# Variance: private API

```@docs
NSkeQuadFormulations
QuadSecondMomentFormulations
SecondMomentFormulation
VarianceFormulation
resolve_deferred_quantities(r::Variance, pr::AbstractPriorResult)
resolve_deferred_quantities(r::StandardDeviation, pr::AbstractPriorResult)
resolve_deferred_quantities(r::UncertaintySetVariance, pr::AbstractPriorResult)
_no_bounds_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
_no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
ucs_risk_measure
ucs_variance
```
