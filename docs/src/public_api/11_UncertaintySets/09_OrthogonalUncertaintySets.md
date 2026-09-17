```@meta
Description = "Orthogonal Uncertainty Sets, public API of PortfolioOptimisers.jl: AbstractOrthogonalScaling, AbstractOrthogonalityMetric, BenchmarkWeightMetric, …"
```

# Orthogonal Uncertainty Sets

```@docs
AbstractOrthogonalScaling
AbstractOrthogonalityMetric
BenchmarkWeightMetric
RegressionWeightMetric
InverseIdiosyncraticVarianceMetric
IdentityMetric
IdentityScaling
IdiosyncraticVarianceScaling
OrthogonalUncertaintySet
orthogonal_scaling
orthogonality_weights
cs_diagnostic_weights
ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
mu_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
