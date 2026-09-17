```@meta
Description = "Orthogonal Uncertainty Sets, public API of PortfolioOptimisers.jl: AbstractOrthogonalScaling, BenchmarkWeightMetric, RegressionWeightMetric, …"
```

# Orthogonal Uncertainty Sets

```@docs
AbstractOrthogonalScaling
BenchmarkWeightMetric
RegressionWeightMetric
InverseIdiosyncraticVarianceMetric
IdentityMetric
IdentityScaling
IdiosyncraticVarianceScaling
OrthogonalUncertaintySet
orthogonal_scaling
ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
mu_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
