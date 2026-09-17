```@meta
Description = "Orthogonal Uncertainty Sets, public API of PortfolioOptimisers.jl: BenchmarkWeightMetric, RegressionWeightMetric, InverseIdiosyncraticVarianceMetric, …"
```

# Orthogonal Uncertainty Sets

```@docs
BenchmarkWeightMetric
RegressionWeightMetric
InverseIdiosyncraticVarianceMetric
IdentityMetric
IdentityScaling
IdiosyncraticVarianceScaling
OrthogonalUncertaintySet
ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
mu_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
