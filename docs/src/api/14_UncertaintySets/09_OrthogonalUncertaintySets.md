# Orthogonal Uncertainty Sets

```@docs
AbstractOrthogonalityMetric
BenchmarkWeightMetric
RegressionWeightMetric
InverseIdiosyncraticVarianceMetric
IdentityMetric
AbstractOrthogonalScaling
IdentityScaling
IdiosyncraticVarianceScaling
orthogonality_weights
latest_orthogonality_weights
orthogonal_scaling
OrthogonalUncertaintySet
orthogonal_factor_span
orthogonal_mu_set
orthogonal_sigma_set
ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
mu_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
