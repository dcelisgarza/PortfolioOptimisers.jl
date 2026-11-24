```@meta
Description = "Robust shrinkage, public API of PortfolioOptimisers.jl: AbstractRobustShrinkageIntensity, AbstractSubsetShrinkageIntensity, ConditionNumberIntensity, …"
```

# Robust shrinkage

```@docs
AbstractRobustShrinkageIntensity
AbstractSubsetShrinkageIntensity
ConditionNumberIntensity
SphericityIntensity
OracleIntensity
AsymptoticIntensity
AbstractTylerShrinkageAlgorithm
TraceNormalisedShrinkage
PenalisedShrinkage
MCDShrinkage
TylerShrinkage
port_opt_view(shr::Union{MCDShrinkage{<:MatNum}, TylerShrinkage{<:Any, <:MatNum}}, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
