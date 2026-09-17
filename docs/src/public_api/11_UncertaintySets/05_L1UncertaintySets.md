```@meta
Description = "ℓ1 Uncertainty Sets, public API of PortfolioOptimisers.jl: L1UncertaintySet, SignedL1UncertaintySet, L1UncertaintySetAlgorithm, …"
```

# ℓ1 Uncertainty Sets

```@docs
L1UncertaintySet
SignedL1UncertaintySet
L1UncertaintySetAlgorithm
SignedL1UncertaintySetAlgorithm
CharacteristicUncertaintySet
ActiveAssetsUncertaintyAlgorithm
port_opt_view(risk_ucs::L1UncertaintySet, i, args...)
port_opt_view(risk_ucs::SignedL1UncertaintySet, i, args...)
mu_ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
