# ℓ1 Uncertainty Sets

```@docs
L1UncertaintySet
SignedL1UncertaintySet
L1UncertaintySetAlgorithm
SignedL1UncertaintySetAlgorithm
CharacteristicUncertaintySet
ActiveAssetsUncertaintyAlgorithm
l1_activation_ladder
l1_active_count
l1_eps_from_ladder
l1_resolve_eps
mu_ucs(ue::CharacteristicUncertaintySet{<:Any, <:L1UncertaintySetAlgorithm}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::CharacteristicUncertaintySet, X::MatNum,
             F::Option{<:MatNum} = nothing; kwargs...)
```
