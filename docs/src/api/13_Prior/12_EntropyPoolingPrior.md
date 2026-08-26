# Entropy Pooling

## Tail views

```@docs
LinearConditionalValueatRiskViewConstraint
IntegerConditionalValueatRiskViewConstraint
ConicEntropicValueatRiskViewConstraint
GridEntropicValueatRiskViewConstraint
ConicRelativisticValueatRiskViewConstraint
GridRelativisticValueatRiskViewConstraint
ep_evar
ep_evar_grid_row
ep_kappa_log
ep_rlvar_tail
ep_rlvar_shift
ep_rlvar
ep_rlvar_grid_row
ep_view_terms
ep_normalise_view_term
ep_view_formulations
ep_sbar
ep_assert_reachable_view
ep_cvar_formulation
ep_evar_formulation
ep_rlvar_formulation
ep_add_cvar_view!
ep_add_evar_view!
ep_add_rlvar_view!
ep_cvar_views!
ep_evar_views!
ep_rlvar_views!
add_ep_tail_view!
```

## Estimator

```@docs
EntropyPoolingPrior
VecEP
prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
      dims::Int = 1, strict::Bool = false, kwargs...)
ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
         strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
         F::Option{<:MatNum}; strict::Bool = false, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
