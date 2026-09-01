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
ep_rlvar_tail
ep_rlvar_shift
ep_rlvar
ep_rlvar_grid_row
ep_row_tilt
ep_evar_anchor
ep_evar_grid
ep_rlvar_anchor
ep_rlvar_grid
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
ep_add_grid_tail_view!
ep_add_rlvar_view!
ep_tail_view_prior_args
ep_assert_absolute_view
ep_normalise_absolute_view
ep_add_tail_view!
ep_tail_views!
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
