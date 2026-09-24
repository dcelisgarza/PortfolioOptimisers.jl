```@meta
Description = "Entropy Pooling (a), private API of PortfolioOptimisers.jl: AbstractSequentialTailViewConstraint, LinearConditionalValueatRiskViewConstraint, …"
```

# Entropy Pooling (a): private API

```@docs
AbstractSequentialTailViewConstraint
LinearConditionalValueatRiskViewConstraint
IntegerConditionalValueatRiskViewConstraint
ConicEntropicValueatRiskViewConstraint
GridEntropicValueatRiskViewConstraint
ConicRelativisticValueatRiskViewConstraint
GridRelativisticValueatRiskViewConstraint
SequentialConditionalValueatRiskViewConstraint
SequentialEntropicValueatRiskViewConstraint
SequentialRelativisticValueatRiskViewConstraint
ep_tail_dual_block!
ep_var_multiplier
ep_tail_surrogate_row
ep_refine_tail_view(tv::AbstractSequentialTailViewConstraint, w::VecNum)
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
ep_sequential_start(tv::AbstractSequentialTailViewConstraint, w::VecNum)
add_ep_tail_view!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
