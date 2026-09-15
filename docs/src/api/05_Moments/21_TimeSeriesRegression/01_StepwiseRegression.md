# Stepwise Regression

```@docs
PValue
ForwardSelection
BackwardElimination
StepwiseRegression
regression(re::StepwiseRegression, X::MatNum, F::MatNum)
_regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:PValue, <:BackwardElimination}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:BackwardElimination}, x::VecNum, F::MatNum)
add_best_factor_after_pval_failure!
get_forward_reg_incl_excl!
get_backward_reg_incl!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
