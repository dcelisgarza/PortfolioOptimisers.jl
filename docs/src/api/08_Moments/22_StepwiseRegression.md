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
get_forward_reg_incl_excl!(::MinValStepwiseRegressionCriterion, value::VecNum, excluded::VecInt, included::VecInt, t::Number)
get_forward_reg_incl_excl!(::MaxValStepwiseRegressionCriterion, value::VecNum, excluded::VecInt, included::VecInt, t::Number)
get_backward_reg_incl!(::MinValStepwiseRegressionCriterion, value::VecNum, included::VecInt, t::Number)
get_backward_reg_incl!(::MaxValStepwiseRegressionCriterion, value::VecNum, included::VecInt, t::Number)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
