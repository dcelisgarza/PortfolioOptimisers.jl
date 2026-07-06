# Stepwise Regression

```@docs
PValue
ForwardSelection
BackwardElimination
StepwiseRegression
regression(re::StepwiseRegression, X::MatNum, F::MatNum)
_regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:AbstractMinMaxValStepwiseRegressionCriterion, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:PValue, <:BackwardElimination}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:AbstractMinMaxValStepwiseRegressionCriterion, <:BackwardElimination}, x::VecNum, F::MatNum)
add_best_feature_after_pval_failure!
get_forward_reg_incl_excl!(::AbstractMinValStepwiseRegressionCriterion, value::VecNum, excluded::VecInt, included::VecInt, t::Number)
get_forward_reg_incl_excl!(::AbstractMaxValStepwiseRegressionCriteria, value::VecNum, excluded::VecInt, included::VecInt, t::Number)
get_backward_reg_incl!(::AbstractMinValStepwiseRegressionCriterion, value::VecNum, included::VecInt, t::Number)
get_backward_reg_incl!(::AbstractMaxValStepwiseRegressionCriteria, value::VecNum, included::VecInt, t::Number)
```
