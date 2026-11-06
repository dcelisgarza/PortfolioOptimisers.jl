```@meta
Description = "Stepwise Regression, private API of PortfolioOptimisers.jl: _regression, add_best_factor_after_pval_failure!, get_forward_reg_incl_excl!, …"
```

# Stepwise Regression: private API

```@docs
_regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:ForwardSelection}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:PValue, <:BackwardElimination}, x::VecNum, F::MatNum)
_regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:BackwardElimination}, x::VecNum, F::MatNum)
add_best_factor_after_pval_failure!
get_forward_reg_incl_excl!
get_backward_reg_incl!
assert_stepwise_included
assert_stepwise_regression
assert_stepwise_regression_fields
stepwise_factor_set
stepwise_asset_fit!
pin_regression_choice(re::StepwiseRegression{<:Any, <:Any, <:Any, <:PinnedChoice}, X::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)
```
