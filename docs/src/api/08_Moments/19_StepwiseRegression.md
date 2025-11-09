# Stepwise Regression

```@docs
PValue
Forward
Backward
StepwiseRegression
regression(re::StepwiseRegression, X::MatNum, F::MatNum)
PortfolioOptimisers._regression(re::StepwiseRegression{<:PValue, <:Forward}, x::VecNum, F::MatNum)
PortfolioOptimisers._regression(re::StepwiseRegression{PortfolioOptimisers.AbstractMinMaxValStepwiseRegressionCriterion, <:Forward}, x::VecNum, F::MatNum)
PortfolioOptimisers._regression(re::StepwiseRegression{<:PValue, <:Backward}, x::VecNum, F::MatNum)
PortfolioOptimisers._regression(re::StepwiseRegression{<:PortfolioOptimisers.AbstractMinMaxValStepwiseRegressionCriterion, <:Backward}, x::VecNum, F::MatNum)
PortfolioOptimisers.add_best_feature_after_pval_failure!
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::VecNum, excluded::VecInt, included::VecInt, threshold::Number)
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::VecNum, excluded::VecInt, included::VecInt, threshold::Number)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::VecNum, included::VecInt, threshold::Number)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::VecNum, included::VecInt, threshold::Number)
```
