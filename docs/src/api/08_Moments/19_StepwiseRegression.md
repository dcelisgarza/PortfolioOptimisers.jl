# Stepwise Regression

```@docs
PValue
Forward
Backward
StepwiseRegression
regression(re::StepwiseRegression, X::NumMat, F::NumMat)
PortfolioOptimisers._regression(re::StepwiseRegression{<:PValue, <:Forward}, x::NumVec,
                    F::NumMat)
PortfolioOptimisers._regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Forward}, x::NumVec, F::NumMat)
PortfolioOptimisers._regression(re::StepwiseRegression{<:PValue, <:Backward}, x::NumVec,
                    F::NumMat)
PortfolioOptimisers._regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Backward}, x::NumVec,
                    F::NumMat)
PortfolioOptimisers.add_best_feature_after_pval_failure!
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::NumVec, excluded::IntVec, included::IntVec, threshold::Real)
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::NumVec, excluded::IntVec, included::IntVec, threshold::Real)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::NumVec, included::IntVec, threshold::Real)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::NumVec, included::IntVec, threshold::Real)
```
