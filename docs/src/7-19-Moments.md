# Stepwise Regression

```@docs
PValue
Forward
Backward
StepwiseRegression
regression(::StepwiseRegression{<:PValue, <:Forward}, ::AbstractVector, ::AbstractMatrix)
regression(::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Forward}, ::AbstractVector, ::AbstractMatrix)
regression(::StepwiseRegression{<:PValue, <:Backward}, ::AbstractVector, ::AbstractMatrix)
regression(::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Backward}, ::AbstractVector, ::AbstractMatrix)
regression(::StepwiseRegression, ::AbstractMatrix, ::AbstractMatrix)
PortfolioOptimisers.add_best_asset_after_failure_pval!
PortfolioOptimisers.get_forward_reg_incl_excl!
PortfolioOptimisers.get_backward_reg_incl!
```
