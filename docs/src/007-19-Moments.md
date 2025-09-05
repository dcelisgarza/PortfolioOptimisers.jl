# Stepwise Regression

```@docs
PValue
PValue()
Forward
Backward
StepwiseRegression
StepwiseRegression()
PortfolioOptimisers.add_best_asset_after_failure_pval!
regression(re::StepwiseRegression{<:PValue, <:Forward},x::AbstractVector, F::AbstractMatrix)
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::AbstractVector, excluded::AbstractVector, included::AbstractVector, threshold::Real)
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::AbstractVector, excluded::AbstractVector, included::AbstractVector, threshold::Real)
regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Forward}, x::AbstractVector, F::AbstractMatrix)
regression(re::StepwiseRegression{<:PValue, <:Backward}, x::AbstractVector, F::AbstractMatrix)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::AbstractVector, included::AbstractVector, threshold::Real)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::AbstractVector, included::AbstractVector, threshold::Real)
regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Backward}, x::AbstractVector, F::AbstractMatrix)
regression(re::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)
```
