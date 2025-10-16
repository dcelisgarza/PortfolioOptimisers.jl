# Stepwise Regression

```@docs
PValue
Forward
Backward
StepwiseRegression
regression(re::StepwiseRegression{<:PValue, <:Forward}, x::AbstractVector,
                    F::AbstractMatrix)
regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Forward}, x::AbstractVector, F::AbstractMatrix)
regression(re::StepwiseRegression{<:PValue, <:Backward}, x::AbstractVector,
                    F::AbstractMatrix)
regression(re::StepwiseRegression{<:Union{<:PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, <:PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria}, <:Backward}, x::AbstractVector,
                    F::AbstractMatrix)
regression(re::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)
PortfolioOptimisers.add_best_feature_after_pval_failure!
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::AbstractVector, excluded::AbstractVector, included::AbstractVector, threshold::Real)
PortfolioOptimisers.get_forward_reg_incl_excl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::AbstractVector, excluded::AbstractVector, included::AbstractVector, threshold::Real)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion, value::AbstractVector, included::AbstractVector, threshold::Real)
PortfolioOptimisers.get_backward_reg_incl!(::PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria, value::AbstractVector, included::AbstractVector, threshold::Real)
```
