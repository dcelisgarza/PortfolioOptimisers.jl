# Regression

```@docs
PortfolioOptimisers.AbstractRegressionEstimator
PortfolioOptimisers.AbstractRegressionResult
PortfolioOptimisers.AbstractRegressionAlgorithm
PortfolioOptimisers.AbstractStepwiseRegressionAlgorithm
PortfolioOptimisers.AbstractStepwiseRegressionCriterion
PortfolioOptimisers.AbstractRegressionTarget
LinearModel
LinearModel()
PortfolioOptimisers.fit(target::LinearModel, X::AbstractMatrix, y::AbstractVector)
GeneralisedLinearModel
GeneralisedLinearModel()
PortfolioOptimisers.fit(target::GeneralisedLinearModel, X::AbstractMatrix, y::AbstractVector)
PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion
PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria
AIC
AICC
BIC
RSquared
AdjustedRSquared
PortfolioOptimisers.regression_criterion_func
Regression
Regression()
PortfolioOptimisers.regression_view
regression(re::Regression, args...)
regression(re::PortfolioOptimisers.AbstractRegressionEstimator, rd::ReturnsResult)
```
