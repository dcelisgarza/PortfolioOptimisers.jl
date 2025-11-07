# Regression

```@docs
LinearModel
PortfolioOptimisers.fit(target::LinearModel, X::NumMat, y::NumVec)
GeneralisedLinearModel
PortfolioOptimisers.fit(target::GeneralisedLinearModel, X::NumMat, y::NumVec)
AIC
AICC
BIC
RSquared
AdjustedRSquared
Regression
regression(re::Regression, args...)
regression(re::PortfolioOptimisers.AbstractRegressionEstimator, rd::ReturnsResult)
PortfolioOptimisers.AbstractRegressionEstimator
PortfolioOptimisers.AbstractRegressionResult
PortfolioOptimisers.AbstractRegressionAlgorithm
PortfolioOptimisers.AbstractStepwiseRegressionAlgorithm
PortfolioOptimisers.AbstractStepwiseRegressionCriterion
PortfolioOptimisers.AbstractRegressionTarget
PortfolioOptimisers.AbstractMinValStepwiseRegressionCriterion
PortfolioOptimisers.AbstractMaxValStepwiseRegressionCriteria
PortfolioOptimisers.regression_view
PortfolioOptimisers.regression_criterion_func
```
