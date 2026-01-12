# [Regression](@id api-regression)

```@docs
LinearModel
fit(tgt::LinearModel, X::MatNum, y::VecNum)
GeneralisedLinearModel
fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
AIC
AICC
BIC
RSquared
AdjustedRSquared
Regression
regression(re::Regression, args...)
regression(re::AbstractRegressionEstimator, rd::ReturnsResult)
AbstractRegressionEstimator
AbstractRegressionResult
AbstractRegressionAlgorithm
AbstractStepwiseRegressionAlgorithm
AbstractStepwiseRegressionCriterion
AbstractMinMaxValStepwiseRegressionCriterion
AbstractRegressionTarget
AbstractMinValStepwiseRegressionCriterion
AbstractMaxValStepwiseRegressionCriteria
regression_view
regression_criterion_func
```
