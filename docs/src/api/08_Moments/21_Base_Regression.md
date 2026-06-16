# [Regression](@id api-regression)

```@docs
LinearModel
factory(re::LinearModel, w::ObsWeights)
StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
GeneralisedLinearModel
factory(re::GeneralisedLinearModel, w::ObsWeights)
StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
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
RegE_Reg
AbstractRegressionAlgorithm
AbstractStepwiseRegressionAlgorithm
AbstractStepwiseRegressionCriterion
AbstractMinMaxValStepwiseRegressionCriterion
AbstractRegressionTarget
AbstractMinValStepwiseRegressionCriterion
AbstractMaxValStepwiseRegressionCriteria
port_opt_view(re::Regression, i, args...)
regression_criterion_func
regression_threshold
```
