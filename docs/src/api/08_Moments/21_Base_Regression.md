# [Regression](@id api-regression)

```@docs
LinearModel
factory(re::LinearModel, w::ObsWeights)
StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
GeneralisedLinearModel
factory(re::GeneralisedLinearModel, w::ObsWeights)
StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
PSEUDO_R2_VARIANTS
ADJUSTED_PSEUDO_R2_VARIANTS
MIN_VAL_STEPWISE_REGRESSION_CRITERIA
MAX_VAL_STEPWISE_REGRESSION_CRITERIA
STEPWISE_REGRESSION_CRITERIA
MinValStepwiseRegressionCriterion
MaxValStepwiseRegressionCriterion
MinMaxValStepwiseRegressionCriterion
Regression
regression(re::Regression, args...)
regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)
AbstractRegressionEstimator
AbstractTimeSeriesRegressionEstimator
AbstractCrossSectionalRegressionEstimator
AbstractRegressionResult
AbstractTimeSeriesRegressionResult
AbstractCrossSectionalRegressionResult
RegE_Reg
AbstractRegressionAlgorithm
AbstractStepwiseRegressionAlgorithm
AbstractStepwiseRegressionCriterion
AbstractRegressionTarget
port_opt_view(re::Regression, i, args...)
default_regression_criterion_variant
regression_criterion_func
regression_polarity
regression_threshold
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
