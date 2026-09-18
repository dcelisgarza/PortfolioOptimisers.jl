```@meta
Description = "Regression, private API of PortfolioOptimisers.jl: PSEUDO_R2_VARIANTS, ADJUSTED_PSEUDO_R2_VARIANTS, MIN_VAL_STEPWISE_REGRESSION_CRITERIA, …"
```

# Regression: private API

```@docs
PSEUDO_R2_VARIANTS
ADJUSTED_PSEUDO_R2_VARIANTS
MIN_VAL_STEPWISE_REGRESSION_CRITERIA
MAX_VAL_STEPWISE_REGRESSION_CRITERIA
STEPWISE_REGRESSION_CRITERIA
AbstractRegressionEstimator
AbstractRegressionResult
AbstractLoadingsRegressionResult
AbstractCrossSectionalRegressionResult
AbstractFactorFamilyBasis
AbstractRegressionAlgorithm
AbstractStepwiseRegressionAlgorithm
AbstractStepwiseRegressionCriterion
AbstractRegressionTarget
MinValStepwiseRegressionCriterion
MaxValStepwiseRegressionCriterion
MinMaxValStepwiseRegressionCriterion
RegE_Reg
set_idiosyncratic_covariance(re::Regression, esigma::Option{<:VecNum_MatNum})
has_family_rebasis(rr::AbstractLoadingsRegressionResult)
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
