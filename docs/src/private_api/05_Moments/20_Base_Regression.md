```@meta
Description = "Base regression, private API of PortfolioOptimisers.jl: PSEUDO_R2_VARIANTS, ADJUSTED_PSEUDO_R2_VARIANTS, MIN_VAL_STEPWISE_REGRESSION_CRITERIA, …"
```

# Base regression: private API

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
MinValStepwiseRegressionCriterion
MaxValStepwiseRegressionCriterion
MinMaxValStepwiseRegressionCriterion
RegE_Reg
set_idiosyncratic_covariance(re::Regression, esigma::Option{<:VecNum_MatNum}, edof::Option{<:VecNum}, ediv::Option{<:VecNum})
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
