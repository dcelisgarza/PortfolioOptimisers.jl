# [Exponentially Weighted Beta Descriptors](@id api-ew-beta-descriptors)

## Types

```@docs
EWBeta
EWMacroSensitivity
EWDownsideBeta
```

## Functions

```@docs
descriptor(de::EWBeta, rd::ReturnsResult)
descriptor(de::EWMacroSensitivity, rd::ReturnsResult)
descriptor(de::EWDownsideBeta, rd::ReturnsResult)
EWMarketBeta
ew_active_returns
ew_agg_series
ew_agg_vector
ew_beta_expand
ew_beta_output
ew_beta_residual_variance
ew_beta_group_prior
ew_beta_shrink
ew_masked_mean
ew_masked_weighted_mean
ew_macro_sensitivity_series
ew_downside_beta_series
assert_ew_agg_obs
assert_ew_shrinkage_bounds
```
