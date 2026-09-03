# [Cross-Sectional Weights](@id api-cross-sectional-weights)

## Types

```@docs
AbstractCrossSectionalWeightsAlgorithm
MarketCapWeights
BlendedInverseVarianceWeights
```

## Functions

```@docs
needs_second_pass
cs_weights_initial
cs_weights_refine
cross_sectional_cap_weights
cross_sectional_lagged_inverse_variance
cross_sectional_winsorise!
cross_sectional_median_cap!
```
