# [Exponentially Weighted Mean Descriptors](@id api-ew-mean-descriptors)

## Types

```@docs
EWMean
EWVolumeRatio
DaysToCover
```

## Functions

```@docs
descriptor(de::EWMean, rd::ReturnsResult)
EWMomentum
EWShareTurnover
EWAmihudIlliquidity
half_life_decay
half_life_min_obs
assert_ew_decay
assert_ew_ratio_side
ew_ratio_values
ew_mean_series
```
