```@meta
Description = "Cross-Sectional Calibration Rules, public API of PortfolioOptimisers.jl: PrecisionBlend, SteinShrinkage, ForecastCalibrationSlope, …"
```

# Cross-Sectional Calibration Rules

The rules that compute the two calibration slots of a [`CrossSectionalFactorPrior`](@ref) from its fit. A rule of the spanned shrinkage stands in `lambda`, and a rule of the orthogonal forecast scale stands in `c`. [`PrecisionBlend`](@ref) is the default of `lambda`, and `c` stays one unless the caller states [`ForecastCalibrationSlope`](@ref).

```@docs
PrecisionBlend
SteinShrinkage
ForecastCalibrationSlope
AbstractForecastErrorAlgorithm
CurrentForecastError
ForecastHistoryError
spanned_forecast_sample
AbstractForecastScaleWarmUpAlgorithm
ThresholdWarmUp
PlugInWarmUp
PositivePartWarmUp
forecast_scale_warm_up
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
