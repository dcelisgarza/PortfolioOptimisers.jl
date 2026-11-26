```@meta
Description = "A time-series regression fits an asset whose model leaves no residual on the intercept alone, and refuses an asset with fewer than two observations."
```

# [A time-series regression leaves a residual in each fit](@id migration-next-regression-saturated-fit)

## [Results that move](@id migration-next-regression-saturated-fit-numbers)

- **`StepwiseRegression` and `DimensionReductionRegression` fit an asset whose model leaves no residual degree of freedom on the intercept alone, with a warning that names the asset.** Such a saturated fit happens when the asset has no more observations in the fit than its model has parameters. The previous release gave loadings that interpolate the returns, `edof = 0`, and a residual variance of zero, with no message. A factor prior then read the asset as a pure factor bet. The asset now gets a row of zero loadings, the mean of its returns as the intercept, and a positive `edof`. It happens to an asset that lists late and has few returns, and to every asset on a short sample, such as the first steps of an online rule. A fit that leaves a residual does not move.
- **An asset with fewer than two finite and active returns raises an `IsNonFiniteError`** that names the asset and its count. The previous release fitted an asset with one return on the intercept alone, with `edof = 0`. **A sample of one observation raises an `ArgumentError`**, where `DimensionReductionRegression` raised a LAPACK error and `StepwiseRegression` fitted the intercept alone. Remove the asset from the sample, or fit over a longer window.
