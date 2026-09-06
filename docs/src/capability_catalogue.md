# [Capability catalogue](@id capability-catalogue)

Everything `PortfolioOptimisers.jl` can do, grouped by the job it does rather
than by the file it lives in. Each entry links to its docstring.

This page is generated (see
[docs/generate_capability_catalogue.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_capability_catalogue.jl)):
the grouping is curated in `docs/capability_catalogue.jl`, and every description
is the first sentence of the corresponding docstring, so the two can never
disagree. A test asserts that every estimator and algorithm in the package
appears here, so the page cannot fall behind the code.

For the same types arranged by subtyping rather than by capability, see the
[type hierarchy](@ref type-hierarchy-AbstractEstimator).

## Core abstractions

Every component is an **Estimator** (a configuration encoding a method and its hyperparameters), an **Algorithm** (a behaviour selector consumed through an Estimator), or a **Result** (computed output). Estimators and Algorithms are what you choose; Results are what you get back.

Because every struct is immutable, runtime values are propagated down a composed estimator tree by rebuilding it.

- No-op factory function for constructing objects with a uniform interface. [`factory`](@ref)

## Preprocessing

- Convert `TimeSeries.TimeArray` price data to returns. [`prices_to_returns`](@ref) and [`ReturnsResult`](@ref)
- A container for aligned, time-indexed price-level data. [`PricesResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Preprocessing estimator converting price-level data into returns-level data. [`PricesToReturns`](@ref), [`fit_preprocessing`](@ref), and [`apply_preprocessing`](@ref)

```@raw html
</summary>
```

- Preprocessing estimator dropping assets and observations with excessive missing data from price-level data. [`MissingDataFilter`](@ref) and [`MissingDataFilterResult`](@ref)
- Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window. [`Imputer`](@ref) and [`ImputerResult`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Asset selection

```@raw html
</summary>
```

- Asset selector that scores every asset with a risk measure and keeps the assets a rule admits. [`ScoreSelector`](@ref), [`ZeroVarianceFilter`](@ref), and [`CompleteAssetSelector`](@ref)
- Asset selector that discards assets which duplicate information already carried by others. [`RedundancySelector`](@ref)
- Fitted result of any [`AbstractAssetSelector`](@ref). [`AssetSelectorResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Selection rules

```@raw html
</summary>
```

- [`RankRule`](@ref) with the tail sizes given as *fractions* of the asset universe. [`QuantileRule`](@ref)
- Take `best` and/or `worst` assets from the tails of the score ordering, then keep or drop them. [`RankRule`](@ref)
- Keep assets whose score falls strictly inside the band `(lo, hi)`. [`ThresholdRule`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Cut price- or returns-level data into a training window (the head) and a held-out test window (the tail). [`train_test_split`](@ref), [`TrainTestSplit`](@ref), and [`TrainTestSplitResult`](@ref)
- Return a `ReturnsResult` appropriate for benchmark-tracking optimisations. [`returns_result_picker`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Build the [`AssetPanel`](@ref) a carrier holds, from the raw, blank-carrying form of each Panel Field. [`asset_panel`](@ref), [`AssetPanel`](@ref), [`panel_field`](@ref), [`panel_feature_matrix`](@ref), [`feature_matrix`](@ref), [`feature_labels`](@ref), and [`panel_input`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Panel Field kinds

```@raw html
</summary>
```

- A Panel Field holding one number per asset, and per observation when it is time-varying. [`NumericPanelField`](@ref)
- A Panel Field holding one category label per asset, and per observation when it is time-varying. [`CategoricalPanelField`](@ref)
- A Panel Field whose trailing axis carries its own labels, and optionally its own groups. [`TensorPanelField`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Panel Field inputs

```@raw html
</summary>
```

- Raw form of a Panel Field holding one numeric quantity per observation and asset. [`NumericPanelInput`](@ref)
- Raw form of a Panel Field holding one category label per observation and asset. [`CategoricalPanelInput`](@ref)
- Raw form of a Panel Field whose third axis carries its own labels, and optionally its own groups. [`TensorPanelInput`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Blank-cell policies

```@raw html
</summary>
```

- Refuses a blank cell instead of resolving one. [`NoPanelFill`](@ref)
- Resolves every blank cell to one constant. [`ConstantPanelFill`](@ref)
- Resolves a blank cell to the nearest earlier observed value of the same asset. [`ForwardPanelFill`](@ref)
- Resolves a blank cell to the nearest later observed value of the same asset. [`BackwardPanelFill`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cross-sectional transforms

```@raw html
</summary>
```

A cross-sectional transform rescales one observation against the other assets of that same observation, through [`cross_sectional_transform`](@ref). The benchmark weights and the group labels are arguments of the call, and [`cross_sectional_groups`](@ref) derives the labels from a one-hot Panel Field.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Outlier treatments

```@raw html
</summary>
```

- Clips every value of an observation into the band between two percentiles of that observation's cross-section. [`CrossSectionalWinsoriser`](@ref), [`cross_sectional_transform`](@ref), and [`cross_sectional_groups`](@ref)
- Compresses every value of an observation towards the centre of that observation's cross-section, through a hyperbolic tangent. [`CrossSectionalTanhShrinker`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Scoring transforms

```@raw html
</summary>
```

- Scores every value of an observation as a cross-sectional z-score, optionally inside its own group first. [`CrossSectionalStandardiser`](@ref)
- Scores every value of an observation by the inverse normal of its cross-sectional percentile rank. [`CrossSectionalGaussianRank`](@ref)
- Scores every value of an observation by its percentile rank inside that observation's cross-section. [`CrossSectionalPercentileRank`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

## Matrix processing

- Projects a matrix to the nearest positive definite matrix, typically used for co-moment matrices. [`Posdef`](@ref), [`posdef!`](@ref), and [`posdef`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Configures and applies denoising algorithms to covariance or correlation matrices. [`Denoise`](@ref), [`denoise!`](@ref), and [`denoise`](@ref)

```@raw html
</summary>
```

- Denoises by setting the noise eigenvalues to zero. [`SpectralDenoise`](@ref)
- Denoises by replacing the noise eigenvalues with their own mean. [`FixedDenoise`](@ref)
- Denoises by shrinking the off-diagonal part of the noise block towards zero, keeping its diagonal whole. [`ShrunkDenoise`](@ref)

```@raw html
</details>
```

- Removes the largest `n` principal components (market modes) from a covariance or correlation matrix. [`Detone`](@ref), [`detone!`](@ref), and [`detone`](@ref)
- Configures and applies matrix processing routines. [`MatrixProcessing`](@ref), [`matrix_processing!`](@ref), [`matrix_processing_step!`](@ref), and [`matrix_processing`](@ref)

## Regression models

Factor prior models and implied volatility use [`regression`](@ref) in their estimation, which return a [`Regression`](@ref) object.

### Regression targets

- Fits each response by ordinary least squares through `GLM.LinearModel`. [`LinearModel`](@ref)
- Fits each response by a generalised linear model through `GLM.GeneralizedLinearModel`. [`GeneralisedLinearModel`](@ref)

### Regression types

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates a loadings matrix by selecting a factor subset per asset, one factor at a time. [`StepwiseRegression`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Algorithms

```@raw html
</summary>
```

- Grows the factor set from empty, adding the factor that most improves the criterion. [`ForwardSelection`](@ref)
- Shrinks the factor set from full, removing the factor whose removal most improves the criterion. [`BackwardElimination`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Selection criteria

```@raw html
</summary>
```

- Selects factors by the statistical significance of their coefficients. [`PValue`](@ref)
- `:aic`
- `:aicc`
- `:bic`
- `:r2`
- `:adjr2`

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates a loadings matrix by regressing each asset on the leading components of the factors. [`DimensionReductionRegression`](@ref)

```@raw html
</summary>
```

- Replaces the factors with the principal components of their standardised covariance. [`PCA`](@ref)
- Replaces the factors with the latent components of a Gaussian latent-variable model. [`PPCA`](@ref)

```@raw html
</details>
```

### Cross-sectional regression types

A cross-sectional regression fits one model per observation across the assets, and implements [`cross_sectional_regression`](@ref), which returns a [`CrossSectionalRegression`](@ref) object. [`cross_sectional_r2`](@ref) and [`mean_cross_sectional_r2`](@ref) score a fit.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Fits one weighted least squares per observation across the assets, in closed form. [`CrossSectionalLinearRegression`](@ref), [`cross_sectional_regression`](@ref), [`cross_sectional_r2`](@ref), and [`mean_cross_sectional_r2`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Rank deficiency policies

```@raw html
</summary>
```

- Solves the full-rank design directly and pseudo-inverts a rank-deficient one. [`PseudoInverseFallback`](@ref)
- Solves the full-rank design directly and refuses a rank-deficient one. [`RankDeficiencyRefusal`](@ref)
- Runs no rank test and takes whatever `\` returns. [`UncheckedSolve`](@ref)
- Always pseudo-inverts, so it runs no rank test and takes no threshold. [`MinimumNormSolve`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Fits one external regression model per observation across the assets. [`CrossSectionalTargetRegression`](@ref)

### Cross-sectional regression weight policies

A weight policy says what weight an asset carries in the cross-sectional fit of an observation. A one-pass policy reads the cross-section alone, and a two-pass policy reads the residuals of a first fit.

- Weights an asset by a power of its market capitalisation, in one pass. [`MarketCapWeights`](@ref)
- Blends market capitalisation weights with inverse idiosyncratic variance weights, in two passes. [`BlendedInverseVarianceWeights`](@ref)

### Cross-sectional regression diagnostics

A diagnostic of the cross-sectional fit reads the exposure history, the factor returns and the residuals, and answers a series over the observations or one value per factor. Every verb takes the lag-aligned histories as bare arrays, and takes a [`CrossSectionalFactorModel`](@ref) as well, which lags the exposures and maps them through a Factor Family Basis before it answers.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Return the weighted Gram history of a cross-sectional regression, one slice per observation. [`cs_gram`](@ref)

```@raw html
</summary>
```

- Return the variance inflation factor of every factor, one row per observation. [`exposure_vif`](@ref)
- Return the two-norm condition number of the cross-sectional design, one entry per observation. [`exposure_condition_number`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Fit quality.

```@raw html
</summary>
```

- Return the weighted cross-sectional coefficient of determination, one entry per observation. [`cs_regression_r2`](@ref)
- Return the cross-sectional coefficient of determination adjusted for the regressor count, one entry per observation. [`cs_regression_adjusted_r2`](@ref)
- Return the Akaike information criterion of every cross-sectional fit, one entry per observation. [`cs_regression_aic`](@ref)
- Return the Bayesian information criterion of every cross-sectional fit, one entry per observation. [`cs_regression_bic`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Significance of a factor return.

```@raw html
</summary>
```

- Return the t-statistic of every factor return, one row per observation. [`cs_regression_t_stats`](@ref)
- Return the fraction of observations at which a factor's t-statistic exceeds a threshold. [`cs_regression_t_stat_exceedance_rate`](@ref)

```@raw html
</details>
```

### Cross-sectional exposure diagnostics

A diagnostic of the exposure history reads the history as the Asset Panel wrote it, unlagged and on the raw factor axis, and answers a matrix over the factors, a series over the observations, or one value per factor. Every verb takes the history as bare arrays, and takes a [`CrossSectionalFactorModel`](@ref) as well, which resolves the cross-sectional weights from an [`AbstractOrthogonalityMetric`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Redundancy and turnover of an exposure.

```@raw html
</summary>
```

- Return the time-averaged correlation between every pair of factor exposures. [`exposure_correlation`](@ref)
- Return the stability of every factor exposure, one row per pair of observations. [`exposure_stability`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Reach and spread of an exposure.

```@raw html
</summary>
```

- Return the weighted cross-sectional standard deviation of every factor exposure, one row per observation. [`exposure_dispersion`](@ref)
- Return the coverage of every factor exposure, one entry per factor. [`exposure_coverage`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Forecasting power of an exposure.

```@raw html
</summary>
```

- Return the information coefficient of every factor exposure, one row per pair of observations. [`exposure_ic`](@ref)
- Return the summary of an information coefficient series, one entry per factor. [`exposure_ic_summary`](@ref)

```@raw html
</details>
```

### Cross-sectional idiosyncratic diagnostics

A diagnostic of the idiosyncratic returns divides each residual of the fit by the volatility the fit predicted for it, and reads the cross-section of the answer one observation at a time. Every verb takes the two histories as bare arrays, and takes a [`CrossSectionalFactorModel`](@ref) as well, which reads them off the block. Neither history carries a factor axis, so the group takes no lag and no family re-basis.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

The kernel every series of the group reads.

```@raw html
</summary>
```

- Return the standardised idiosyncratic returns of a cross-sectional fit. [`standardised_idio_returns`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Shape of the standardised cross-section.

```@raw html
</summary>
```

- Return the cross-sectional standard deviation of the standardised idiosyncratic returns, one entry per observation. [`idio_calibration`](@ref)
- Return the share of assets whose standardised idiosyncratic return exceeds a threshold, one entry per observation. [`idio_tail_rate`](@ref)
- Return the cross-sectional excess kurtosis of the standardised idiosyncratic returns, one entry per observation. [`idio_kurtosis`](@ref)
- Return the cross-sectional skewness of the standardised idiosyncratic returns, one entry per observation. [`idio_skewness`](@ref)
- Return the five time-aggregated numbers of the calibration of a cross-sectional fit. [`idio_calibration_summary`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Ranking power of the predicted volatility.

```@raw html
</summary>
```

- Return the information coefficient of the predicted idiosyncratic volatility, one entry per pair of observations. [`idio_vol_ic`](@ref)
- Return the rank correlation of the predicted idiosyncratic volatility against the next observation's standardised absolute idiosyncratic return, one entry per pair of observations. [`idio_vol_residual_dependence`](@ref)

```@raw html
</details>
```

### Factor model summary

The summary is the top of the diagnostic hierarchy. It calls one level-2 verb of the regression group and one of the exposure group per column, aggregates each series over the observations, and answers on the raw factor axis. A column that reads the exposure history is absent as a whole when the block carries none.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Summarise every factor of a cross-sectional factor model as a [`FactorSummaryResult`](@ref). [`factor_model_summary`](@ref)

```@raw html
</summary>
```

- The headline statistics of every factor of a cross-sectional factor model. [`FactorSummaryResult`](@ref)

```@raw html
</details>
```

### Descriptors

A Descriptor Estimator maps the Panel Fields of an Asset Panel to one value per observation and asset through [`descriptor`](@ref). Every named Descriptor is a constructor function that fixes the Panel Fields, the window, or the half-life of one archetype, and each accepts a keyword that overrides what it fixes.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Divides one Panel Field, or a combination of Panel Fields, by another at every observation. [`PanelFieldRatio`](@ref)

```@raw html
</summary>
```

- Book equity over market capitalisation, the value Descriptor. [`BookToPrice`](@ref)
- Trailing operating cash flow over market capitalisation, a value Descriptor. [`CashFlowToPrice`](@ref)
- Trailing sales over market capitalisation, a value Descriptor. [`SalesToPrice`](@ref)
- Trailing net income over market capitalisation, the earnings yield Descriptor. [`EarningsToPrice`](@ref)
- Forward earnings per share over the adjusted close, the forward earnings yield Descriptor. [`ForwardEarningsToPrice`](@ref)
- Trailing EBITDA over enterprise value, an earnings yield Descriptor that is neutral to the capital structure. [`EbitdaToEnterpriseValue`](@ref)
- Trailing common dividends over market capitalisation, the dividend yield Descriptor. [`DividendToPrice`](@ref)
- Forward dividends per share over the adjusted close, the forward dividend yield Descriptor. [`ForwardDividendToPrice`](@ref)
- Trailing dividends plus net buybacks over market capitalisation, the total payout Descriptor. [`ShareholderYield`](@ref)
- Total debt over total book capital, the book leverage Descriptor. [`BookLeverage`](@ref)
- Total debt over total market capital, the market leverage Descriptor. [`MarketLeverage`](@ref)
- Total debt over total assets, a leverage Descriptor. [`DebtToAssets`](@ref)
- Gross profit over total assets, the gross profitability Descriptor. [`GrossProfitability`](@ref)
- Gross profit over sales, the gross margin Descriptor. [`GrossMargin`](@ref)
- Trailing net income over total assets, the return on assets Descriptor. [`ReturnOnAssets`](@ref)
- Trailing net income over book equity, the return on equity Descriptor. [`ReturnOnEquity`](@ref)
- Trailing sales over total assets, the asset turnover Descriptor. [`AssetTurnover`](@ref)
- Trailing operating cash flow over total assets, a profitability Descriptor. [`CashFlowToAssets`](@ref)
- Trailing sales over enterprise value, a profitability Descriptor that is neutral to the capital structure. [`SalesToEnterpriseValue`](@ref)
- Accruals over total assets, the earnings quality Descriptor. [`AccrualsCashFlow`](@ref)
- Dispersion of the forward earnings estimates over the adjusted close, an earnings quality Descriptor. [`AnalystDispersionToPrice`](@ref)
- Shares sold short over shares outstanding, the short interest Descriptor. [`ShortInterest`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Takes the natural logarithm of one Panel Field at every observation. [`PanelFieldLog`](@ref)

```@raw html
</summary>
```

- Natural logarithm of the market capitalisation, the size Descriptor. [`LogMarketCap`](@ref)

```@raw html
</details>
```

- Returns one numeric Panel Field unchanged, as a Descriptor. [`Passthrough`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Growth of a non-negative Panel Field over a fixed lag, at every observation. [`GrowthRate`](@ref)

```@raw html
</summary>
```

- Growth of total assets over one year, the investment Descriptor. [`AssetsGrowthRate`](@ref)
- Growth of trailing sales over one year, the growth Descriptor. [`SalesGrowthRate`](@ref)
- Growth of the split-adjusted share count over one year, the net issuance Descriptor. [`IssuanceGrowthRate`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Change of a Panel Field over a fixed lag, scaled by the current value of a second Panel Field. [`ChangeToScale`](@ref)

```@raw html
</summary>
```

- Change of trailing net income over one year, divided by the current market capitalisation. [`EarningsChangeToPrice`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Change of the ratio of two Panel Fields over a fixed lag. [`ChangeInIntensity`](@ref)

```@raw html
</summary>
```

- Change of the capital expenditure to total assets ratio over one year. [`CapexToAssetsChangeInIntensity`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Exponentially weighted mean of the log returns, at every observation, with an optional skip. [`EWMean`](@ref)

```@raw html
</summary>
```

- Exponentially weighted mean of the log returns of the past year, less the past month. [`EWMomentum`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Exponentially weighted mean of a ratio of Panel Fields, at every observation. [`EWVolumeRatio`](@ref)

```@raw html
</summary>
```

- Exponentially weighted share turnover, the fraction of the shares outstanding that changes hands. [`EWShareTurnover`](@ref)
- Exponentially weighted price impact, the absolute return earned per unit of traded amount. [`EWAmihudIlliquidity`](@ref)

```@raw html
</details>
```

- Ratio of a Panel Field to the exponentially weighted mean of a second one, at every observation. [`DaysToCover`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Exponentially weighted volatility of the returns, at every observation. [`EWVolatility`](@ref)

```@raw html
</summary>
```

- Exponentially weighted volatility of the returns that fall short of a minimum acceptable return. [`EWDownsideVolatility`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Exponentially weighted volatility of the market-model residual, at every observation. [`EWResidualVolatility`](@ref)

```@raw html
</summary>
```

- Exponentially weighted volatility of the market-model residuals that fall short of a minimum acceptable return. [`EWResidualDownsideVolatility`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Exponentially weighted beta of the returns against the market return, at every observation. [`EWBeta`](@ref)

```@raw html
</summary>
```

- Exponentially weighted sensitivity of an asset to the market portfolio. [`EWMarketBeta`](@ref)

```@raw html
</details>
```

- Exponentially weighted sensitivity of the returns to a reference series, after the market is removed. [`EWMacroSensitivity`](@ref)
- Exponentially weighted sensitivity of the returns to the falls of the market, at every observation. [`EWDownsideBeta`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Sum of log returns over a fixed window that ends a fixed number of observations back, at every observation. [`RollingLogReturn`](@ref)

```@raw html
</summary>
```

- Sum of log returns over one year, ending one month back. [`RollingMomentum`](@ref)
- Negated sum of log returns over one month, ending at the current observation. [`Reversal`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Maximum return over a fixed trailing window, at every observation. [`RollingMax`](@ref)

```@raw html
</summary>
```

- Maximum return over one month. [`MaxReturn`](@ref)

```@raw html
</details>
```

### Factor exposures

An Exposure Estimator maps Descriptors, one categorical Panel Field, or nothing at all, to one asset loading per observation through [`factor_exposure`](@ref). A member producing one factor returns an observations by assets matrix, and the one-hot member returns one factor per level of its Panel Field.

- A Factor Exposure that is a fixed weighted combination of Descriptors. [`CompositeExposure`](@ref)
- A Factor Exposure derived from the Factor Exposure of another factor. [`DerivedExposure`](@ref)
- A Factor Exposure that expands one categorical Panel Field into one factor per level. [`OneHotExposure`](@ref)
- A Factor Exposure equal to one for every asset at every observation. [`ConstantExposure`](@ref)

### Factor family basis

A Factor Family whose one-hot exposures are collinear with a global factor is re-based before the fit. [`factor_family_basis`](@ref) drops one member per family and returns the compact, time-varying change of basis the fit runs in, and the reduced factor returns expand back to the named raw ones.

- Compact change of basis between the raw factor axis and the reduced axis a re-based Factor Family is fitted in. [`FactorFamilyBasis`](@ref) and [`factor_family_basis`](@ref)

### Return forecasts

A Return Forecast Estimator maps Descriptor Scores and a fitted factor-model block, or a stated vector, to one forecast per asset through [`return_forecast`](@ref). [`descriptor_scores`](@ref) is the recipe every fitted member starts from: each Descriptor is winsorised, standardised, optionally neutralised against named Factor Exposures, and standardised again. The Forecast Unit says what the Descriptors forecast, and the member converts the answer to return units.

- The shared recipe that turns Descriptors into cross-sectional scores. [`DescriptorScores`](@ref) and [`descriptor_scores`](@ref)
- A Return Forecast the caller states outright. [`CustomValueReturnForecast`](@ref)
- A Return Forecast that is a fixed signed combination of Descriptor scores. [`FixedWeightedReturnForecast`](@ref)
- A Return Forecast whose Descriptor weights are fitted by exponentially weighted least squares. [`ExpWeightedReturnForecast`](@ref)
- A Return Forecast fitted by a regression target over every observation and asset at once. [`TargetReturnForecast`](@ref)
- The Descriptors forecast the idiosyncratic return itself. [`IdiosyncraticReturnUnit`](@ref)
- The Descriptors forecast the idiosyncratic return divided by the idiosyncratic volatility. [`IdiosyncraticSharpeUnit`](@ref)

## Moment estimation

### Expected returns

Overloads `Statistics.mean`.

- Computes the expected returns as the sample mean of the asset returns. [`SimpleExpectedReturns`](@ref)
- Computes the expected excess returns that a set of equilibrium weights implies, by reverse optimisation. [`EquilibriumExpectedReturns`](@ref)
- Subtracts a risk-free rate from the expected returns that a nested estimator computes. [`ExcessExpectedReturns`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Shrinks the sample expected returns toward a target chosen by the shrinkage algorithm. [`ShrunkExpectedReturns`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Algorithms

```@raw html
</summary>
```

- James-Stein [`JamesStein`](@ref)
- Bayes-Stein [`BayesStein`](@ref)
- Bodnar-Okhrin-Parolya [`BodnarOkhrinParolya`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Targets: all algorithms can have any of the following targets

```@raw html
</summary>
```

- Grand Mean [`GrandMean`](@ref)
- Volatility Weighted [`VolatilityWeighted`](@ref)
- Mean Squared Error [`MeanSquaredError`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Expected returns estimator that returns the asset standard deviations. [`StandardDeviationExpectedReturns`](@ref)
- Expected returns estimator that returns the asset variances. [`VarianceExpectedReturns`](@ref)
- Computes the expected returns as the per-asset median of the asset returns. [`MedianExpectedReturns`](@ref)
- Returns a caller-supplied value for each asset instead of estimating one from the data. [`CustomValueExpectedReturns`](@ref)
- Expected returns estimator that restricts computation to a rolling or indexed observation window. [`WindowedExpectedReturns`](@ref)

### Variance and standard deviation

Overloads `Statistics.var` and `Statistics.std`.

- Computes the marginal variance and standard deviation, optionally weighted and optionally bias-corrected. [`SimpleVariance`](@ref)
- Variance estimator that restricts computation to a rolling or indexed observation window. [`WindowedVariance`](@ref)

### Covariance and correlation

Overloads `Statistics.cov` and `Statistics.cor`.

- Adapts any `StatsBase.CovarianceEstimator` to the library's calling convention, carrying its observation weights alongside it. [`GeneralCovariance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates the covariance matrix of asset returns from a centring estimator, a covariance estimator, and a moment algorithm. [`Covariance`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Configures and applies Gerber covariance estimators. [`GerberCovariance`](@ref)

```@raw html
</summary>
```

- Normalises the net co-movement vote by the observations on which both assets crossed their threshold. [`Gerber0`](@ref)
- Normalises the net co-movement vote by every observation on which at least one asset crossed its threshold. [`Gerber1`](@ref)
- Normalises the raw net co-movement vote by the geometric mean of its own diagonal. [`Gerber2`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Configures and applies Smyth-Broby covariance estimators. [`SmythBrobyCovariance`](@ref)

```@raw html
</summary>
```

- Divides the difference of the concordant and discordant Smyth-Broby contribution sums by their sum. [`SmythBroby0`](@ref)
- Divides the difference of the concordant and discordant Smyth-Broby contribution sums by their sum plus the neutral sum. [`SmythBroby1`](@ref)
- Normalises the net Smyth-Broby contribution of a pair by the geometric mean of its own diagonal. [`SmythBroby2`](@ref)
- Weights each Smyth-Broby contribution sum by its own observation count, then divides the difference by the sum. [`SmythBrobyGerber0`](@ref)
- Weights each Smyth-Broby contribution sum by its own count, then divides the difference by the sum plus the neutral term. [`SmythBrobyGerber1`](@ref)
- Weights each Smyth-Broby contribution sum by its own count, then normalises the net score by the geometric mean of its own diagonal. [`SmythBrobyGerber2`](@ref)
- Counts concordant and discordant observations, discards the contribution sums, and divides their difference by their sum. [`SmythBrobyCount0`](@ref)
- Counts concordant, discordant and neutral observations, discards the contribution sums, and divides the net count by the total. [`SmythBrobyCount1`](@ref)
- Counts concordant and discordant observations, discards the contribution sums, and normalises the net count by the geometric mean of its own diagonal. [`SmythBrobyCount2`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Gerber Information Quality [`GerberIQCovariance`](@ref) with custom variance, demeaning, temporal decay and numerator + denominator estimators

```@raw html
</summary>
```

- Implements the basic Gerber IQ covariance template. [`BasicGerberIQ`](@ref)
- Gerber Information Quality template with asymmetric thresholds. [`PartialGerberIQ`](@ref)
- Gerber Information Quality template with fine-grained asymmetric thresholds. [`FullGerberIQ`](@ref)
- Exponential Gerber IQ temporal decay. [`ExpGerberIQDecay`](@ref)
- Scales the threshold parameters using the individual asset volatilities. [`AssetVolatilityGerberIQScaler`](@ref)

```@raw html
</details>
```

- Measures linear and non-linear codependence from doubly-centred pairwise distance matrices. [`DistanceCovariance`](@ref)
- Measures co-movement in the lower tail: the share of an asset's worst returns that fall on the same dates as another's. [`LowerTailDependenceCovariance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Rank covariances

```@raw html
</summary>
```

- Measures monotonic association with Kendall's tau, counting concordant against discordant pairs. [`KendallCovariance`](@ref)
- Measures monotonic association with Spearman's rho, the Pearson correlation of the rank-transformed returns. [`SpearmanCovariance`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Measures codependence with mutual information, which captures a non-linear relationship a correlation misses. [`MutualInfoCovariance`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Abstract supertype for all histogram binning algorithms based on a bin width selection rule. [`BinWidthBins`](@ref)

```@raw html
</summary>
```

- Knuth's optimal bin width [`Knuth`](@ref)
- Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
- Scott's bin width [`Scott`](@ref)

```@raw html
</details>
```

- Histogram binning algorithm using the Hacine-Gharbi–Ravier rule. [`HacineGharbiRavier`](@ref)
- Predefined number of bins

```@raw html
</details>
```

- Convenience constructor. [`DenoiseCovariance`](@ref)
- Convenience constructor. [`DetoneCovariance`](@ref)
- Convenience constructor. [`ProcessedCovariance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Covariance estimator based on implied volatility scaling. [`ImpliedVolatility`](@ref)

```@raw html
</summary>
```

- Implied volatility algorithm that divides the latest implied volatility by a volatility risk premium adjustment. [`ImpliedVolatilityPremium`](@ref)
- Implied volatility algorithm that predicts realised volatility via regression on implied volatility. [`ImpliedVolatilityRegression`](@ref)

```@raw html
</details>
```

- Runs any covariance estimator, then applies a matrix post-processing step to its result. [`PortfolioOptimisersCovariance`](@ref)
- Answers both `cov` and `cor` with the wrapped estimator's correlation matrix. [`CorrelationCovariance`](@ref)
- Covariance estimator that restricts computation to a rolling or indexed observation window. [`WindowedCovariance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Regime-adjusted covariance and variance

```@raw html
</summary>
```

- Online exponentially weighted covariance estimator with regime-state adjustment. [`RegimeAdjustedExpWeightedCovariance`](@ref)
- Online exponentially weighted variance estimator with regime-state adjustment. [`RegimeAdjustedExpWeightedVariance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Regime adjustment methods

```@raw html
</summary>
```

- Regime adjustment method that scales variance by the ratio of the mean absolute deviation of standardised returns to the first-moment normalisation constant `x`. [`FirstMomentRegimeAdjusted`](@ref)
- Regime adjustment method that scales variance exponentially with the smoothed log-deviation of standardised squared returns from its expected value under stationarity. [`LogRegimeAdjusted`](@ref)
- Regime adjustment method that scales variance by the square root of the mean of the standardised squared returns. [`RootMeanSquaredAdjusted`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Shrinkage targets

```@raw html
</summary>
```

- Regime-adjustment target that uses a diagonal baseline covariance structure. [`DiagonalTarget`](@ref)
- Regime-adjustment target that uses a Mahalanobis-distance-based baseline covariance structure. [`MahalanobisTarget`](@ref)
- Regime-adjustment target that uses a portfolio-weighted baseline covariance structure. [`PortfolioTarget`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Demeaning

```@raw html
</summary>
```

- Centres the returns series using the (weighted) mean before computing the Median Absolute Deviation. [`MeanCentering`](@ref)
- Centres the returns series using the (weighted) median before computing the Median Absolute Deviation. [`MedianCentering`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Correlation smoothing

```@raw html
</summary>
```

- Greedy pairwise correlation pruning: drop assets until no surviving pair exceeds `t`. [`PairwiseCorrelation`](@ref)
- Group assets by connected component of the over-threshold correlation graph, and keep the best-scoring member of each. [`CorrelationComponents`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

### [Coskewness](@id catalogue-coskewness)

Implements [`coskewness`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates the coskewness tensor of a returns matrix, together with its negative spectral skewness matrix. [`Coskewness`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

- Coskewness estimator that restricts computation to a rolling or indexed observation window. [`WindowedCoskewness`](@ref)

### [Cokurtosis](@id catalogue-cokurtosis)

Implements [`cokurtosis`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates the square cokurtosis matrix of a returns matrix. [`Cokurtosis`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

- Cokurtosis estimator that restricts computation to a rolling or indexed observation window. [`WindowedCokurtosis`](@ref)

### Windowed moments

Every windowed estimator wraps a base moment estimator and recomputes it over a trailing window, so a moment can vary across the folds of a cross-validation scheme. The window is set by a fixed length or by a [`WindowSizeEstimator`](@ref).

- Abstract supertype for estimators that determine the rolling window size. [`WindowSizeEstimator`](@ref)

### Incremental fit

An estimator whose statistic has an exact update from its running state plus one new observation folds observations into that state, and its read-out verb answers from the state without reading the sample again. The state lives in the estimator's `cache` field, and ADR 0106 records why it is the one result an estimator holds. The sample mean, the sample variance and the full-moment sample covariance take part, and so does the `FullMoment` arm of [`Coskewness`](@ref) and of [`Cokurtosis`](@ref). The `SemiMoment` arm does not, because it clips against a centre that a new observation moves.

Two verbs fold, and ADR 0107 records what each promises. [`partial_fit!`](@ref) is the method each family writes, and it is that family's cheapest exact fold; it promises nothing about an estimator kept from before the call. [`partial_fit`](@ref) is one generic method with value semantics: it folds a copy of the state, so the estimator handed over reads what it read before.

- Folds observations into an estimator's partial-fit state, and returns the estimator. [`partial_fit!`](@ref)
- Folds observations into a copy of an estimator's partial-fit state, and returns the estimator that carries the copy. [`partial_fit`](@ref)

## Distance matrices

Implements [`distance`](@ref) and [`cor_and_dist`](@ref).

- Pairs a distance algorithm with an optional integer power, and applies it to a correlation matrix or to the data. [`Distance`](@ref)
- Measures how differently two assets relate to the whole universe, by applying a metric to a distance matrix. [`DistanceDistance`](@ref)

The distance estimators are used together with various distance matrix algorithms.

- Turns a signed correlation into a distance by $\sqrt{(1 - \rho) / 2}$. [`SimpleDistance`](@ref)
- Turns the magnitude of a correlation into a distance by $\sqrt{1 - \lvert\rho\rvert}$. [`SimpleAbsoluteDistance`](@ref)
- Turns the magnitude of a correlation into an unbounded distance by $-\log\lvert\rho\rvert$. [`LogDistance`](@ref)
- Turns a non-negative codependence into a distance by $\sqrt{1 - \rho}$, without halving. [`CorrelationDistance`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Measures the information one asset loses about another, from the entropies of a joint histogram. [`VariationInfoDistance`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Abstract supertype for all histogram binning algorithms based on a bin width selection rule. [`BinWidthBins`](@ref)

```@raw html
</summary>
```

- Knuth's optimal bin width [`Knuth`](@ref)
- Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
- Scott's bin width [`Scott`](@ref)

```@raw html
</details>
```

- Histogram binning algorithm using the Hacine-Gharbi–Ravier rule. [`HacineGharbiRavier`](@ref)
- Predefined number of bins

```@raw html
</details>
```

- Selects the distance algorithm that matches the covariance estimator it is given. [`CanonicalDistance`](@ref)

### Feature distances

A feature matrix describes assets by their exposures, memberships, loadings or adjacencies rather than by their returns, and can be turned into a distance matrix directly — no correlation matrix in between.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Turns a feature matrix into a distance matrix, by applying a metric to the rows of that matrix. [`FeatureDistance`](@ref)

```@raw html
</summary>
```

- Normalised angular distance metric. [`AngularDist`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Stack the Panel Fields a Feature Selector names into the Feature Matrix a distance measures. [`feature_matrix`](@ref) and [`feature_labels`](@ref)

```@raw html
</summary>
```

The Feature Matrix is derived from an [`AssetPanel`](@ref) and stored nowhere. [`feature_matrix`](@ref) stacks the Panel Fields the selector names, and [`feature_labels`](@ref) gives one label per column — a label vector is itself a selector, so it rebuilds the matrix the kernel measured.

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Asset Panel producers

```@raw html
</summary>
```

`ape` says which [`AssetPanel`](@ref) the metric measures. `nothing` reads the panel the data carrier holds; a producer builds a static one at the point of use, from the prior result and the returns of the subproblem that runs it, so a view passes it through and a fold refits it.

- Builds an Asset Panel holding the factor loadings the wrapped prior fitted. [`RegressionPanel`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Builds an Asset Panel holding a square proximity matrix graded from a graph or a partition. [`PhylogenyPanel`](@ref) and [`phylogeny_features`](@ref)

```@raw html
</summary>
```

Grades a graph neighbourhood into a square `assets × assets` proximity block, so the distance measures neighbourhood overlap. The one producer whose trailing axis *is* the asset axis; its source is always an estimator, so every fold and subproblem refits the graph on its own universe.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Phylogeny feature algorithm scoring each pair by how far apart it sits. [`Proximity`](@ref)

```@raw html
</summary>
```

Keeps the step count `phylogeny_matrix`'s clamp throws away, scoring each pair by how far apart it sits. `decay` shapes the fall-off and the source's `sep` truncates it -- two knobs, deliberately separate, because an exponential never reaches zero. Apart from [`NoDecay`](@ref) no decay emits zero inside the budget, so a zero entry means unreachable and nothing else.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Separations: the open [`AbstractSeparationAlgorithm`](@ref) family, applied by [`separation_matrix`](@ref) and [`separation_budget`](@ref)

```@raw html
</summary>
```

Carried by [`NetworkEstimator`](@ref) as `sep`. Says how far apart two assets sit *and* how far is too far, because the two share a unit. It sits on the network estimator rather than on the producer: every consumer of a network needs to know which pairs it relates, and the constraint path never sees the producer at all.

- Separation measured as the number of graph edges between two assets. [`HopCount`](@ref)
- [`PathLength`](@ref) sums the distances along the shortest path instead of counting its edges, and budgets in the distance estimator's units -- `dmax = nothing` means the observed diameter

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Budget rules: a callable in place of the budget number, resolved by [`resolve_separation`](@ref) once the data is in hand

```@raw html
</summary>
```

A budget cannot always be named in advance -- a cross-validation fold and a meta optimiser's subproblem each refit the graph. `HopCount(; n = ⋅)` takes a `HopCountAlgorithm` and `PathLength(; dmax = ⋅)` a `PathLengthAlgorithm`, each a callable struct; a bare `Function` is admitted in either field. The hop obligation is an `Integer`, checked at resolution because a functor's return type is not part of its signature. A rule changes *which* quantity stays put: a stated budget holds the radius still, a quantile rule holds the related-pair count still.

- [`HopCountQuantile`](@ref) places the hop budget at a quantile of the observed hop separations, rounded to a shell -- so it lands near the requested share rather than on it
- [`PathLengthQuantile`](@ref) does the same in distance units with no rounding, so it hits the requested share of related pairs -- which is how the radius ball's intermediate cardinalities become reachable by name

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Separation decays: the open [`AbstractSeparationDecayAlgorithm`](@ref) family, applied by [`separation_decay`](@ref)

```@raw html
</summary>
```

The argument is a *real* separation, so one family serves a hop count and any continuous separation alike. The contract -- `f(0) > 0` and maximal, monotone non-increasing, non-negative inside the budget, never assumed to reach zero -- is probed by a fail-safe fallback that the shipped members opt out of.

- Separation decay falling off linearly to the edge of the budget. [`LinearDecay`](@ref)
- Separation decay falling off exponentially. [`ExponentialDecay`](@ref)
- Separation decay falling off as a power of the separation. [`ReciprocalDecay`](@ref)
- [`NoDecay`](@ref) is the flat end of the dial, and *not* no truncation: the budget still cuts, so it yields the neighbourhood indicator

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Collapsing a window of time-varying features

```@raw html
</summary>
```

- Discards the window and measures the last observation's feature matrix alone. [`LastObservation`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Collapses the window to one `assets × features` matrix, then applies the metric once. [`AggregateFeatures`](@ref) and [`AggregateDistances`](@ref)

```@raw html
</summary>
```

- Aggregates along the observation axis with the possibly weighted arithmetic mean. [`MeanCollapse`](@ref)
- Aggregates along the observation axis with the possibly weighted median, which resists an outlying observation. [`MedianCollapse`](@ref)

```@raw html
</details>
```

- Concatenates the window into one long feature vector per asset, so nothing is averaged away. [`StackObservations`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

### Similarity matrices

Every similarity matrix algorithm is a pure transformation of a distance matrix, applied by [`distance_to_similarity`](@ref). [`FeatureDistance`](@ref) picks one from its metric via [`default_similarity`](@ref); the Planar Maximally Filtered Graph used by [`DBHT`](@ref) and [`LoGo`](@ref) takes its own.

The PMFG cannot take a negative weight, so it admits only the narrower [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) and refuses [`AngularSimilarity`](@ref) at construction. Two of the admitted members carry a domain precondition on the distance matrix, checked by [`assert_similarity_domain`](@ref): [`ComplementSimilarity`](@ref) needs `D <= 1`, and [`MaximumDistanceSimilarity`](@ref) needs a finite `D`.

- Takes the linear complement $1 - D$, the exact counterpart of a metric that is itself one minus a similarity. [`ComplementSimilarity`](@ref)
- Recovers a correlation from a normalised angular distance by $\cos(\pi D)$. [`AngularSimilarity`](@ref)
- Subtracts the squared distance from a ceiling placed above the largest squared distance. [`MaximumDistanceSimilarity`](@ref)
- Maps a distance of any magnitude into $(0,\,1]$ by $e^{-D}$. [`ExponentialSimilarity`](@ref)
- Applies $e^{-c D^{p}}$, adding a scale and an exponent to the exponential transformation. [`GeneralExponentialSimilarity`](@ref)

## Phylogeny

`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios.

### Clustering

Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`](@ref), [`Clusters`](@ref), and [`clusterise`](@ref). Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Decides how many clusters to cut a dendrogram or a partition into. [`OptimalNumberClusters`](@ref) and [`VectorToScalarMeasure`](@ref)

```@raw html
</summary>
```

- Picks the number of clusters at which the within-cluster dispersion curve bends most sharply. [`SecondOrderDifference`](@ref)
- Picks the number of clusters whose assets sit best inside their own cluster. [`SilhouetteScore`](@ref)
- Predefined number of clusters.
- Cut a dendrogram at the number of clusters `onc` selects. [`optimal_number_clusters`](@ref)

```@raw html
</details>
```

- Get the vector of cluster indices for each point. [`assignments`](@ref)

#### Hierarchical

- Builds a dendrogram by merging the two nearest clusters until one remains. [`HClustAlgorithm`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Direct Bubble Hierarchical Trees [`DBHT`](@ref) and Local Global sparsification of the covariance matrix [`LoGo`](@ref), [`logo!`](@ref), and [`logo`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Root selection

```@raw html
</summary>
```

- Takes one clique of the planar hierarchy as its single root. [`UniqueRoot`](@ref)
- Builds one root from the adjacency tree of every root candidate. [`EqualRoot`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

#### Non-hierarchical

Non-hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`](@ref) optimisations.

- Partitions assets into `k` groups by Lloyd's algorithm, with no dendrogram. [`KMeansAlgorithm`](@ref)

### Networks

#### Adjacency matrices

Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`](@ref) and [`PhylogenyResult`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Network adjacency [`NetworkEstimator`](@ref) with custom tree algorithms, covariance, and distance estimators

```@raw html
</summary>
```

- Grows the minimum spanning tree by taking the lightest edge that joins two components. [`KruskalTree`](@ref), [`BoruvkaTree`](@ref), and [`PrimTree`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Triangulated Maximally Filtered Graph with various similarity matrix estimators

```@raw html
</summary>
```

Any member of [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref): [`MaximumDistanceSimilarity`](@ref), [`ExponentialSimilarity`](@ref), [`GeneralExponentialSimilarity`](@ref) or [`ComplementSimilarity`](@ref). [`AngularSimilarity`](@ref) is refused, because a PMFG cannot take the negative weight it returns whenever a correlation is negative.

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Which pairs count as related: the `sep` separation

```@raw html
</summary>
```

[`HopCount`](@ref) gives the **hop ball**, every pair within `n` edges. [`PathLength`](@ref) gives the **radius ball**, every pair whose shortest path is no longer than `dmax` -- which buys the cardinalities *between* the hop shells, since a hop knob can only step whole neighbourhoods at a time. It does not re-rank: both are selected by distance to begin with, so a path length refines a hop count rather than rivalling it. Note that `PathLength()` with no `dmax` means the observed diameter and therefore relates every reachable pair, the opposite end of the dial from `HopCount()`'s default. Both reach [`SemiDefinitePhylogenyEstimator`](@ref) and [`IntegerPhylogenyEstimator`](@ref); [`NetworkClustersEstimator`](@ref) takes only the hop count, because its power sum is indexed by edges.

```@raw html
</details>
```

```@raw html
</details>
```

- Turns a return matrix into a clustering of the asset universe. [`ClustersEstimator`](@ref) and [`Clusters`](@ref)
- Clusters assets by the pseudo-distances that a network's structure induces. [`NetworkClustersEstimator`](@ref)
- Group assets by clustering them, and keep the best-scoring member of each cluster. [`ClusterGroups`](@ref)

#### Centrality and phylogeny measures

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Centrality estimator [`CentralityEstimator`](@ref) with custom adjacency matrix estimators (clustering and network) and centrality measures

```@raw html
</summary>
```

- Betweenness [`BetweennessCentrality`](@ref)
- Closeness [`ClosenessCentrality`](@ref)
- Degree [`DegreeCentrality`](@ref)
- Eigenvector [`EigenvectorCentrality`](@ref)
- Katz [`KatzCentrality`](@ref)
- Pagerank [`Pagerank`](@ref)
- Radiality [`RadialityCentrality`](@ref)
- Stress [`StressCentrality`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

The network is weighted where it can be, in the polarity [`centrality_polarity`](@ref) answers for the algorithm

```@raw html
</summary>
```

- [`DistancePolarity`](@ref) for the shortest-path algorithms -- betweenness, closeness, radiality and stress
- [`SimilarityPolarity`](@ref) for eigenvector centrality, which reads the weighted adjacency matrix itself
- [`TopologyOnly`](@ref) in the `ov` field of any of the five polarity-declaring algorithms, which withdraws the declaration and asks for the centrality over the network's topology alone

Five cases run on the plain unweighted graph and none of them raises: a weightless source (a [`ClustersEstimator`](@ref), a precomputed [`Clusters`](@ref), a precomputed [`PhylogenyResult`](@ref)), [`DegreeCentrality`](@ref), [`Pagerank`](@ref), [`KatzCentrality`](@ref), and [`EigenvectorCentrality`](@ref) on a tree branch. Polarity says *which* weights an algorithm receives, never *whether* the call succeeds. Note that the `sep` of a [`NetworkEstimator`](@ref) is inert on the weighted routes, which read the structure rather than the separation closure -- at the default `HopCount(; n = 1)` the two agree.

The override runs one way: it removes the weights and never supplies them, so there is no value that forces a polarity onto an algorithm. Every source honours it, because the topology-only answer is what a partition source, a precomputed [`PhylogenyResult`](@ref) and the tree branch already compute. Only the five that declare a polarity carry the field -- `DegreeCentrality(; ov = TopologyOnly())` is a `MethodError`, since the other three already read the topology alone. `ct` is positional on every centrality surface, so a configured algorithm reaches all of them, and a [`CentralityEstimator`](@ref) stays a pure bundle of `pl` and `ct`.

```@raw html
</details>
```

- Fallback no-op for returning a validated centrality vector result as-is. [`centrality_vector`](@ref)
- Compute the weighted average centrality for a network and centrality algorithm. [`average_centrality`](@ref)
- Compute the asset phylogeny score for a set of weights and a phylogeny matrix. [`asset_phylogeny`](@ref)

#### Cluster trees

Hierarchical clustering produces a tree of [`ClusterNode`](@ref)s, walked by [`to_tree`](@ref), [`pre_order`](@ref), and [`is_leaf`](@ref).

- Collects each leaf's `id`, which for a leaf is its asset index. [`PreorderTreeByID`](@ref)

## Optimisation constraints

Non clustering optimisers support a wide range of constraints, while naive and clustering optimisers only support weight bounds. Furthermore, entropy pooling prior supports a variety of views constraints. It is therefore important to provide users with the ability to generate constraints manually and/or programmatically. We therefore provide a wide, robust, and extensible range of types such as [`AbstractEstimatorValueAlgorithm`](@ref) and [`UniformValues`](@ref), and functions that make this easy, fast, and safe.

Constraints can be defined via their estimators or directly by their result types. Some using estimators need to map key-value pairs to the asset universe, this is done by defining the assets and asset groups in [`UniverseSets`](@ref). Internally, `PortfolioOptimisers.jl` uses all the information and calls [`name_to_val!`](@ref), and [`replace_group_by_assets`](@ref) to produce the appropriate arrays.

- Equation parsing [`parse_equation`](@ref) and [`ParsingResult`](@ref)
- No-op fallback for returning an existing `LinearConstraint` object, `nothing`, or a vector of them. [`linear_constraints`](@ref), [`LinearConstraintEstimator`](@ref), [`PartialLinearConstraint`](@ref), and [`LinearConstraint`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Factor exposure constraints [`ExposureConstraintEstimator`](@ref)

```@raw html
</summary>
```

Wraps whatever `lcse` already accepts and declares the [`AbstractConstraintSpace`](@ref) its rows are written in, so a mandate can be stated in factor names -- "at most 30% momentum" -- and re-based through the prior's loadings. The projection happens during constraint generation, so what reaches the optimiser is an ordinary asset-space [`LinearConstraint`](@ref) and every optimiser sharing [`JuMPOptimiser`](@ref) supports one. The names resolve against the factor axis a [`UniverseSets`](@ref) declares.

- The factor basis: a constraint written in factor names, re-based through a regression's loadings. [`FactorSpace`](@ref)

```@raw html
</details>
```

- No-op fallback for risk budget constraint generation. [`risk_budget_constraints`](@ref), [`RiskBudgetEstimator`](@ref), and [`RiskBudget`](@ref)
- Generate phylogeny-based portfolio constraints from an estimator or result. [`phylogeny_constraints`](@ref), [`centrality_constraints`](@ref), [`SemiDefinitePhylogenyEstimator`](@ref), [`SemiDefinitePhylogeny`](@ref), [`IntegerPhylogenyEstimator`](@ref), [`IntegerPhylogeny`](@ref), and [`CentralityConstraint`](@ref)
- Generate portfolio weight bounds constraints from a `WeightBoundsEstimator` and asset set. [`weight_bounds_constraints`](@ref), [`WeightBoundsEstimator`](@ref), and [`WeightBounds`](@ref)
- Declares the universes a portfolio problem is written against, and any groupings or partitions of them. [`UniverseSets`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Budget constraints [`BudgetEstimator`](@ref) and [`BudgetRange`](@ref)

```@raw html
</summary>
```

- Charges the portfolio budget for transaction costs that grow linearly with the traded volume. [`BudgetCosts`](@ref)
- Charges the portfolio budget and the return for market impact costs that follow an empirical power law. [`BudgetMarketImpact`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Constraint values [`AbstractEstimatorValueAlgorithm`](@ref)

```@raw html
</summary>
```

Where a constraint takes one value per asset or group, these algorithms say how to derive it from data rather than stating it outright.

- Fills every entry of a value vector with `1/N`, where `N` is the number of assets in the universe. [`UniformValues`](@ref)
- Return value for assets or groups, based on a mapping and asset sets. [`estimator_to_val`](@ref)
- Algorithm for reducing a vector of real values to its minimum. [`MinValue`](@ref)
- Algorithm for reducing a vector of real values to its maximum. [`MaxValue`](@ref)
- Algorithm for reducing a vector of real values to its optionally weighted mean. [`MeanValue`](@ref)
- Algorithm for reducing a vector of real values to its optionally weighted median. [`MedianValue`](@ref)
- Algorithm for reducing a vector of real values to its mode. [`ModeValue`](@ref)
- Algorithm for reducing a vector of real values to its sum. [`SumValue`](@ref)
- Algorithm for reducing a vector of real values to its product. [`ProdValue`](@ref)
- Algorithm for reducing a vector of real values to its optionally weighted standard deviation. [`StdValue`](@ref)
- Algorithm for reducing a vector of real values to its optionally weighted variance. [`VarValue`](@ref)
- Algorithm for reducing a vector of real values to its optionally weighted mean divided by its optionally weighted standard deviation. [`StandardisedValue`](@ref)
- States that no fold-less value exists. [`NoDefault`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Varies one optimiser input across the folds of a cross-validation scheme. [`TimeDependent`](@ref)

```@raw html
</summary>
```

A time-dependent input takes a different value in each fold of a cross-validation scheme, and is inert outside one.

- Abstract supertype for the callable structs used as time-dependent values. [`TimeDependentCallable`](@ref)
- Abstract supertype for callable structs whose per-fold value is a *constraint value*. [`TimeDependentConstraintCallable`](@ref)
- Abstract supertype for callable structs whose per-fold value is an *optimiser*. [`TimeDependentOptimiserCallable`](@ref)
- Describes one fold to the time-dependent constraints that resolve against it. [`TimeDependentContext`](@ref)
- Declares that a callable time-dependent entry requires the previous optimisation's weights. [`PreviousWeightsFunction`](@ref)

```@raw html
</details>
```

- Construct a binary asset-group membership matrix from asset set groupings. [`asset_sets_matrix`](@ref) and [`AssetSetsMatrixEstimator`](@ref)
- Propagate or pass through buy-in threshold portfolio constraints. [`threshold_constraints`](@ref), [`ThresholdEstimator`](@ref), and [`Threshold`](@ref)

## Prior statistics

Many optimisations and constraints use prior statistics computed via [`prior`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Carries the returns, mean and covariance a low order prior estimator produced. [`LowOrderPrior`](@ref)

```@raw html
</summary>
```

- Empirical prior estimator for asset returns. [`EmpiricalPrior`](@ref)
- Factor-based prior estimator for asset returns. [`FactorPrior`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Estimates a point-in-time cross-sectional factor model from an Asset Panel, and lifts it onto the assets. [`CrossSectionalFactorPrior`](@ref)

```@raw html
</summary>
```

The cross-sectional counterpart of [`FactorPrior`](@ref). It reads a point-in-time [`AssetPanel`](@ref) rather than a factor-return series, builds one Factor Exposure per named factor, regresses each observation's returns on the lagged exposures across the assets, and returns a [`CrossSectionalFactorModel`](@ref) in the `rr` slot. It composes the Descriptors, the Factor Exposures, the cross-sectional regression, its weight policy and the Factor Family Basis catalogued above.

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Black-Litterman

```@raw html
</summary>
```

- Unified interface for constructing or passing through Black-Litterman investor views. [`black_litterman_views`](@ref)
- Black-Litterman prior estimator for asset returns. [`BlackLittermanPrior`](@ref)
- Bayesian Black-Litterman prior estimator for asset returns. [`BayesianBlackLittermanPrior`](@ref)
- Factor Black-Litterman prior estimator for asset returns. [`FactorBlackLittermanPrior`](@ref)
- Augmented Black-Litterman prior estimator for asset returns. [`AugmentedBlackLittermanPrior`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Reweights the observations of a prior so that its moments and its tails meet a set of views. [`EntropyPoolingPrior`](@ref)

```@raw html
</summary>
```

Entropy pooling reweights the observations so that the posterior satisfies the stated views while staying as close as possible to the prior. Alongside the moment views it takes views on the conditional, entropic and relativistic value at risk, each written as constraints of the one entropy pooling problem.

- Container for Black-Litterman investor views in canonical matrix form. [`BlackLittermanViews`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

View constraint algorithms

```@raw html
</summary>
```

- Enforces every view in a single entropy pooling optimisation. [`H0_EntropyPooling`](@ref)
- Enforces the views in stages, and starts every stage from the prior probabilities. [`H1_EntropyPooling`](@ref)
- Enforces the views in stages, and starts every stage from the previous stage's probabilities. [`H2_EntropyPooling`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Tail view formulations

```@raw html
</summary>
```

- Linear formulation of a conditional value-at-risk view [EPTail](@cite). [`LinearConditionalValueatRiskView`](@ref)
- Integer formulation of a conditional value-at-risk view [EPTail](@cite). [`IntegerConditionalValueatRiskView`](@ref)
- Exponential cone formulation of an entropic value-at-risk view [EPTail](@cite). [`ConicEntropicValueatRiskView`](@ref)
- Grid formulation of an entropic value-at-risk view [EPTail](@cite). [`GridEntropicValueatRiskView`](@ref)
- Power cone formulation of a relativistic value-at-risk view [EPRLVaR](@cite). [`ConicRelativisticValueatRiskView`](@ref)
- Grid formulation of a relativistic value-at-risk view. [`GridRelativisticValueatRiskView`](@ref)
- Sequential convex formulation of a conditional value-at-risk view. [`SequentialConditionalValueatRiskView`](@ref)
- Sequential convex formulation of an entropic value-at-risk view. [`SequentialEntropicValueatRiskView`](@ref)
- Sequential convex formulation of a relativistic value-at-risk view. [`SequentialRelativisticValueatRiskView`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

View groups

```@raw html
</summary>
```

A significance level belongs to a view, not to the estimator holding it. These pair a group of view equations with the level, and for a tail view the formulation, they are read under, so one estimator can state views at several levels. A relativistic view group carries its deformation parameter on the same reasoning.

- A group of **value at risk** views, with the significance level they are read under. [`ValueatRiskView`](@ref)
- A group of **conditional value at risk** views, with the significance level and formulation they are read under. [`ConditionalValueatRiskView`](@ref)
- A group of **entropic value at risk** views, with the significance level and formulation they are read under. [`EntropicValueatRiskView`](@ref)
- A group of **relativistic value at risk** views, with the significance level, the deformation parameter and the formulation they are read under. [`RelativisticValueatRiskView`](@ref)
- Spans of the two searches that read a relativistic value at risk. [`RelativisticValueatRiskViewBracket`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Divergence formulations

```@raw html
</summary>
```

- Evaluates the entropy pooling objective through the exponential of the dual variables. [`ExpEntropyPooling`](@ref)
- Evaluates the entropy pooling objective in log space. [`LogEntropyPooling`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Optimisers

```@raw html
</summary>
```

- Solves the dual of the entropy pooling problem with Optim.jl. [`OptimEntropyPooling`](@ref)
- Solves the primal of the entropy pooling problem with JuMP.jl. [`JuMPEntropyPooling`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Reweights the observations of a prior so that its moments meet a set of views, and root-finds a CVaR view. [`MeucciEntropyPoolingPrior`](@ref)

```@raw html
</summary>
```

The earlier entropy pooling estimator, which hunts a conditional value at risk target with the recursive algorithm of Meucci, Ardia and Keel. It takes equality CVaR views alone, and re-solves the whole problem once per candidate value at risk level.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

View constraint algorithms

```@raw html
</summary>
```

- Enforces every view in a single entropy pooling optimisation. [`H0_EntropyPooling`](@ref)
- Enforces the views in stages, and starts every stage from the prior probabilities. [`H1_EntropyPooling`](@ref)
- Enforces the views in stages, and starts every stage from the previous stage's probabilities. [`H2_EntropyPooling`](@ref)
- Root-finds the value at risk level that meets a single conditional value-at-risk view. [`ConditionalValueatRiskEntropyPooling`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Opinion pooling prior estimator for asset returns. [`OpinionPoolingPrior`](@ref)

```@raw html
</summary>
```

- Pools the opinions as a weighted arithmetic mean of their scenario weights. [`LinearOpinionPooling`](@ref)
- Pools the opinions as a weighted geometric mean of their scenario weights, renormalised. [`LogarithmicOpinionPooling`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Carries the coskewness and cokurtosis a high order prior estimator produced, over the low order prior it wraps. [`HighOrderPrior`](@ref)

```@raw html
</summary>
```

- High order prior estimator for asset returns. [`HighOrderPriorEstimator`](@ref)
- Projects factor coskewness and cokurtosis onto the asset axis through the regression loadings. [`HighOrderFactorPriorEstimator`](@ref)

```@raw html
</details>
```

## Uncertainty sets

In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`](@ref), [`mu_ucs`](@ref), and [`sigma_ucs`](@ref).

`PortfolioOptimisers.jl` implements two types of uncertainty sets.

- Holds the element-wise lower and upper bounds of a box uncertainty set on a mean vector or on a covariance matrix. [`BoxUncertaintySet`](@ref) and [`BoxUncertaintySetAlgorithm`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

[`EllipsoidalUncertaintySet`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) with various algorithms for computing the scaling parameter via [`k_ucs`](@ref)

```@raw html
</summary>
```

- Fits the ellipsoid radius `k` empirically, as the `1 - q` quantile of the Mahalanobis distances of the sampled estimation errors. [`NormalKUncertaintyAlgorithm`](@ref)
- Computes the ellipsoid radius `k` as `sqrt((1 - q) / q)`, the closed form that holds for any distribution of the estimation errors. [`GeneralKUncertaintyAlgorithm`](@ref)
- Computes the ellipsoid radius `k` as the square root of the `1 - q` chi-squared quantile, the closed form that holds when the estimation errors are normal. [`ChiSqKUncertaintyAlgorithm`](@ref)
- Predefined scaling parameter

```@raw html
</details>
```

A third shape holds the worst-case variance of a covariance set as a quadratic penalty on the weights, so the optimisation stays a second-order cone programme and lifts no semidefinite block.

- Holds a worst-case variance penalty as a radius, a diagonal metric square root and a basis of the directions the penalty spares. [`CompactCovarianceUncertaintySet`](@ref)

A fourth shape is the image of a norm ball under a geometry map of any rank, on either axis, so a flat set on the directions a factor model does not span needs no full-rank shape matrix. A built ellipsoid converts into it with one Cholesky factorisation.

- [`NormBallUncertaintySet`](@ref) and [`NormBallUncertaintySetAlgorithm`](@ref), which take the same scaling algorithms as the ellipsoid and read them off the geometry map via [`k_norm_ball`](@ref)

It also implements various estimators for the uncertainty sets, the following two can generate box, ellipsoidal and norm-ball sets.

- Fits a box or an ellipsoidal uncertainty set from the sampling laws that normal returns imply: the mean is normal and the covariance is Wishart. [`NormalUncertaintySet`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Bootstrapping via Autoregressive Conditional Heteroscedasticity [`ARCHUncertaintySet`](@ref) via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)

```@raw html
</summary>
```

- Circular [`CircularBootstrap`](@ref)
- Moving [`MovingBootstrap`](@ref)
- Stationary [`StationaryBootstrap`](@ref)

```@raw html
</details>
```

The following estimator can only generate box sets.

- Fits a box uncertainty set by widening the prior statistics by a fixed fraction of their own absolute value. [`DeltaUncertaintySet`](@ref)

Quintile portfolios are expressed as an uncertainty set on the characteristic vector rather than as an optimiser of their own (ADR 0032).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Fits an $\ell_1$ uncertainty set on the characteristic vector, mean-only and with a calibrated radius. [`CharacteristicUncertaintySet`](@ref)

```@raw html
</summary>
```

- $\ell_1$ (cross-polytope) uncertainty set on the characteristic vector. [`L1UncertaintySet`](@ref) and [`L1UncertaintySetAlgorithm`](@ref)
- Signed $\ell_1$ uncertainty set on the characteristic vector, with a separate error budget per sign. [`SignedL1UncertaintySet`](@ref) and [`SignedL1UncertaintySetAlgorithm`](@ref)
- Calibrates the $\ell_1$ uncertainty radius to a target number of active assets. [`ActiveAssetsUncertaintyAlgorithm`](@ref)

```@raw html
</details>
```

One estimator reads no returns data at all. It is handed the fitted prior of the optimisation it serves, and confines both of its sets to the directions the prior's factor model does not span.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Fits both uncertainty sets from the factor model of the optimisation's own prior, confined to the directions the factors do not span. [`OrthogonalUncertaintySet`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Orthogonality metrics, the cross-sectional weighting the factor span is taken under

```@raw html
</summary>
```

- Names the inverse of the idiosyncratic variances as the cross-sectional weight source, the default. [`InverseIdiosyncraticVarianceMetric`](@ref)
- Names the regression weights as the cross-sectional weight source. [`RegressionWeightMetric`](@ref)
- Names the benchmark weights as the cross-sectional weight source. [`BenchmarkWeightMetric`](@ref)
- Names no weight source: every asset carries the same weight. [`IdentityMetric`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Scalings of the mean set inside the Orthogonal Subspace

```@raw html
</summary>
```

- Gives every direction of the Orthogonal Subspace the same uncertainty, the default. [`IdentityScaling`](@ref)
- Sizes each direction of the Orthogonal Subspace by the idiosyncratic covariance projected onto it. [`IdiosyncraticVarianceScaling`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Axis tags of the ellipsoid and the norm ball

```@raw html
</summary>
```

- Tags an [`EllipsoidalUncertaintySet`](@ref) or a [`NormBallUncertaintySet`](@ref) as living on the mean axis, where the shape matrix is $N \times N$ and the geometry map has $N$ rows. [`MuUncertaintySetClass`](@ref)
- Tags an [`EllipsoidalUncertaintySet`](@ref) or a [`NormBallUncertaintySet`](@ref) as living on the covariance axis, where the shape matrix is $N^{2} \times N^{2}$ and the geometry map has $N^{2}$ rows. [`SigmaUncertaintySetClass`](@ref)

```@raw html
</details>
```

## Turnover

The turnover is defined as the element-wise absolute difference between the vector of current weights and a vector of benchmark weights. It can be used as a constraint, method for fee calculation, and risk measure. These are all implemented using [`turnover_constraints`](@ref), [`TurnoverEstimator`](@ref), and [`Turnover`](@ref).

## Fees

Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naive ones. They can also be used to adjust expected returns calculations via [`calc_fees`](@ref) and [`calc_asset_fees`](@ref).

- Resolve the name-keyed fee fields of a [`FeesEstimator`](@ref) against a universe, giving a [`Fees`](@ref) of plain per-asset vectors. [`fees_constraints`](@ref)
- Compute the fixed portfolio fees for assets that have been allocated. [`calc_fixed_fees`](@ref) and [`calc_asset_fixed_fees`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)

```@raw html
</summary>
```

- Proportional long
- Proportional short
- Fixed long
- Fixed short
- Turnover

```@raw html
</details>
```

- Spreads the one-off terms of a fee, the turnover charge and the two fixed charges, over a holding period. [`AmortisedFees`](@ref)

## Portfolio returns and drawdowns

Various risk measures and analyses require the computation of simple and cumulative portfolio returns and drawdowns both in aggregate and per-asset. These are computed by [`calc_net_returns`](@ref), [`calc_net_asset_returns`](@ref), [`cumulative_returns`](@ref), [`drawdowns`](@ref).

A window may instead be scored on the weights a fund holds, which grow at their own asset returns while no trade is placed. The series is then the wealth ratio of the drifted holdings.

- Grows each position at its own asset return and holds no trade in between, so the weights drift and the series is the wealth ratio of the drifted holdings. [`SelfFinancingDrift`](@ref)

## [Tracking](@id catalogue-tracking)

It is often useful to create portfolios that track the performance of an index, indicator, or another portfolio.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Compute the benchmark portfolio returns for a weights-based tracking algorithm. [`tracking_benchmark`](@ref) and [`TrackingError`](@ref)

```@raw html
</summary>
```

- Carries the benchmark return series itself, for a benchmark whose weights are unknown. [`ReturnsTracking`](@ref)
- Builds the benchmark return series by holding a fixed weight vector, net of its own fees. [`WeightsTracking`](@ref)

```@raw html
</details>
```

The error can be computed using different algorithms using [`norm_error`](@ref).

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Norm tracking algorithms

```@raw html
</summary>
```

- Norm-one (NOC) error formulation. [`L1Norm`](@ref)
- Second-order cone (SOC) norm-based error formulation. [`L2Norm`](@ref)
- Second-order cone (SOC) squared norm-based error formulation. [`SquaredL2Norm`](@ref)
- L-p norm error estimator. [`LpNorm`](@ref)
- L-infinity norm (maximum absolute deviation) error estimator. [`LInfNorm`](@ref)

```@raw html
</details>
```

The distance may also be a risk distance rather than a norm of the return difference, measured against a [`WeightsTracking`](@ref) benchmark. Two approaches are available.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Constrains how far a portfolio's **risk** may stand from a benchmark portfolio's risk. [`RiskTrackingError`](@ref)

```@raw html
</summary>
```

- Applies the risk measure to each portfolio, then takes the absolute difference of the two risks. [`DependentVariableTracking`](@ref)
- Applies the risk measure to the difference between the portfolio weights and the benchmark weights. [`IndependentVariableTracking`](@ref)

```@raw html
</details>
```

## Risk measures

`PortfolioOptimisers.jl` provides a wide range of risk measures. These are broadly categorised into two types based on the type of optimisations that support them.

Every prior-derived slot on a risk measure -- `mu`, `sigma`, `kt`, `sk` -- takes the value itself or the estimator that computes it, a [`DeferredQuantity`](@ref). The estimator is resolved against the optimisation's own prior, so it refits per cross-validation fold and per meta-optimiser subset where a pasted matrix cannot. A measure with two or more deferrable slots names one prior estimator in `pe` instead, and one fit fills every slot the measure leaves unstated. See ADR 0051.

### Calibration

A tail probability and a deformation parameter also take a **rule** in place of the number. The slot names the quantity and the end of the distribution it addresses, so the caller writes the rule alone and the slot stores it. The rule is resolved against the optimisation's own prior, so the quantity refits per cross-validation fold and per meta-optimiser subset where a stated number holds still. Each slot's type bound names the one rule family that computes its quantity, and it admits a callable estimator of that family, a plain function of `(key, pr, w, slv, ctx)`, or the number itself. Five rules ship: two compute a significance level, and three compute a deformation parameter. Forty slots across twenty-seven risk measures and weight builders take one: every `alpha` and `beta`, and every `kappa`, `kappa_a` and `kappa_b`. The two inner integration bounds of the tail-Gini family, `alpha_i` and `beta_i`, are starting points rather than quantities to estimate, so they keep their numbers and the joint `0 < alpha_i < alpha < 1` bound is checked against the calibrated `alpha` at fold time.

A significance rule reads the sample length, and reads the effective observation weights where the count it states is a count of observations. A deformation rule reads the probability of its own end, which reaches it in the `CalibrationContext` the slot owner builds. One spends a stated entropy budget on the sample length. The other two read the shape of the sample's own tail and return the reciprocal of its index: one standardises each column and answers per end, so a skewed sample gives two numbers, and one whitens each observation with the covariance matrix and answers one number for both ends.

A rule that reads the shape of a series is told which series to read. A drawdown measure prices the drawdown series of the portfolio rather than its returns, and the slot key names neither, so the measure states its own series in the `CalibrationContext` beside the significance level. The two rules then run the same reading over the drawdown series of each column, in place of the columns themselves: the pooled rule pools those series, and the radial rule whitens their rows with the covariance matrix of that same sample, because a prior result states no drawdown moment. The series belongs to the measure, so no rule holds a marker of its own and a caller who runs a rule by hand states the marker in the context the measure would have built.

- Computes a significance level from a count of observations, so that the tail keeps the same number of scenarios whatever the sample length becomes. [`ScenarioCount`](@ref)
- Computes a significance level that shrinks with the square root of the sample length. [`RateSignificance`](@ref)
- Computes the Kaniadakis deformation parameter that makes a relativistic measure spend a stated entropy budget. [`EntropyBudget`](@ref)
- Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's own tail decays at. [`HillTailDecay`](@ref)
- Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's radial series decays at. [`RadialTailDecay`](@ref)
- Names the returns themselves, the columns of `pr.X` unchanged. [`ReturnsSeries`](@ref)
- Names the absolute drawdown series of a column, which [`absolute_drawdown_vec`](@ref) builds. [`AbsoluteDrawdownSeries`](@ref)
- Names the relative drawdown series of a column, which [`relative_drawdown_vec`](@ref) builds. [`RelativeDrawdownSeries`](@ref)

An ambiguity radius takes a rule on the same terms. It names no end of the distribution, so one bound serves every radius slot, and it reaches the four regularisation coefficients of [`JuMPOptimiser`](@ref) as well as the two distributionally robust risk measures. Four rules ship, and all four compute a radius. Two shrink the ball at the square-root rate of the sample length, and the third shrinks it at the rate the number of assets sets, which is far slower over a wide universe.

- Computes an ambiguity radius from the concentration of measure, so that the ball shrinks as the sample grows. [`ConcentrationRadius`](@ref)
- Computes an ambiguity radius that shrinks with the square root of the sample length. [`RateRadius`](@ref)
- Computes an ambiguity radius that shrinks at the dimensional rate a Wasserstein ball earns, not at the square-root rate. [`DimensionalRateRadius`](@ref)

Those three return one number for every slot, and the eight radius slots do not measure distance in one norm. The fourth rule reads the slot's key, picks the ground metric that slot names, and returns the sampling error of the empirical measure in it, so the `l1` and `linf` coefficients of one optimiser get two coefficients rather than one.

- Computes an ambiguity radius in the ground metric that the slot it stands in names, so that two slots of two different norms get two different numbers. [`DualNormRadius`](@ref)

The tail weight of an Esfahani-Kuhn loss carries a family of its own. One rule ships there, and it prices the tail term of the loss at a stated multiple of its mean term: a stated tail weight is dimensionless and is not scale-free in the sample, so one number is a different trade-off at every sampling frequency. That rule reads the probability of its own slot, which reaches it in the same context.

- Computes the Esfahani-Kuhn tail weight that prices the tail term of the loss at a stated multiple of its mean term. [`TailTermParity`](@ref)

A norm ceiling is a different quantity from a radius, so it carries a family of its own and neither family is admitted in the other's slot. A radius is the coefficient of a norm penalty in the objective; a ceiling bounds that norm in a constraint, and its reciprocal is a floor on the effective number of assets. It reaches the three norm-constraint slots of [`JuMPOptimiser`](@ref), `l2c`, `lpc` and `linfc`. One rule ships, and it holds a stated fraction of the universe effective, so the floor moves with the universe the prior carries. The order the ceiling is read against belongs to the constraint rather than to the rule, so each constraint site hands its own order over before the slot resolves.

- Computes a norm ceiling that holds a stated fraction of the universe effective, so that the floor refits whenever the universe changes. [`EffectiveAssetFloor`](@ref)

### Risk measures for traditional optimisation

These are all subtypes of [`RiskMeasure`](@ref), and are supported by all optimisation estimators.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the portfolio variance using a covariance matrix. [`Variance`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisations also support:

```@raw html
</summary>
```

- Risk contribution

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Formulations

```@raw html
</summary>
```

- Encodes the second moment as an explicit quadratic form, without an auxiliary variable or a cone. [`QuadRiskExpr`](@ref)
- Encodes the second moment as the square of a second-order cone variable. [`SquaredSOCRiskExpr`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

- Represents the portfolio standard deviation using a covariance matrix. [`StandardDeviation`](@ref)
- Uncertainty set variance [`UncertaintySetVariance`](@ref) (same as variance when used in non-traditional optimisation)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents a low-order moment risk measure. [`LowOrderMoment`](@ref)

```@raw html
</summary>
```

- Represents the first lower moment risk measure algorithm. [`FirstLowerMoment`](@ref)
- Represents the mean absolute deviation risk measure algorithm. [`MeanAbsoluteDeviation`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents a second moment (variance or standard deviation) risk measure algorithm. [`SecondMoment`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Second squared moments

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- Encodes the second moment as an explicit quadratic form, without an auxiliary variable or a cone. [`QuadRiskExpr`](@ref)
- Encodes the second moment as the square of a second-order cone variable. [`SquaredSOCRiskExpr`](@ref)
- Encodes the second moment as a variable that a rotated second-order cone bounds. [`RSOCRiskExpr`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Encodes the square root of the second moment as a second-order cone variable. [`SOCRiskExpr`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the square root kurtosis risk measure. [`Kurtosis`](@ref)

```@raw html
</summary>
```

- Actual kurtosis

```@raw html
<details class="cap-group" style="margin-left: 4em">
<summary>
```

FullMoment and semi-kurtosis are supported in traditional optimisers via the `kt` field. Risk calculation uses

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 4em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- Encodes the second moment as an explicit quadratic form, without an auxiliary variable or a cone. [`QuadRiskExpr`](@ref)
- Encodes the second moment as the square of a second-order cone variable. [`SquaredSOCRiskExpr`](@ref)
- Encodes the second moment as a variable that a rotated second-order cone bounds. [`RSOCRiskExpr`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Encodes the square root of the second moment as a second-order cone variable. [`SOCRiskExpr`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Negative Skewness risk measure. [`NegativeSkewness`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Squared negative skewness

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

FullMoment and semi-skewness are supported in traditional optimisers via the `sk` and `V` fields. Risk calculation uses

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- Encodes the second moment as an explicit quadratic form, without an auxiliary variable or a cone. [`QuadRiskExpr`](@ref)
- Encodes the second moment as the square of a second-order cone variable. [`SquaredSOCRiskExpr`](@ref)

```@raw html
</details>
```

- Encodes the square root of the second moment as a second-order cone variable. [`SOCRiskExpr`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Value-at-Risk (VaR) risk measure. [`ValueatRisk`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- Mixed-integer programming (MIP) formulation for Value-at-Risk. [`MIPValueatRisk`](@ref)
- Distribution-based formulation for Value-at-Risk. [`DistributionValueatRisk`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Value-at-Risk Range risk measure. [`ValueatRiskRange`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- Mixed-integer programming (MIP) formulation for Value-at-Risk. [`MIPValueatRisk`](@ref)
- Distribution-based formulation for Value-at-Risk. [`DistributionValueatRisk`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Represents the Drawdown-at-Risk (DaR) risk measure. [`DrawdownatRisk`](@ref)
- Represents the Conditional Value-at-Risk (CVaR) risk measure, also known as Expected Shortfall (ES). [`ConditionalValueatRisk`](@ref)
- Distributionally Robust Conditional Value at Risk [`DistributionallyRobustConditionalValueatRisk`](@ref) (same as conditional value at risk when used in non-traditional optimisation)
- Represents the Conditional Value-at-Risk Range (CVaR Range) risk measure. [`ConditionalValueatRiskRange`](@ref)
- Distributionally Robust Conditional Value at Risk Range [`DistributionallyRobustConditionalValueatRiskRange`](@ref) (same as conditional value at risk range when used in non-traditional optimisation)
- Represents the Conditional Drawdown-at-Risk (CDaR) risk measure, also known as Expected Maximum Drawdown. [`ConditionalDrawdownatRisk`](@ref)
- Distributionally Robust Conditional Drawdown at Risk [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)(same as conditional drawdown at risk when used in non-traditional optimisation)
- Represents the Entropic Value-at-Risk (EVaR) risk measure. [`EntropicValueatRisk`](@ref)
- Represents the Entropic Value-at-Risk Range (EVaR Range) risk measure. [`EntropicValueatRiskRange`](@ref)
- Represents the Entropic Drawdown-at-Risk (EDaR) risk measure. [`EntropicDrawdownatRisk`](@ref)
- Represents the Relativistic Value-at-Risk (RLVaR) risk measure. [`RelativisticValueatRisk`](@ref)
- Represents the Relativistic Value-at-Risk Range (RLVaR Range) risk measure. [`RelativisticValueatRiskRange`](@ref)
- Represents the Relativistic Drawdown-at-Risk (RLDaR) risk measure. [`RelativisticDrawdownatRisk`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Ordered Weights Array

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk measures

```@raw html
</summary>
```

- Ordered Weights Array (OWA) risk measure. [`OrderedWeightsArray`](@ref)
- Ordered Weights Array Range (OWA Range) risk measure. [`OrderedWeightsArrayRange`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

- OWA formulation that computes the exact OWA risk by solving a linear programme. [`ExactOrderedWeightsArray`](@ref)
- OWA formulation that approximates the OWA risk using a set of p-norm parameters. [`ApproxOrderedWeightsArray`](@ref)
- Estimator type for OWA weights using JuMP-based optimization. [`OWAJuMP`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

One-call OWA measures

```@raw html
</summary>
```

- Callable OWA weight estimator for the Conditional Value at Risk (CVaR) risk measure. [`OrderedWeightsArrayConditionalValueatRisk`](@ref)
- Callable OWA weight estimator for the Conditional Value at Risk Range risk measure. [`OrderedWeightsArrayConditionalValueatRiskRange`](@ref)
- Callable OWA weight estimator for the tail Gini risk measure. [`OrderedWeightsArrayTailGini`](@ref)
- Callable OWA weight estimator for the tail Gini range risk measure. [`OrderedWeightsArrayTailGiniRange`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Array functions

```@raw html
</summary>
```

- Gini Mean Difference [`owa_gmd`](@ref)
- Worst Realisation [`owa_wr`](@ref)
- Range [`owa_rg`](@ref)
- Conditional Value at Risk [`owa_cvar`](@ref)
- Weighted Conditional Value at Risk [`owa_wcvar`](@ref)
- Conditional Value at Risk Range [`owa_cvarrg`](@ref)
- Weighted Conditional Value at Risk Range [`owa_wcvarrg`](@ref)
- Tail Gini [`owa_tg`](@ref)
- Tail Gini Range [`owa_tgrg`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Linear moments (L-moments)

```@raw html
</summary>
```

- Compute the linear moment weights for the linear moments convex risk measure (CRM). [`owa_l_moment`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Compute Ordered Weights Array (OWA) linear moment convex risk measure (CRM) weights using various estimation methods. [`owa_l_moment_crm`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

L-moment combination formulations

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Maximum Entropy algorithm for Ordered Weights Array (OWA) estimation. [`MaximumEntropy`](@ref)

```@raw html
</summary>
```

- Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the exponential cone entropy constraint in JuMP. [`ExponentialConeEntropy`](@ref)
- Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the relative entropy cone constraint in JuMP. [`RelativeEntropy`](@ref)

```@raw html
</details>
```

- Represents the Minimum Squared Distance algorithm for Ordered Weights Array (OWA) estimation. [`MinimumSquaredDistance`](@ref)
- Represents the Minimum Sum of Squares algorithm for Ordered Weights Array (OWA) estimation. [`MinimumSumSquares`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

- Represents the Average Drawdown risk measure. [`AverageDrawdown`](@ref)
- Represents the Ulcer Index risk measure. [`UlcerIndex`](@ref)
- Represents the Maximum Drawdown risk measure. [`MaximumDrawdown`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Brownian Distance Variance (BDVar) risk measure. [`BrownianDistanceVariance`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Traditional optimisation formulations

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Distance matrix constraint formulations

```@raw html
</summary>
```

- Norm-one cone formulation for the Brownian Distance Variance optimisation constraint. [`NormOneConeBrownianDistanceVariance`](@ref)
- Inequality formulation for the Brownian Distance Variance optimisation constraint. [`IneqBrownianDistanceVariance`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk formulation

```@raw html
</summary>
```

- Encodes the second moment as an explicit quadratic form, without an auxiliary variable or a cone. [`QuadRiskExpr`](@ref)
- Encodes the second moment as a variable that a rotated second-order cone bounds. [`RSOCRiskExpr`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

- Represents the Worst Realisation risk measure. [`WorstRealisation`](@ref)
- Represents the Range risk measure. [`Range`](@ref)
- Represents the Turnover risk measure. [`TurnoverRiskMeasure`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the Tracking Error risk measure. [`TrackingRiskMeasure`](@ref)

```@raw html
</summary>
```

- Norm-one (NOC) error formulation. [`L1Norm`](@ref)
- Second-order cone (SOC) norm-based error formulation. [`L2Norm`](@ref)
- Second-order cone (SOC) squared norm-based error formulation. [`SquaredL2Norm`](@ref)
- L-p norm error estimator. [`LpNorm`](@ref)
- L-infinity norm (maximum absolute deviation) error estimator. [`LInfNorm`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk Tracking Risk Measure

```@raw html
</summary>
```

- Applies the risk measure to each portfolio, then takes the absolute difference of the two risks. [`DependentVariableTracking`](@ref)
- Applies the risk measure to the difference between the portfolio weights and the benchmark weights. [`IndependentVariableTracking`](@ref)

```@raw html
</details>
```

- Represents the Power Norm Value-at-Risk (PNVaR) risk measure. [`PowerNormValueatRisk`](@ref)
- Represents the Power Norm Value-at-Risk Range (PNVaRRange) risk measure. [`PowerNormValueatRiskRange`](@ref)
- Represents the Power Norm Drawdown-at-Risk (PNDaR) risk measure. [`PowerNormDrawdownatRisk`](@ref)
- Represents a generic Value-at-Risk range risk measure that combines any pair of XatRisk-type measures applied to the loss and gain sides of the return distribution. [`GenericValueatRiskRange`](@ref)
- Represents the Risk Tracking risk measure. [`RiskTrackingRiskMeasure`](@ref)
- Risk measure that contributes no risk. [`NoRisk`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk measure settings

```@raw html
</summary>
```

Every risk measure carries a settings object saying how it enters the problem: as the objective, as a constraint with an upper bound, and with what scale.

- Weights a risk measure inside an aggregate, and bounds its risk expression from above. [`RiskMeasureSettings`](@ref)
- Weights a hierarchical risk measure inside an aggregate, and carries no bound. [`HierarchicalRiskMeasureSettings`](@ref)
- Weights a risk measure inside an aggregate, and bounds its risk expression from **below**. [`MaxRiskMeasureSettings`](@ref)

```@raw html
</details>
```

### Risk measures for hierarchical optimisation

These are all subtypes of [`HierarchicalRiskMeasure`](@ref), and are only supported by hierarchical optimisation estimators.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents a high-order moment risk measure. [`HighOrderMoment`](@ref)

```@raw html
</summary>
```

- Represents the unstandardised semi-skewness risk measure algorithm. [`ThirdLowerMoment`](@ref)
- Represents a standardised high-order moment risk measure algorithm. [`StandardisedHighOrderMoment`](@ref) and [`ThirdLowerMoment`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents the unstandardised fourth moment (kurtosis or semi-kurtosis) risk measure algorithm. [`FourthMoment`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Represents a standardised high-order moment risk measure algorithm. [`StandardisedHighOrderMoment`](@ref) and [`FourthMoment`](@ref)

```@raw html
</summary>
```

- Keeps every deviation from the target, so the moment is two-sided. [`FullMoment`](@ref)
- Clips every deviation above the target to zero, so the moment reads the downside alone. [`SemiMoment`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Represents the Relative Drawdown-at-Risk risk measure for hierarchical optimisation. [`RelativeDrawdownatRisk`](@ref)
- Represents the Relative Conditional Drawdown-at-Risk risk measure for hierarchical optimisation. [`RelativeConditionalDrawdownatRisk`](@ref)
- Represents the Relative Entropic Drawdown-at-Risk (Relative EDaR) risk measure for hierarchical optimisation. [`RelativeEntropicDrawdownatRisk`](@ref)
- Represents the Relative Relativistic Drawdown-at-Risk (Relative RLDaR) risk measure for hierarchical optimisation. [`RelativeRelativisticDrawdownatRisk`](@ref)
- Represents the Relative Average Drawdown risk measure for hierarchical optimisation. [`RelativeAverageDrawdown`](@ref)
- Represents the Relative Ulcer Index risk measure for hierarchical optimisation. [`RelativeUlcerIndex`](@ref)
- Represents the Relative Maximum Drawdown risk measure for hierarchical optimisation. [`RelativeMaximumDrawdown`](@ref)
- Represents the Relative Power Norm Drawdown-at-Risk (Relative PNDaR) risk measure for hierarchical optimisation. [`RelativePowerNormDrawdownatRisk`](@ref)
- Represents a risk ratio risk measure for hierarchical portfolio optimisation. [`RiskRatio`](@ref)
- Represents the Equal Risk Measure for hierarchical portfolio optimisation. [`EqualRisk`](@ref)
- Represents the Median Absolute Deviation (MAD) risk measure for hierarchical portfolio optimisation. [`MedianAbsoluteDeviation`](@ref)
- Composite risk measure combining variance, skewness, and kurtosis into a single expression. [`VarianceSkewKurtosis`](@ref)
- Represents an even-order moment risk measure algorithm. [`EvenMoment`](@ref)
- Callable estimator that generates OWA linear moment convex risk measure (CRM) weights for a given number of observations. [`LinearMoment`](@ref)

### Non-optimisation risk measures

These risk measures are unsuitable for optimisation because they can return negative values. However, they can be used for performance metrics.

- Represents a simple mean return measure for use in non-optimisation contexts. [`MeanReturn`](@ref)
- Represents the Third Central Moment risk measure. [`ThirdCentralMoment`](@ref)
- Represents the standardised Skewness risk measure. [`Skewness`](@ref)
- Return-based risk measure. [`ExpectedReturn`](@ref)
- Ratio-based risk measure. [`ExpectedReturnRiskRatio`](@ref)
- Represents a mean return to risk ratio measure. [`MeanReturnRiskRatio`](@ref)
- Represents a non-optimisation risk ratio measure. [`NonOptimisationRiskRatio`](@ref)

## Performance metrics

Every reader here takes one risk measure or a vector of them, scalarised into one number by a `sca` keyword bounded [`Scalariser`](@ref) — all four scalarisers, [`MinScalariser`](@ref) included, because the value level combines computed numbers rather than building a model expression. Where a return axis is present it takes one term or a vector of them, summed at the terms' own combination weights; there is no scalariser on the return axis. A result carries the measure and scalariser it ran under, so `expected_risk(res.r, res.w, res.pr; sca = res.sca)` reports the optimised figure without naming either by hand.

- Compute the expected value of a risk measure. [`expected_risk`](@ref)
- Compute the effective number of assets (Herfindahl-Hirschman inverse index). [`number_effective_assets`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk contribution

```@raw html
</summary>
```

- Compute the risk contribution of each asset to the total portfolio risk using numerical differentiation. [`risk_contribution`](@ref)
- Compute the risk contribution of each factor (and the idiosyncratic component) to the total portfolio risk using a factor regression. [`factor_risk_contribution`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Factor attribution

```@raw html
</summary>
```

[`factor_attribution`](@ref) decomposes a portfolio's volatility and mean return over the factors, the factor families and the assets of a factor model, and returns one [`FactorAttributionResult`](@ref). The predicted methods read the moments the optimiser saw; the realised methods read a net return series, and each has a rolling twin. What the model does not explain is a fourth component of its own.

- Decompose a portfolio's volatility and mean return over the factors of a factor model. [`factor_attribution`](@ref) and [`FactorAttributionResult`](@ref)
- One row of a factor attribution: the volatility, the volatility contribution, the variance share, the mean return contribution and the correlation of one component of the portfolio return. [`AttributionComponent`](@ref)
- The factor axis or the family axis of a factor attribution, one entry per row of the axis. [`AttributionBreakdown`](@ref)
- The asset axis of a factor attribution, one entry per asset. [`AssetAttributionBreakdown`](@ref)
- The asset-by-factor contributions of a factor attribution, two matrices of assets by factors. [`AssetFactorContribution`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Compute the expected portfolio return using the specified return estimator. [`expected_return`](@ref)

```@raw html
</summary>
```

- Arithmetic [`ArithmeticReturn`](@ref)
- Logarithmic [`LogarithmicReturn`](@ref)
- None [`NoReturn`](@ref)

```@raw html
</details>
```

- Compute the expected risk of a measure from a precomputed net-return series. [`expected_risk_from_returns`](@ref)
- Compute the expected risk of a risk measure over rolling windows of the returns data. [`rolling_window_measure`](@ref)
- Sort the successful paths in a [`PopulationPredictionResult`](@ref) by their expected risk under `r`. [`sort_by_measure`](@ref)
- Compute the expected risk-adjusted return ratio for a portfolio. [`expected_ratio`](@ref) and [`expected_risk_ret_ratio`](@ref)
- Compute the risk-adjusted ratio information criterion (SRIC) for a portfolio. [`expected_sric`](@ref) and [`expected_risk_ret_sric`](@ref)
- Compute Brinson performance attribution aggregated per asset class [brinson_attribution](@cite). [`brinson_attribution`](@ref)
- Summarise a realised return series as a [`PerformanceSummaryResult`](@ref). [`performance_summary`](@ref) and [`PerformanceSummaryResult`](@ref)

## Portfolio optimisation

Optimisations are implemented via [`optimise`](@ref). Optimisations consume an estimator and return a result.

### Naive

These return a [`NaiveOptimisationResult`](@ref).

- Allocates each asset a weight inversely proportional to its volatility, or to its variance when `sq = true`. [`InverseVolatility`](@ref)
- Allocates the same weight to every asset in the universe. [`EqualWeighted`](@ref)
- Draws portfolio weights at random from a Dirichlet distribution with concentration parameter `alpha`. [`RandomWeighted`](@ref)

#### Naive optimisation features

- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

### Traditional

These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`](@ref), which encodes all supported constraints.

#### Objective function optimisations

These optimisations support a variety of objective functions.

- Configures one solver backend, its attributes, and the statuses its solutions must reach. [`Solver`](@ref)
- Main JuMP-based portfolio optimiser configuration. [`JuMPOptimiser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Objective functions

```@raw html
</summary>
```

- Minimum risk [`MinimumRisk`](@ref)
- Maximum utility [`MaximumUtility`](@ref)
- Maximum return over risk ratio [`MaximumRatio`](@ref)
- Maximum return [`MaximumReturn`](@ref)
- Internal objective that maximises the expression of **one** return term. [`MaximumElementReturn`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Mean-Risk portfolio optimiser. [`MeanRisk`](@ref) and [`NearOptimalCentering`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Sweeps the efficient frontier by solving the model once at each of `N` evenly spaced bound values. [`Frontier`](@ref)

```@raw html
</summary>
```

- Return based
- Risk based

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Bound spacing [`FrontierBoundEstimator`](@ref)

```@raw html
</summary>
```

- Passes bound values through unchanged (identity transformation). [`LinearBound`](@ref)
- Applies a square-root transformation to bound values before enforcing them. [`SquareRootBound`](@ref)
- Applies a squaring transformation to bound values before enforcing them. [`SquaredBound`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Optimisation estimators

```@raw html
</summary>
```

- Mean-Risk [`MeanRisk`](@ref) returns a [`MeanRiskResult`](@ref)
- Near Optimal Centering [`NearOptimalCentering`](@ref) returns a [`NearOptimalCenteringResult`](@ref)
- Factor Risk Contribution [`FactorRiskContribution`](@ref) returns a [`FactorRiskContributionResult`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Near Optimal Centering formulations [`NearOptimalCentering`](@ref)

```@raw html
</summary>
```

- Constrained Near Optimal Centering algorithm. [`ConstrainedNearOptimalCentering`](@ref)
- Unconstrained Near Optimal Centering algorithm. [`UnconstrainedNearOptimalCentering`](@ref)
- Intermediate result type storing the setup data for Near Optimal Centering. [`NearOptimalSetup`](@ref)

```@raw html
</details>
```

#### Risk budgeting optimisations

These optimisations attempt to achieve weight values according to a risk budget vector. This vector can be provided on a per asset or per factor basis.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Budget targets

```@raw html
</summary>
```

- Asset-level Risk Budgeting algorithm. [`AssetRiskBudgeting`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Fromulations

```@raw html
</summary>
```

- Log-barrier formulation for Risk Budgeting. [`LogRiskBudgeting`](@ref)
- Mixed-integer formulation for Risk Budgeting. [`MixedIntegerRiskBudgeting`](@ref)

```@raw html
</details>
```

- Factor-level Risk Budgeting algorithm. [`FactorRiskBudgeting`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Optimisation estimators

```@raw html
</summary>
```

- Risk Budgeting [`RiskBudgeting`](@ref) returns a [`RiskBudgetingResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Relaxed Risk Budgeting [`RelaxedRiskBudgeting`](@ref) returns a [`RelaxedRiskBudgetingResult`](@ref)

```@raw html
</summary>
```

- Bounds the risk variable by the portfolio standard deviation alone, which is the relaxation with no extra term. [`BasicRelaxedRiskBudgeting`](@ref)
- Adds a second cone on an auxiliary scalar, which lifts the floor on the risk variable and improves numerical stability. [`RegularisedRelaxedRiskBudgeting`](@ref)
- Bounds the auxiliary scalar by the individual standard deviations rather than the portfolio one, weighted by `p`. [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

#### Traditional optimisation features

- Abstract supertype for custom JuMP objective implementations. [`CustomJuMPObjective`](@ref)
- Abstract supertype for custom JuMP constraint implementations. [`CustomJuMPConstraint`](@ref)
- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Budget

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Directionality

```@raw html
</summary>
```

- Long
- Short

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Type

```@raw html
</summary>
```

- Exact
- Bounds the sum of the portfolio weights inside a closed interval, rather than pinning it to one value. [`BudgetRange`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Resolves a minimum-holding threshold written in asset or group names against a universe. [`ThresholdEstimator`](@ref) and [`Threshold`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Directionality

```@raw html
</summary>
```

- Long
- Short

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Type

```@raw html
</summary>
```

- Asset
- Names the group name key a binary asset-group membership matrix is built from. [`AssetSetsMatrixEstimator`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Holds the linear constraint equations to parse, and the universe key their names resolve against. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Bundles a network source with the centrality algorithm that scores its assets. [`CentralityEstimator`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cardinality

```@raw html
</summary>
```

- Asset
- Holds the linear constraint equations to parse, and the universe key their names resolve against. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Set(s)
- Holds the linear constraint equations to parse, and the universe key their names resolve against. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)

```@raw html
</details>
```

- Names the per-asset turnover bounds, for [`turnover_constraints`](@ref) to align to a universe. [`TurnoverEstimator`](@ref) and [`Turnover`](@ref)
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Bounds how far the portfolio return series may drift from a benchmark return series. [`TrackingError`](@ref)
- Caps how many related assets may be held at once, refitting the structure from returns. [`IntegerPhylogenyEstimator`](@ref) and [`SemiDefinitePhylogenyEstimator`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Portfolio returns

```@raw html
</summary>
```

- One return term, or a vector of them weighted-summed into the model's return expression
- Carries one return term's own weight in the return sum, its own lower bound, and the two charges netted out of it. [`JuMPReturnsSettings`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Arithmetic [`ArithmeticReturn`](@ref)

```@raw html
</summary>
```

- Holds the element-wise lower and upper bounds of a box uncertainty set on a mean vector or on a covariance matrix. [`BoxUncertaintySet`](@ref), [`BoxUncertaintySetAlgorithm`](@ref), [`EllipsoidalUncertaintySet`](@ref), [`EllipsoidalUncertaintySetAlgorithm`](@ref), [`NormBallUncertaintySet`](@ref), and [`NormBallUncertaintySetAlgorithm`](@ref)
- Custom expected returns vector
- Deferred expected returns estimator, resolved against the optimisation's own prior

```@raw html
</details>
```

- Logarithmic [`LogarithmicReturn`](@ref)
- None [`NoReturn`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk vector scalarisation

```@raw html
</summary>
```

- Adds the scaled risk measures together. [`SumScalariser`](@ref)
- Reports the largest of the scaled risk measures, so the aggregate is the worst of them. [`MaxScalariser`](@ref)
- Smooths the maximum of the scaled risk measures, so every measure keeps a share of the aggregate. [`LogSumExpScalariser`](@ref)

```@raw html
</details>
```

- Custom constraint
- Number of effective assets

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Regularisation penalty

```@raw html
</summary>
```

- L1
- L2-norm regularisation term added to the optimisation objective. [`L2Regularisation`](@ref)
- Lp-norm regularisation term added to the optimisation objective. [`LpRegularisation`](@ref)
- L-Inf

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight-norm constraints

```@raw html
</summary>
```

Where a regularisation penalty prices a norm in the objective, these bound it instead.

- L2 (`l2c`)
- Lp (`lpc`)
- L-Inf (`linfc`)

```@raw html
</details>
```

- Estimator type for normalised constant relative risk aversion (CRRA) OWA weights. [`NormalisedConstantRelativeRiskAversion`](@ref)

### Clustering optimisation

Clustering optimisations make use of asset relationships to either minimise the risk exposure by breaking the asset universe into subsets which are hierarchically or individually optimised.

- Base configuration for hierarchical clustering-based portfolio optimisers. [`HierarchicalOptimiser`](@ref)

#### Hierarchical clustering optimisation

These optimisations minimise risk by hierarchically splitting the asset universe into subsets, computing the risk of each subset, and combining them according to their hierarchy.

Each result carries the measures and scalarisers its optimisation ran under, stored resolved, and shares its remaining fields through an embedded [`HierarchicalResult`](@ref) core reached as `res.hr` or directly as `res.w`, `res.pr` and the rest.

- Hierarchical Risk Parity [`HierarchicalRiskParity`](@ref) returns a [`HierarchicalRiskParityResult`](@ref)
- Hierarchical Equal Risk Contribution [`HierarchicalEqualRiskContribution`](@ref) returns a [`HierarchicalEqualRiskContributionResult`](@ref)
- Shared field core for hierarchical (clustering-based) optimisation results. [`HierarchicalResult`](@ref)

##### Hierarchical clustering optimisation features

- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk vector scalarisation

```@raw html
</summary>
```

- The clustering optimisers accept every scalariser; a `JuMP` optimiser accepts only the non-hierarchical three
- Adds the scaled risk measures together. [`SumScalariser`](@ref)
- Reports the largest of the scaled risk measures, so the aggregate is the worst of them. [`MaxScalariser`](@ref)
- Smooths the maximum of the scaled risk measures, so every measure keeps a share of the aggregate. [`LogSumExpScalariser`](@ref)
- Reports the smallest of the scaled risk measures, so the aggregate is the mildest of them. [`MinScalariser`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

#### Schur complementary optimisation

Schur complementary hierarchical risk parity provides a bridge between mean variance optimisation and hierarchical risk parity by using an interpolation parameter. It converges to hierarchical risk parity, and approximates mean variance by adjusting this parameter. It uses the Schur complement to adjust the weights of a portfolio according to how much more useful information is gained by assigning more weight to a group of assets.

- Schur Complementary Hierarchical Risk Parity [`SchurComplementHierarchicalRiskParity`](@ref) returns a [`SchurComplementHierarchicalRiskParityResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Collects the risk measure, the interpolation parameter $\gamma$, and the two algorithms that one Schur complement bundle runs with. [`SchurComplementParams`](@ref)

```@raw html
</summary>
```

- Searches $[0, \gamma]$ for the value that gives the lowest portfolio variance. [`MonotonicSchurComplement`](@ref)
- Runs the allocation at the $\gamma$ the caller gave, with no search. [`NonMonotonicSchurComplement`](@ref)

```@raw html
</details>
```

##### Schur complementary optimisation features

- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

#### Nested clusters optimisation

Nested clustered optimisation breaks the asset universe of size `N` into `C` smaller subsets and treats every subset as an individual portfolio. The weights assigned to each asset are placed in an `N × C` matrix. In each column, non-zero values correspond to assets assigned to that subset, this means that assets only contribute to the column (and therefore synthetic asset) corresponding to their assigned subset. In other words, each row of the matrix contains a single non-zero value and each row contains as many non-zero values as there are assets in that subset.

From here there are two options:

1. Compute the returns matrix of the synthetic assets directly by multiplying the original `T × N` matrix by the `N × C` matrix of asset weights to produce a `T × C` matrix of predicted returns, where `T` is the number of observations.
2. For each subset perform a cross validation prediction, yielding a vector of returns for that subset. These vectors are then horizontally concatenated into a `Y × C` matrix of cross-validation predicted returns, where `Y ≤ T` because the cross validation may not use the full history.

This matrix of predicted returns is then used by the outer optimisation estimator to generate an optimisation of the synthetic assets. This produces a `C × 1` vector, essentially optimising a portfolio of asset clusters. The final weights are the product of the original `N × C` matrix of asset weights per cluster by the `C × 1` vector of optimal synthetic asset weights to produce the final `N × 1` vector of asset weights.

- Nested Clustered [`NestedClustered`](@ref) returns a [`NestedClusteredResult`](@ref)

#### Nested clusters optimisation features

- Any features supported by the inner and outer estimators.
- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Cross validation predictor for the outer estimator

### Ensemble optimisation

This works similarly to the Nested Clustered estimator, only instead of breaking the asset universe into subsets, a list of inner estimators is provided. The procedure is then exactly the same as the nested clusters optimisation, only instead of an `N × C` matrix of asset weights where each column corresponds to a subset of assets, each column corresponds to a completely independent and isolated inner estimator, which also means there is no enforced sparsity pattern on this matrix.

- Stacking [`Stacking`](@ref) returns a [`StackingResult`](@ref)

#### Ensemble optimisation features

- Any features supported by the inner and outer estimators.
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Cross validation predictor for the outer estimator

### Subset resampling optimisation

This optimiser takes ideas from [`MultipleRandomised`](@ref) cross validation to randomly sample the asset universe and optimise each sample individually using a given optimiser. The final asset weights are the average weight per asset across all samples, if an asset does not appear in a sample, it is taken to be zero.

- [`SubsetResampling`](@ref) returns a [`SubsetResamplingResult`](@ref)

#### Subset resampling optimisation features

- Any features supported by the inner estimator.
- Names the per-asset fee rates, for [`fees_constraints`](@ref) to align to a universe. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Resolves weight bounds written in asset or group names against a universe. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

### Finite allocation optimisation

Unlike all other estimators, finite allocation does not yield an "optimal" value, but rather the optimal attainable solution based on a finite amount of capital. They use the result of other estimations, the latest prices, and a cash amount.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Discrete Allocation portfolio optimiser. [`DiscreteAllocation`](@ref)

```@raw html
</summary>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Weight finalisers

```@raw html
</summary>
```

- Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)

```@raw html
</summary>
```

- Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
- Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
- Minimises the L2 norm of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
</details>
```

- Greedy Allocation portfolio optimiser. [`GreedyAllocation`](@ref)
- Problem data fed to a finite allocation optimiser. [`FiniteAllocationInput`](@ref)

## Cross validation

- Prediction on unseen data [`PredictionReturnsResult`](@ref), [`PredictionResult`](@ref), [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref) via [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), [`fit_and_predict`](@ref)
- Union of concrete [`PredictionScorer`](@ref) subtypes and plain functions that score a [`PopulationPredictionResult`](@ref). [`PredictionCrossValScorer`](@ref), [`NearestQuantilePrediction`](@ref), and [`quantile_by_measure`](@ref)
- Run cross-validated portfolio optimisation and return predictions over all folds. [`cross_val_predict`](@ref)
- Fit optimisation estimator `opt` on returns data `rd` and immediately produce a [`PredictionResult`](@ref) for the same data. [`fit_predict`](@ref)
- Return the number of cross-validation splits (folds) that would be produced by `cv` for the given returns data `rd`. [`n_splits`](@ref)
- Find the optimal `(n_folds, n_test_folds)` pair for combinatorial cross-validation by minimising a weighted cost that balances the average training size against the number of test paths. [`optimal_number_folds`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Split `str` into an array of substrings on occurrences of the delimiter(s) `dlm`. [`split`](@ref) and [`fit_and_predict`](@ref)

```@raw html
</summary>
```

- K-Fold [`KFold`](@ref) returns a [`KFoldResult`](@ref)
- Combinatorial [`CombinatorialCrossValidation`](@ref) returns a [`CombinatorialCrossValidationResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Walk forward [`WalkForwardEstimator`](@ref) return a [`WalkForwardResult`](@ref)

```@raw html
</summary>
```

- Implements index-based walk-forward cross-validation for time series, supporting purging and flexible train/test windowing. [`IndexWalkForward`](@ref) and [`DateWalkForward`](@ref)

```@raw html
</details>
```

- Multiple randomised [`MultipleRandomised`](@ref) returns a [`MultipleRandomisedResult`](@ref)

```@raw html
</details>
```

A scheme reads a fold under an evaluation convention. [`SelfFinancingDrift`](@ref) reads a fold's series on the weights the fund holds rather than the weights the optimiser chose, and the fold then carries a [`HeldWeightsResult`](@ref). A walk-forward may also thread those held weights into the fold that follows it.

- Thread the weights a fold **held** after its last observation into the fold that follows it. [`DriftedWeights`](@ref)
- Records what a fold actually held, so a reader can recover the weight path of that fold. [`HeldWeightsResult`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Performs grid search cross-validation for portfolio optimisation estimators. [`search_cross_validation`](@ref)

```@raw html
</summary>
```

- Performs grid search cross-validation for portfolio optimisation estimators. [`GridSearchCrossValidation`](@ref)
- Randomised search cross-validation estimator for portfolio optimisation. [`RandomisedSearchCrossValidation`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Scoring a parameter set [`CrossValidationSearchScorer`](@ref)

```@raw html
</summary>
```

- A [`CrossValidationSearchScorer`](@ref) that selects the parameter set with the highest mean score across cross-validation splits. [`HighestMeanScore`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

- Wraps a cross-validation scheme and an optional scorer to form a complete optimisation cross-validation pipeline. [`OptimisationCrossValidation`](@ref)
- Abstract supertype for estimators that determine the number of random subsets to draw. [`NumberSubsetsEstimator`](@ref) and [`SubsetSizeEstimator`](@ref)

## [Pipeline](@id catalogue-pipeline)

A [`Pipeline`](@ref) reifies an end-to-end workflow as data: an ordered list of steps run left-to-right over a [`PipelineContext`](@ref), so preprocessing, priors, and the optimiser travel together as one estimator and can be cross-validated or tuned as a unit.

- A reified end-to-end portfolio workflow: an ordered list of steps executed left-to-right over a [`PipelineContext`](@ref). [`Pipeline`](@ref) and [`PipelineResult`](@ref)
- Explicit pipeline step wrapper — used when a step's slots or its routing intent must be stated rather than inferred. [`PipelineStep`](@ref)
- The accumulating blackboard threaded through a pipeline's steps. [`PipelineContext`](@ref)
- The mu/sigma pair held by the `uncertainty` slot of a [`PipelineContext`](@ref). [`PipelineUncertaintySets`](@ref)

## Plotting

Visualising the results is quite a useful way of summarising the portfolio characteristics or evolution. To this extent we provide a few plotting functions with more to come.

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Simple or compound cumulative returns.

```@raw html
</summary>
```

- Plot the cumulative returns of a portfolio. [`plot_portfolio_cumulative_returns`](@ref)
- Plot the cumulative returns of individual assets, selecting the most relevant via `N`. [`plot_asset_cumulative_returns`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Portfolio composition.

```@raw html
</summary>
```

- Plot portfolio composition as a bar chart of asset weights. [`plot_composition`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Multi portfolio.

```@raw html
</summary>
```

- Plot portfolio composition as a stacked bar chart. [`plot_stacked_bar_composition`](@ref)
- Plot portfolio composition as a stacked area chart. [`plot_stacked_area_composition`](@ref)

```@raw html
</details>
```

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Risk contribution.

```@raw html
</summary>
```

- Plot per-asset risk contribution as a bar chart. [`plot_risk_contribution`](@ref)
- Plot per-factor risk contribution as a bar chart, including the constant (idiosyncratic) term. [`plot_factor_risk_contribution`](@ref)

```@raw html
</details>
```

- Plot a hierarchical clustering dendrogram with coloured cluster regions. [`plot_dendrogram`](@ref)
- Plot a reordered correlation/covariance heatmap with flanking dendrograms and coloured cluster boxes. [`plot_clusters`](@ref)
- Plot portfolio drawdown over time. [`plot_drawdowns`](@ref)
- Line plot of the rolling maximum drawdown over a sliding window. [`plot_rolling_drawdowns`](@ref)
- Plot a histogram of portfolio returns with vertical risk-measure lines and an optional fitted Normal distribution. [`plot_histogram`](@ref)
- Scatter plot of risk/return measures across a collection of portfolio weight vectors. [`plot_measures`](@ref)
- Line plot of a risk or return measure evaluated over a rolling window of portfolio returns. [`plot_rolling_measure`](@ref)
- Sort a collection of portfolio results by risk (`x`), connect them with a line to trace the efficient frontier, and optionally annotate the minimum-risk and maximum-score portfolios. [`plot_efficient_frontier`](@ref)
- Box plot of per-asset weight distributions across cross-validation folds or population members. [`plot_weight_stability`](@ref)
- Line plot of portfolio turnover (L1 weight change) over time. [`plot_turnover`](@ref)
- Overlay portfolio cumulative returns against one or more benchmark return series from `rd.B`. [`plot_benchmark`](@ref)

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Moments and priors

```@raw html
</summary>
```

- Bar chart of per-asset expected returns (μ vector). [`plot_mu`](@ref)
- Bar chart of per-asset volatility (√diag(Σ)). [`plot_sigma`](@ref)
- Standalone correlation (or covariance) heatmap without clustering or dendrograms. [`plot_correlation`](@ref)
- Heatmap of the coskewness matrix (N × N²) from a [`HighOrderPrior`](@ref). [`plot_coskewness`](@ref)
- Eigenvalue spectrum of the cokurtosis matrix (N² × N²) from a [`HighOrderPrior`](@ref). [`plot_cokurtosis`](@ref)
- Bar chart of eigenvalues of the covariance/correlation matrix, sorted in descending order. [`plot_eigenspectrum`](@ref)
- Three-panel composite plot summarising a prior result: [`plot_prior`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Factor models

```@raw html
</summary>
```

- Bar chart of per-factor expected returns (`pr.fpr.mu`, from a factor model prior). [`plot_factor_mu`](@ref)
- Correlation/covariance heatmap of the factor covariance matrix (`pr.fpr.sigma`). [`plot_factor_sigma`](@ref)
- Heatmap of the factor loadings matrix B (assets × factors) from a prior with a regression model. [`plot_factor_loadings`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cross-sectional regression diagnostics

```@raw html
</summary>
```

- Plot the weighted cross-sectional coefficient of determination of every observation. [`plot_cs_regression_r2`](@ref)
- Plot the adjusted cross-sectional coefficient of determination of every observation. [`plot_cs_regression_adjusted_r2`](@ref)
- Plot the Akaike information criterion of every cross-sectional fit. [`plot_cs_regression_aic`](@ref)
- Plot the Bayesian information criterion of every cross-sectional fit. [`plot_cs_regression_bic`](@ref)
- Plot the t-statistic of every factor return, one series per factor. [`plot_cs_regression_t_stats`](@ref)
- Plot the fraction of observations at which each factor's t-statistic exceeds a threshold. [`plot_cs_regression_t_stat_exceedance_rate`](@ref)
- Plot the variance inflation factor of every factor, one series per factor. [`plot_exposure_vif`](@ref)
- Plot the condition number of the cross-sectional design of every observation. [`plot_exposure_condition_number`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cross-sectional exposure diagnostics

```@raw html
</summary>
```

- Plot the time-averaged correlation between every pair of factor exposures as a heatmap. [`plot_exposure_correlation`](@ref)
- Plot the stability of every factor exposure, one series per factor. [`plot_exposure_stability`](@ref)
- Plot the weighted cross-sectional standard deviation of every factor exposure, one series per factor. [`plot_exposure_dispersion`](@ref)
- Plot the cross-sectional distribution of one factor exposure as a histogram. [`plot_exposure_distribution`](@ref)
- Plot the running sum of the information coefficient of every factor exposure, one series per factor. [`plot_cumulative_exposure_ic`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cross-sectional idiosyncratic diagnostics

```@raw html
</summary>
```

- Plot the cross-sectional standard deviation of the standardised idiosyncratic returns against the observation axis. [`plot_idio_calibration`](@ref)
- Plot the share of assets whose standardised idiosyncratic return exceeds a threshold, against the observation axis. [`plot_idio_tail_rate`](@ref)
- Plot the cross-sectional excess kurtosis of the standardised idiosyncratic returns against the observation axis. [`plot_idio_kurtosis`](@ref)
- Plot the cross-sectional skewness of the standardised idiosyncratic returns against the observation axis. [`plot_idio_skewness`](@ref)
- Plot the information coefficient of the predicted idiosyncratic volatility against the observation axis. [`plot_idio_vol_ic`](@ref)
- Plot the residual dependence of the standardised idiosyncratic returns on the predicted volatility, against the observation axis. [`plot_idio_vol_residual_dependence`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Factor model summary and factor forecasts

```@raw html
</summary>
```

- Plot the columns of a factor model summary as a grouped bar chart, one group per column and one bar per factor. [`plot_factor_model_summary`](@ref)
- Plot the cumulative return of every factor, one series per factor. [`plot_factor_cumulative_returns`](@ref)
- Plot the forecast correlation of the factor returns as a heatmap. [`plot_factor_forecast_correlation`](@ref)
- Plot the forecast volatility of every factor return as a horizontal bar chart, ordered from the smallest. [`plot_factor_forecast_volatilities`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Phylogeny

```@raw html
</summary>
```

- Plot the asset network (MST, PMFG, TMFG, or adjacency) as a graph using `GraphRecipes.graphplot`. [`plot_network`](@ref)
- Bar chart of asset centrality scores, sorted in descending order. [`plot_centrality`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Cross validation

```@raw html
</summary>
```

- Bar chart of cross-validation scores (one bar per fold or population member). [`plot_cv_scores`](@ref)
- Four-panel composite plot for a walk-forward cross-validation result: [`plot_cv_dashboard`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Factor attribution

```@raw html
</summary>
```

- Plot the volatility contribution of each factor, or of each factor family, as a bar chart. [`plot_attribution_vol_contrib`](@ref)
- Plot the mean return contribution of each factor, or of each factor family, as a bar chart with error bars. [`plot_attribution_mu_contrib`](@ref)
- Plot the portfolio's exposure to each factor, or to each factor family, as a bar chart with its spread. [`plot_attribution_exposure`](@ref)
- Plot the mean return contribution of each factor against its volatility contribution, as a labelled scatter. [`plot_attribution_mu_vs_vol`](@ref)

```@raw html
</details>
```

```@raw html
<details class="cap-group" style="margin-left: 2em">
<summary>
```

Dashboards

```@raw html
</summary>
```

- Four-panel composite plot for a single optimisation result: [`plot_portfolio_dashboard`](@ref)
- Bar chart of annualised portfolio performance metrics: annualised return, annualised volatility, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown %, and CVaR %. [`plot_performance_summary`](@ref)

```@raw html
</details>
```
