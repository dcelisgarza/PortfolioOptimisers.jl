# Pre-processing

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
check_names_and_returns_matrix
prices_to_returns
port_opt_view(rd::ReturnsResult, i::Any)
port_opt_view(rd::ReturnsResult, i::Any, j::Any, k::Any)
port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)
returns_result_picker
```

## Price-level data

```@docs
AbstractPricesResult
PricesResult
port_opt_view(pr::PricesResult, ::Colon)
```

## Preprocessing estimators

Preprocessing estimators transform price- or returns-level data under a **fit/apply contract**: [`fit_preprocessing`](@ref) learns whatever state the transformation needs from a training window — the surviving asset universe, imputation parameters, thresholds — and [`apply_preprocessing`](@ref) replays that state on unseen windows, so no information flows from test data back into the transformation.

They are ordinary estimators and know nothing about pipelines. A [`Pipeline`](@ref) drives them through these two verbs, exactly as it drives prior estimators through [`prior`](@ref) or optimisers through [`optimise`](@ref).

```@docs
AbstractPreprocessingEstimator
AbstractPricesPreprocessingEstimator
AbstractReturnsPreprocessingEstimator
AbstractPreprocessingResult
AbstractPricesPreprocessingResult
AbstractReturnsPreprocessingResult
fit_preprocessing
apply_preprocessing
is_missing_value
```

```@docs
PricesToReturns
MissingDataFilter
MissingDataFilterResult
Imputer
ImputerResult
```

## Asset selection

Asset selectors are the returns-level preprocessing subfamily that restricts the *asset universe*. The universe chosen on the training window is the selector's fitted state, so a selector is safe inside cross-validation. The concrete selectors live in [Asset selection](@ref); this is the seam they share.

```@docs
AbstractAssetSelector
AssetSelectorResult
select_assets
find_complete_indices
```
