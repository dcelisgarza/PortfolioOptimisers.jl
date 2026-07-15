# Pre-processing

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
check_names_and_returns_matrix
prices_to_returns
port_opt_view(::ReturnsResult, ::Any)
returns_result_picker
Prices_RR
```

## Price-level data

```@docs
AbstractPricesResult
PricesResult
port_opt_view(pr::PricesResult, ::Colon, ::Colon)
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

## Train/test splitting

A **holdout split** reserves the tail of the time-ordered observations as a test window and trains on the head. It comes in two forms: the free function [`train_test_split`](@ref), which cuts data into a train/test pair, and the estimator [`TrainTestSplit`](@ref) (alias `TTS`), which carries the protocol *inside* a [`Pipeline`](@ref) as its first step — so every fitted step downstream sees the training window alone, and `fit_predict(pipe, data)` evaluates on the held-out window in one line.

Sizes are row counts (`Integer`) or fractions of the observations (`AbstractFloat` in `(0, 1)`). Giving one side makes the other its complement; giving both **embargoes** the rows between the two windows. See `docs/adr/0031-holdout-split-as-a-pipeline-step.md`.

The keyword form returns a bare `(train, test)` tuple; the estimator form, `train_test_split(tts, data)`, returns the same [`TrainTestSplitResult`](@ref) a pipeline's split step produces, so one configured holdout can be reused inside and outside a pipeline.

```@docs
train_test_split
TrainTestSplit
TrainTestSplitResult
PortfolioOptimisers.safe_index
PortfolioOptimisers.split_count
```

## Asset selection infrastructure

Asset selectors are the returns-level preprocessing subfamily that restricts the *asset universe*. The universe chosen on the training window is the selector's fitted state, so a selector is safe inside cross-validation. The concrete selectors live in [Asset selection](@ref); this is the seam they share.

```@docs
AbstractAssetSelector
AssetSelectorResult
select_assets
find_complete_indices
```
