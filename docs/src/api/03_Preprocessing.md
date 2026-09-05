# Pre-processing

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
check_names_and_returns_matrix
features_are_assets
feature_matrix_view
feature_row_indices
PortfolioOptimisers.matched_row_indices
PortfolioOptimisers.carrier_feature_matrix
PortfolioOptimisers.collapsed_asset_panel
PortfolioOptimisers.panel_feature_names
PortfolioOptimisers.panel_carrier_view
prices_to_returns
port_opt_view(::ReturnsResult, ::Any)
returns_result_picker
Prices_RR
PortfolioOptimisers.apply_impute_method
```

!!! note "`Impute` is optional; `Imputer` is not `Impute`"
    The `impute_method` keyword of [`prices_to_returns`](@ref) takes an
    [`Impute.jl`](https://github.com/invenia/Impute.jl) imputor, and `Impute` is a **weak**
    dependency: add it to your project and `using Impute` to load `PortfolioOptimisersImputeExt`,
    which supplies the method. Without it, passing an imputor raises an `ArgumentError` rather than
    working silently. This is unrelated to [`Imputer`](@ref), PortfolioOptimisers' own imputation
    estimator for pipelines, despite the similar name. See
    `docs/adr/0042-impute-is-a-weak-dependency.md`.

## The Asset Panel

A **point-in-time panel** of per-asset fields — market capitalisation, a sector classification, a
factor exposure tensor — is what [`ReturnsResult`](@ref) and [`PricesResult`](@ref) carry in their
`pnl` slot. Each **Panel Field** owns its own values and its own observed mask, so the panel *is*
the feature data: no carrier holds a feature matrix beside it, and the Feature Matrix a distance
measures is derived by [`panel_feature_matrix`](@ref) and stored nowhere.

A panel takes one of two shapes. A **static** panel indexes its Panel Fields by asset alone and
carries no universe mask; a **time-varying** panel prepends an observation axis and carries both.
The shape rides the type parameters, so a mask consumer dispatches rather than branches.

A blank cell never reaches a carrier. [`asset_panel`](@ref) resolves every one of them, so every
Panel Field comes out finite, and each Panel Field that can blank carries the observed mask that
says which cells the resolution touched.

```@docs
AssetPanel
asset_panel
panel_field
panel_feature_matrix
feature_matrix_panel
port_opt_view(::AssetPanel, ::Any)
PortfolioOptimisers.AbstractPanelField
NumericPanelField
CategoricalPanelField
TensorPanelField
PortfolioOptimisers.AbstractPanelFieldInput
NumericPanelInput
CategoricalPanelInput
TensorPanelInput
PortfolioOptimisers.AbstractPanelFillAlgorithm
NoPanelFill
ConstantPanelFill
ForwardPanelFill
BackwardPanelFill
PortfolioOptimisers.panel_is_static
PortfolioOptimisers.panel_field_axes
PortfolioOptimisers.panel_field_labels
PortfolioOptimisers.panel_field_observed_labels
PortfolioOptimisers.panel_field_stack!
PortfolioOptimisers.panel_field_stack_observed!
PortfolioOptimisers.panel_field_view
PortfolioOptimisers.panel_array_view
PortfolioOptimisers.panel_tensor_view
PortfolioOptimisers.panel_mask_view
PortfolioOptimisers.panel_claim!
PortfolioOptimisers.panel_fill
PortfolioOptimisers.panel_fill_array
PortfolioOptimisers.panel_directional_fill
PortfolioOptimisers.panel_resolve
PortfolioOptimisers.panel_input_field
PortfolioOptimisers.panel_input_is_static
PortfolioOptimisers.is_panel_blank
PortfolioOptimisers.check_asset_panel
PortfolioOptimisers.assert_panel_labels
PortfolioOptimisers.assert_panel_field_name
PortfolioOptimisers.assert_panel_field_shape
PortfolioOptimisers.assert_panel_field_mask
PortfolioOptimisers.assert_panel_masks
PortfolioOptimisers.assert_feature_matrix_columns
PortfolioOptimisers.assert_panel_fill
PortfolioOptimisers.assert_panel_input
PortfolioOptimisers.assert_panel_input_fill
PortfolioOptimisers.assert_panel_finite
```

## Cross-sectional transforms

A **cross-sectional transform** rescales one observation of an `observations × assets` matrix
against the other assets of that same observation. No member reads a second observation and no
member is fitted, so a transform is configuration alone and it runs on a plain matrix.

The **estimation set** of an observation is what its statistics are computed from: the finite cells
carrying a positive benchmark weight when `w` is given, and the finite cells otherwise. A cell
outside that set is still transformed against it, so an asset the benchmark does not hold is scored
on the same scale as one it does. An observation whose estimation set is empty returns a `NaN` at
every asset.

The benchmark weights and the group labels are **arguments** of
[`cross_sectional_transform`](@ref), never fields, because one transform runs against a different
benchmark and a different classification at every call site.
[`cross_sectional_groups`](@ref) reads the labels off the codes of a
[`CategoricalPanelField`](@ref).

```@docs
PortfolioOptimisers.AbstractCrossSectionalTransform
cross_sectional_transform
cross_sectional_groups
CrossSectionalWinsoriser
CrossSectionalTanhShrinker
CrossSectionalStandardiser
CrossSectionalGaussianRank
CrossSectionalPercentileRank
PortfolioOptimisers.CS_MISSING_GROUP
PortfolioOptimisers.CS_MAD_CONSISTENCY
PortfolioOptimisers.assert_cross_sectional_matrix
PortfolioOptimisers.assert_cross_sectional_weights
PortfolioOptimisers.assert_cross_sectional_groups
PortfolioOptimisers.cross_sectional_estimation_mask
PortfolioOptimisers.cross_sectional_indices
PortfolioOptimisers.cross_sectional_weight_type
PortfolioOptimisers.cross_sectional_weighted_mean
PortfolioOptimisers.cross_sectional_equal_std
PortfolioOptimisers.cross_sectional_stat
PortfolioOptimisers.cross_sectional_blank_row!
PortfolioOptimisers.cross_sectional_zscore_row!
PortfolioOptimisers.cross_sectional_recentre_rescale!
PortfolioOptimisers.cross_sectional_midranks!
PortfolioOptimisers.cross_sectional_rank_counts
PortfolioOptimisers.cross_sectional_row_groups
PortfolioOptimisers.cross_sectional_group_split
PortfolioOptimisers.cross_sectional_percentile_ranks
PortfolioOptimisers.cross_sectional_standardise!
PortfolioOptimisers.cross_sectional_cell_stats
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
