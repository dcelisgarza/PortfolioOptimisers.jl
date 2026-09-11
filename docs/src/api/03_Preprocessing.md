# Pre-processing

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
check_names_and_returns_matrix
feature_row_indices
PortfolioOptimisers.matched_row_indices
PortfolioOptimisers.panel_feature_names
PortfolioOptimisers.panel_carrier_view
PortfolioOptimisers.asset_panel(::Nothing, ::Any, ::ReturnsResult, ::Any)
PortfolioOptimisers.assert_asset_panel_supplied
PortfolioOptimisers.project_panel_clock
PortfolioOptimisers.append_carrier_block!
prices_to_returns
PortfolioOptimisers.AbstractGapReturnAlgorithm
CatchUpGapReturn
PortfolioOptimisers.gap_return
PortfolioOptimisers.gap_return_writable
PortfolioOptimisers.gap_return_value
PortfolioOptimisers.apply_gap_return
port_opt_view(::ReturnsResult, ::Any)
returns_result_picker
Prices_RR
```

!!! note "An absent price has one spelling"
    The conversion unifies `missing` and `NaN` as `NaN` and carries the gap into the returns; it
    deletes nothing. Filling a gap is [`PriceGapFill`](@ref)'s, bounded by the **Listing Span**, and
    deleting an observation or an asset is [`MissingDataFilter`](@ref)'s — both of them fitted steps,
    because a **Universe Policy** is fitted on a training window and replayed by name. See
    `docs/adr/0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md`.

## The fold context of the online step

An optimiser's online step forwards each observation to its prior and records the rest of the
carrier in a [`PortfolioOptimisers.ReturnsBufferState`](@ref): the factor, benchmark and
timestamp columns as buffers of their own, and the names and the static Asset Panel pinned by
the first step. The returns are owned once, by the prior, and this state holds them only where
no prior sits beneath the optimiser. [`PortfolioOptimisers.returns_result`](@ref) rebuilds the
[`ReturnsResult`](@ref) a batch fit over the same observations would have read.

```@docs
PortfolioOptimisers.ReturnsBufferState
PortfolioOptimisers.assert_pinned_context
PortfolioOptimisers.pinned_agree
PortfolioOptimisers.pinned_repr
PortfolioOptimisers.assert_column_presence
PortfolioOptimisers.context_count
PortfolioOptimisers.column_count
PortfolioOptimisers.fold_column
PortfolioOptimisers.fold_column_masked
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.ReturnsBufferState, rd::ReturnsResult; own_returns::Bool = false)
PortfolioOptimisers.returns_result(state::PortfolioOptimisers.ReturnsBufferState, rows::PortfolioOptimisers.SampleBufferState)
PortfolioOptimisers.column_matrix
PortfolioOptimisers.merge_states(a::PortfolioOptimisers.ReturnsBufferState, b::PortfolioOptimisers.ReturnsBufferState)
PortfolioOptimisers.merge_column
Base.copy(x::PortfolioOptimisers.ReturnsBufferState)
PortfolioOptimisers.copy_column
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.ReturnsBufferState, i, args...)
```

## The Listing Span

A **Listing Span** is the interval of the price clock over which an asset is listed, one per asset
column. The **Span Rule** derives it from the position of a gap: a *leading* run of gaps is an asset
not yet listed, a *trailing* run is a delisting, and an *interior* gap is a suspension or a holiday
on an asset that is still listed and still held. So a caller holding only prices can state a
universe, because the position already carries the distinction a listing calendar would supply.

[`listing_span`](@ref) derives the span, and [`universe_masks`](@ref) projects it onto the returns
clock and intersects it with finiteness, giving the two universe masks an [`AssetPanel`](@ref)
carries. A return consumes the *earlier* price of its pair, so the projection is `[first + 1, last]`
under padding and `[first, last - 1]` without it — the active mask bounds exactly the run of a
column's finite returns, and an inception emits no **Held Gap**. The two masks then differ only
inside an interior gap, which is what the fold reports and zeroes.

Both verbs work on bare arrays and carry no estimator. The type bound is `AbstractMatrix{Bool}`, so
a caller's own declaration — a listing calendar, or a constituency that leaves and rejoins — enters
at the same point and replaces the derived active mask outright. The estimation mask is never the
caller's to state and is always re-derived, which is what makes `emsk ⊆ amsk` hold by construction.
See `docs/adr/0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md`
and `docs/adr/0131-a-return-needs-two-consecutive-prices-and-a-gap-return-writes-only-the-cells-that-lack-them.md`.

```@docs
listing_span
universe_masks
PortfolioOptimisers.ListingSpan
PortfolioOptimisers.project_span
```

## The ingestion layer

[`PriceIngestion`](@ref) assembles raw price series into the carrier the conversion reads. It runs
**once on the whole panel** and is deliberately not a [`Pipeline`](@ref) step: unification of the
two absent-price conventions, the factor and benchmark join, and the frequency collapse each move
or renumber the observations, folds are cut once on the carrier's clock, and a hyperparameter that
changes the test set cannot be scored against one that does not. Running outside the `Pipeline` is
also what lets the **Span Rule** read the whole panel, which a step — seeing only a window — cannot.

The carrier it emits holds the Listing Span in its `span` field. [`PricesToReturns`](@ref) projects
that onto the returns clock and hands the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) stating
the universe — **always**, a gapless panel included, so `pnl === nothing` on a returns carrier means
one thing only: the carrier was not built by the layer. The gapless case costs nothing to say: both
masks become a [`PortfolioOptimisers.AllTrueMask`](@ref), which stores no cell.

The layer spells an absent price `NaN`, which is the library's one spelling for absence, and the
conversion carries it into the returns rather than deleting the observation or the asset that holds
one. A carrier the layer did not build states no span, and then no universe: the conversion does not
guess one from the window, because a delisting straddling the window end reads there as an asset
that was never listed. See
`docs/adr/0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md`
and
`docs/adr/0132-the-layer-emits-one-carrier-and-alignment-splits-into-a-fixed-axis-and-a-provenance-check.md`.

**The asset table states the clock.** Under the default `join_method = :left` the factor,
benchmark and implied-volatility series are aligned to the asset clock and padded `NaN` where they
are silent, so `timestamp(pr.X) == timestamp(X)` unless `collapse_args` is non-empty; `:outer` and
`:inner` stay reachable. What the join padded is named — which table, how many observations, which
columns — warning by default and refusing under `strict`. The value type the carrier holds is
derived from the series rather than named: a `Float32` panel stays `Float32`, `Float32` beside
`Float64` joins in `Float64`, an integer panel takes the floating-point type the return arithmetic
gives it, and a type that cannot spell an absence is refused by name at the first gap it would have
to spell. An absent implied volatility is carried like an absent price, and
[`ImpliedVolatility`](@ref) narrows its Coverage Universe to the columns whose implied
volatilities are complete. See
`docs/adr/0135-the-asset-table-states-the-clock-and-the-layer-carries-every-absence-and-names-it.md`.

```@docs
PriceIngestion
price_ingestion
PortfolioOptimisers.unify_gaps
PortfolioOptimisers.series_value_type
PortfolioOptimisers.absence_type
PortfolioOptimisers.absent_value
PortfolioOptimisers.assert_pad_spellable
PortfolioOptimisers.align_series
PortfolioOptimisers.padded_observations
PortfolioOptimisers.padding_report_line
PortfolioOptimisers.assert_join_padding
PortfolioOptimisers.series_names
PortfolioOptimisers.assert_disjoint_series_names
PortfolioOptimisers.assert_unreserved_series_names
PortfolioOptimisers.assert_distinct_series_names
PortfolioOptimisers.assert_span_shape
PortfolioOptimisers.returns_universe_masks
PortfolioOptimisers.compress_all_true
PortfolioOptimisers.attach_universe_masks
PortfolioOptimisers.AllTrueMask
PortfolioOptimisers.span_carrier_view
```

## The price gap fill

A **Held Price** is the last priced observation of an asset, carried forward across a gap. Stating
one is what [`PriceGapFill`](@ref) is for: it is the ingestion layer's only fill, it is off unless a
caller adds it, and it exists to state a *price convention* across a suspension rather than to
remove a gap — a gap is carried through the conversion, so nothing downstream needs it gone.

The fill is bounded by the **Listing Span**, so it touches Held Gaps alone and can never fabricate a
price where an asset was not yet listed or has been delisted. It is fitted on a training window and
replayed: [`CarriedPrice`](@ref) records the last observed training price, which seeds a
carry-forward on a window that opens inside a gap, and a [`Num_VecToScaM`](@ref) records the
reduction of that window's observed prices. A per-asset constant manufactures two moves the market
never printed, which is why the carried price is the convention a caller reaching for a fill
normally wants.

It runs at the price level, before [`PricesToReturns`](@ref), because a carried price cannot be
stated after the conversion: zeroing the returns a gap left non-finite discards the move across the
gap entirely. A filled cell is therefore finite in the returns and its estimation mask entry is
`true` — a caller who filled has said the asset traded. See
`docs/adr/0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md`.

```@docs
CarriedPrice
PriceGapFill
PriceGapFillResult
PortfolioOptimisers.carrier_listing_span
PortfolioOptimisers.gap_fill_span
PortfolioOptimisers.gap_fill_seed
PortfolioOptimisers.gap_fill_column!
```

## The Asset Panel

A **point-in-time panel** of per-asset fields — market capitalisation, a sector classification, a
factor exposure tensor — is what [`ReturnsResult`](@ref) and [`PricesResult`](@ref) carry in their
`pnl` slot. Each **Panel Field** owns its own values and its own observed mask, so the panel *is*
the feature data: no carrier holds a feature matrix beside it, and the Feature Matrix a distance
measures is derived by [`feature_matrix`](@ref) and stored nowhere.

A **Feature Selector** says which Panel Fields the matrix stacks. An entry names one Panel Field,
one field with the levels or labels it keeps, one field with a single level or label, or one
field's observed mask. [`feature_labels`](@ref) names each resulting column with the entry that
selects exactly it, so a label vector is itself a selector that rebuilds the same matrix.

A panel takes one of two shapes. A **static** panel indexes its Panel Fields by asset alone and
carries no universe mask; a **time-varying** panel prepends an observation axis and carries both.
The shape rides the type parameters, so a mask consumer dispatches rather than branches.

A blank cell never reaches a carrier. [`asset_panel`](@ref) resolves every one of them, so every
Panel Field comes out finite, and each Panel Field that can blank carries the observed mask that
says which cells the resolution touched.

The library persists no panel of its own, and it needs no format to: [`panel_dataframe`](@ref)
renders a panel as a `DataFrames.DataFrame`, and a caller writes that with whatever they already
use. One Panel Field name gives that field laid out as it stands, a `:long` layout gives one row
per `(observation, asset)` filtered by the active mask, and a `:wide` layout gives one column per
`(Panel Field column, asset)` and keeps every cell. A [`TensorPanelField`](@ref) spreads into one
column per trailing-axis label there, under the same `"<field>=<label>"` name it takes in a
Feature Matrix.

```@docs
AssetPanel
asset_panel
panel_field
PortfolioOptimisers.panel_axes
panel_feature_matrix
panel_dataframe
PortfolioOptimisers.panel_frame_columns
PortfolioOptimisers.panel_frame_fields
PortfolioOptimisers.panel_frame_assets
PortfolioOptimisers.panel_frame_block!
PortfolioOptimisers.panel_frame_field
PortfolioOptimisers.panel_frame_long
PortfolioOptimisers.panel_frame_wide
PortfolioOptimisers.features_are_assets
PortfolioOptimisers.panel_onehot
PortfolioOptimisers.RepeatedLeading
PortfolioOptimisers.panel_field_lift
PortfolioOptimisers.panel_build_observations
PortfolioOptimisers.AbstractAssetPanelEstimator
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
PortfolioOptimisers.panel_value_eltype
PortfolioOptimisers.panel_field_labels
PortfolioOptimisers.panel_field_observed_labels
PortfolioOptimisers.panel_field_stack!
PortfolioOptimisers.panel_field_stack_observed!
PortfolioOptimisers.panel_field_view
PortfolioOptimisers.panel_field_keys
PortfolioOptimisers.panel_value_columns!
PortfolioOptimisers.panel_key_column!
PortfolioOptimisers.panel_column_label
PortfolioOptimisers.panel_field_value_column!
PortfolioOptimisers.panel_field_observed_column!
PortfolioOptimisers.select_fields
PortfolioOptimisers.select_fields_push!
PortfolioOptimisers.panel_selector_msg
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
PortfolioOptimisers.assert_feature_selector
PortfolioOptimisers.assert_selector_entry
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
PortfolioOptimisers.assert_nonneg_where_present
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
