# The Asset Panel

## The ingestion layer

```@docs
PortfolioOptimisers.AllTrueMask
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
panel_field
PortfolioOptimisers.panel_axes
panel_feature_matrix
PortfolioOptimisers.features_are_assets
PortfolioOptimisers.panel_onehot
PortfolioOptimisers.RepeatedLeading
PortfolioOptimisers.panel_field_lift
port_opt_view(::AssetPanel, ::Any)
PortfolioOptimisers.AbstractPanelField
NumericPanelField
CategoricalPanelField
TensorPanelField
PortfolioOptimisers.panel_is_static
PortfolioOptimisers.panel_field_axes
PortfolioOptimisers.panel_value_eltype
PortfolioOptimisers.panel_field_labels
PortfolioOptimisers.panel_field_observed_labels
PortfolioOptimisers.panel_field_stack!
PortfolioOptimisers.panel_field_stack_observed!
PortfolioOptimisers.panel_field_view
PortfolioOptimisers.panel_groups_view
PortfolioOptimisers.panel_array_view
PortfolioOptimisers.panel_tensor_view
PortfolioOptimisers.panel_mask_view
PortfolioOptimisers.panel_claim!
PortfolioOptimisers.check_asset_panel
PortfolioOptimisers.assert_panel_labels
PortfolioOptimisers.assert_panel_field_name
PortfolioOptimisers.assert_panel_field_shape
PortfolioOptimisers.assert_panel_field_mask
PortfolioOptimisers.assert_panel_masks
PortfolioOptimisers.assert_panel_finite
```
