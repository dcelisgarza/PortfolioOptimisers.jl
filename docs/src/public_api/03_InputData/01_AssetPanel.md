```@meta
Description = "The Asset Panel, public API of PortfolioOptimisers.jl: AbstractPanelField, AssetPanel, NumericPanelField, CategoricalPanelField, TensorPanelField, …"
```

# The Asset Panel

## The asset panel

An asset panel holds data about each asset next to its returns or prices, such as a market
capitalisation, a sector classification or a table of factor exposures. [`ReturnsResult`](@ref)
and [`PricesResult`](@ref) keep it in their `pnl` field. Each field of the panel holds its own
values and its own observed mask, which marks the cells that held data. A distance that compares
assets by these fields reads a feature matrix, and [`feature_matrix`](@ref) builds that matrix from
the panel each time a distance needs it. No object stores the matrix.

A feature selector names the panel fields that the matrix stacks. Each entry of a selector names
one of four things: a whole panel field, a field with the levels or labels to keep, a field with
one level or label, or the observed mask of a field. [`feature_labels`](@ref) names each column of
the matrix with the entry that selects that column alone. Pass the labels back as a selector, and you
get the same matrix.

A panel is static or time-varying. A static panel has no observation axis and no universe
masks. A time-varying panel adds an observation axis, and it carries the active mask
and the estimation mask.

[`asset_panel`](@ref) builds a panel from raw fields that can hold blank cells. It fills every blank
by the fill policy of its field, so every field of the panel is finite. A field that can hold
blanks carries an observed mask, which is `false` at each cell that the fill wrote.

The library has no file format for a panel. [`panel_dataframe`](@ref) converts a panel to a
`DataFrames.DataFrame`, which you can write with any tool you already use. Name one panel field,
and you get that field in the layout it has. The `:long` layout gives one row per
`(observation, asset)` pair where the asset is active. The `:wide` layout gives one row per
observation and one column per `(panel field column, asset)` pair, and keeps every cell. A
[`TensorPanelField`](@ref) gives one column per label of its last axis, named `"<field>=<label>"`,
as in a feature matrix.

## Types

```@docs
AbstractPanelField
AssetPanel
NumericPanelField
CategoricalPanelField
TensorPanelField
```

## Functions

```@docs
panel_field
panel_feature_matrix
port_opt_view(::AssetPanel, ::Any)
panel_field_axes
panel_field_labels
panel_field_stack!
panel_field_view
```
