```@meta
Description = "Panel round trip, public API of PortfolioOptimisers.jl: panel_manifest, asset_panel."
```

# Panel round trip

## Read a panel back from its tables

[`panel_dataframe`](@ref) writes the values and the masks of an [`AssetPanel`](@ref) as a table.
`panel_manifest` writes the parts that a table column cannot hold: the axes, the kind of each
field, the order of the levels, and the axis name, the labels and the groups of a tensor field.
`asset_panel` reads the panel back from the two tables, after they go through any table format,
for example CSV.jl or Arrow.jl.

```@docs
panel_manifest
asset_panel(::DataFrames.DataFrame, ::DataFrames.DataFrame)
```
