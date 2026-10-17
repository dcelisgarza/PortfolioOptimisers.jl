```@meta
Description = "Asset panel summary, public API of PortfolioOptimisers.jl: DataFrames.describe, panel_info, panel_align_active."
```

# Asset panel summary

## Report the missing cells of a panel

`describe` gives one row for each field of an [`AssetPanel`](@ref), with the share of its cells
that the builder filled, over all cells and over the active cells. With `by`, it gives one row for
each field and each level of a categorical field. `panel_info` prints the dimensions of the panel,
the coverage of its two universe masks, the same shares, and the levels of each categorical field.

```@docs
DataFrames.describe(pnl::AssetPanel)
panel_info
```

## Align the active mask to the fields

`panel_align_active` moves the start of each asset to the first active observation where each
named field is observed, and returns the new panel and the count of removed cells.

```@docs
panel_align_active
```
