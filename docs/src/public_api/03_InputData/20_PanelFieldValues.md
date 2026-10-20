```@meta
Description = "Panel field values, public API of PortfolioOptimisers.jl: panel_field_values."
```

# Panel field values

## Read a panel field at a policy for its inactive cells

An [`AssetPanel`](@ref) keeps a finite value in every cell of a panel field, and its masks say
which cells are in the universe and which cells a fill policy wrote. `panel_field_values` reads a
copy of the values and writes a value of your choice into the inactive cells and into the
unobserved cells, for example `NaN` for code that reads no mask. The panel does not change.

```@docs
panel_field_values
```
