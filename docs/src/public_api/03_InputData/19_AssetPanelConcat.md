```@meta
Description = "Asset panel concat, public API of PortfolioOptimisers.jl: Base.vcat."
```

# Asset panel concat

## Join panels along the observation axis

`vcat` joins [`AssetPanel`](@ref)s of one universe, for example the panel of 2024 and the panel of
2025. It joins the values and the observed mask of each Panel Field, and both universe masks. The
parts must hold the same assets, the same Panel Fields in the same order, and the same schema for
each field. The schema of a categorical field is its levels. The schema of a tensor field is its
axis name, its labels and its groups. A part that differs throws an error that names the part and
the field.

```@docs
Base.vcat(a::AssetPanel, bs::AssetPanel...)
```
