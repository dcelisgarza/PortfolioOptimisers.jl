```@meta
Description = "Base asset selection, public API of PortfolioOptimisers.jl: AssetSelectorResult, select_assets."
```

# Base asset selection

## What every asset selector shares

An asset selector is a preprocessing estimator that removes assets from the returns. It chooses the assets on the training window, and keeps the same assets on every later window. The selectors are on the [Asset selection](@ref) page. This page has the result type they all return, `AssetSelectorResult`, and `select_assets`, the one method a new selector must write.

## Types

```@docs
AssetSelectorResult
```

## Functions

```@docs
select_assets
```
