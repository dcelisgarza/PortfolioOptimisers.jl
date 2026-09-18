```@meta
Description = "Base asset selection, public API of PortfolioOptimisers.jl: AssetSelectorResult, select_assets."
```

# Base asset selection

## Asset selection infrastructure

Asset selectors are the returns-level preprocessing subfamily that restricts the *asset universe*. The universe chosen on the training window is the selector's fitted state, so a selector is safe inside cross-validation. The concrete selectors live in [Asset selection](@ref); this is the seam they share.

## Types

```@docs
AssetSelectorResult
```

## Functions

```@docs
select_assets
```
