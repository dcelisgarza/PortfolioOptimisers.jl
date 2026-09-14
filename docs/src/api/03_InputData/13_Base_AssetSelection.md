# Base asset selection

## Asset selection infrastructure

Asset selectors are the returns-level preprocessing subfamily that restricts the *asset universe*. The universe chosen on the training window is the selector's fitted state, so a selector is safe inside cross-validation. The concrete selectors live in [Asset selection](@ref); this is the seam they share.

```@docs
AbstractAssetSelector
AssetSelectorResult
select_assets
find_complete_indices
```
