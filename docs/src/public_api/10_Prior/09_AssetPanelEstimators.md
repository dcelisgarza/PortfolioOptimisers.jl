```@meta
Description = "Asset Panel Estimators, public API of PortfolioOptimisers.jl: Proximity, RegressionPanel, PhylogenyPanel, phylogeny_features, asset_panel."
```

# Asset Panel Estimators

An asset panel estimator builds a static [`AssetPanel`](@ref) each time a distance needs one. [`FeatureDistance`](@ref) holds one in its `ape` field. The estimator builds the panel from the prior result and the returns of the optimisation that runs it, which inside a nested optimisation is the inner one. `RegressionPanel` takes the factor loadings that a factor prior fitted. `PhylogenyPanel` takes the closeness of each pair of assets in a network or a clustering of the assets. It reads no prior, so it also works before a prior exists, as in asset selection.

```@docs
Proximity
RegressionPanel
PhylogenyPanel
phylogeny_features
asset_panel(ape::RegressionPanel, pr, rd, ::Any)
```
