```@meta
Description = "Asset Panel Estimators, public API of PortfolioOptimisers.jl: Proximity, RegressionPanel, PhylogenyPanel, phylogeny_features, asset_panel."
```

# Asset Panel Estimators

An Asset Panel estimator is a **producer**: [`FeatureDistance`](@ref) holds one in its `ape` slot, and it builds a static [`AssetPanel`](@ref) at the point of use, from the prior result and the returns of the subproblem that runs it.

```@docs
Proximity
RegressionPanel
PhylogenyPanel
phylogeny_features
asset_panel(ape::RegressionPanel, pr, rd, ::Any)
```
