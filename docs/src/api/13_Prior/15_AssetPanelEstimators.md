# Asset Panel Estimators

An Asset Panel estimator is a **producer**: [`FeatureDistance`](@ref) holds one in its `ape` slot, and it builds a static [`AssetPanel`](@ref) at the point of use, from the prior result and the returns of the subproblem that runs it.

```@docs
AbstractPhylogenyFeatureAlgorithm
Proximity
PortfolioOptimisers._proximity_features
phylogeny_features
PortfolioOptimisers.panel_axis_labels
PortfolioOptimisers.carrier_asset_names
PortfolioOptimisers.regression_factor_names
RegressionPanel
PhylogenyPanel
asset_panel(ape::RegressionPanel, pr, rd, ::Any)
PortfolioOptimisers.assert_producer_prior
```
