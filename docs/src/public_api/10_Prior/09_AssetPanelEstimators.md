```@meta
Description = "Asset Panel Estimators, public API of PortfolioOptimisers.jl: AbstractPhylogenyFeatureAlgorithm, Proximity, RegressionPanel, PhylogenyPanel, …"
```

# Asset Panel Estimators

An asset panel estimator builds a static [`AssetPanel`](@ref) each time a distance needs one. [`FeatureDistance`](@ref) holds one in its `ape` field. The estimator builds the panel from the prior result and the returns of the optimisation that runs it, which inside a nested optimisation is the inner one. `RegressionPanel` takes the factor loadings that a factor prior fitted. `PhylogenyPanel` takes the closeness of each pair of assets in a network or a clustering of the assets. It reads no prior, so it also works before a prior exists, as in asset selection. A new phylogeny feature rule subtypes `AbstractPhylogenyFeatureAlgorithm` and adds a method of `phylogeny_features`.

`WalkProfilePanel` takes the walk profile of each asset over an exogenous graph of the returns data: the nodes that the walks of a few hops from the asset reach, weighted by a decay of the number of hops. `WalkDistance` measures the profiles. The panel cannot read an asset that is not a node of a graph, or whose profile is zero, and `readable_assets` names those assets before a fit, so the fit drops them.

```@docs
PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm
Proximity
RegressionPanel
PhylogenyPanel
phylogeny_features
asset_panel(ape::RegressionPanel, pr, rd, ::Any)
```

## Walk profiles

```@docs
WalkProfilePanel
PortfolioOptimisers.AbstractEdgeDirection
OutEdges
InEdges
BothEdges
PortfolioOptimisers.walk_parts
PortfolioOptimisers.AbstractEdgeNormalisation
WeightShare
PortfolioOptimisers.normalise_edge_weights
PortfolioOptimisers.AbstractEdgeWeightRule
UnitEdgeWeights
PortfolioOptimisers.read_edge_weights
PortfolioOptimisers.readable_assets
```
