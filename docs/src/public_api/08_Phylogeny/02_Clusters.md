```@meta
Description = "Clusters, public API of PortfolioOptimisers.jl: Clusters, SecondOrderDifference, SilhouetteScore, OptimalNumberClusters, AbstractLinkage, SingleLinkage, …"
```

# Clusters

```@docs
Clusters
SecondOrderDifference
SilhouetteScore
OptimalNumberClusters
AbstractLinkage
SingleLinkage
CompleteLinkage
AverageLinkage
WeightedLinkage
CentroidLinkage
MedianLinkage
WardLinkage
linkage_symbol
HClustAlgorithm
ClustersEstimator
AbstractBranchOrder
OptimalLeafOrder
BarJosephOrder
ROrder
clusterise(cle::AbstractClusteringResult, args...; kwargs...)
factory(alg::AbstractClustersAlgorithm, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
