# Non hierarchical clustering

```@docs
KMeansAlgorithm
factory(alg::KMeansAlgorithm, w::StatsBase.AbstractWeights)
clusterise(cle::ClustersEstimator{<:Any, <:Any, <:AbstractNonHierarchicalClusteringAlgorithm, <:Any}, X::MatNum; dims::Int = 1, kwargs...)
assignments(clr::Clusters{<:Clustering.ClusteringResult, <:Any, <:Any, <:Any})
_get_k_clusters_from_alg
```
