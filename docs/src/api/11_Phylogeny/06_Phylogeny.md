# Phylogeny

```@docs
PhylogenyResult
BetweennessCentrality
ClosenessCentrality
DegreeCentrality
EigenvectorCentrality
KatzCentrality
Pagerank
RadialityCentrality
StressCentrality
KruskalTree
BoruvkaTree
PrimTree
NetworkEstimator
NetworkClustersEstimator
_clusterise(alg::HClustAlgorithm, onc::AbstractOptimalNumberClustersEstimator,
                    S::MatNum, D::MatNum, P::MatNum; branchorder::Symbol = :optimal)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractTreeType,
                                                                     <:HopCount}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
CentralityEstimator
_phylogeny_matrix
phylogeny_matrix
centrality_vector
average_centrality
asset_phylogeny
AbstractCentralityAlgorithm
AbstractTreeType
calc_mst
AbstractNetworkEstimator
AbstractCentralityEstimator
graph_weight_matrix
calc_weighted_adjacency_graph
calc_weighted_adjacency
calc_adjacency
calc_distance_weighted_graph
PortfolioOptimisers.separation_graph
separation_matrix
separation_budget
separation_quantile
HopCountQuantile
PathLengthQuantile
resolve_separation
AbstractCentralityPolarity
DistancePolarity
SimilarityPolarity
TopologyOnly
centrality_polarity
centrality_graph
assert_no_weight_channel_args
assert_centrality_args
assert_tree_args
calc_centrality
Tree_SimMat
NwE_ClE
NwE_ClE_Cl
NwE_Pl_ClE_Cl
HClE_HCl
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
