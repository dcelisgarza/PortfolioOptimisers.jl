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
                                                                     <:Any}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractSimilarityMatrixAlgorithm,
                                                                     <:Any}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
CentralityEstimator
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
calc_adjacency
separation_matrix
separation_budget
calc_centrality
Tree_SimMat
NwE_ClE
NwE_ClE_Cl
NwE_Pl_ClE_Cl
HClE_HCl
```
