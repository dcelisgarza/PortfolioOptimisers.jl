# Phylogeny

```@docs
PortfolioOptimisers.AbstractCentralityAlgorithm
BetweennessCentrality
ClosenessCentrality
DegreeCentrality
EigenvectorCentrality
KatzCentrality
Pagerank
RadialityCentrality
StressCentrality
PortfolioOptimisers.AbstractTreeType
KruskalTree
KruskalTree()
BoruvkaTree
BoruvkaTree()
PrimTree
PrimTree()
PortfolioOptimisers.calc_mst
PortfolioOptimisers.AbstractNetworkEstimator
Network
Network()
PortfolioOptimisers.AbstractCentralityEstimator
Centrality
Centrality()
PortfolioOptimisers.calc_adjacency
phylogeny_matrix(ne::Network, X::AbstractMatrix; dims::Int = 1, kwargs...)
phylogeny_matrix(cle::Union{<:ClusteringEstimator, <:PortfolioOptimisers.AbstractClusteringResult}, X::AbstractMatrix; branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
PortfolioOptimisers.calc_centrality
centrality_vector(ne::Network, cent::PortfolioOptimisers.AbstractCentralityAlgorithm, X::AbstractMatrix; dims::Int = 1, kwargs...)
centrality_vector(cte::Centrality, X::AbstractMatrix; dims::Int = 1, kwargs...)
average_centrality(ne::Network, cent::PortfolioOptimisers.AbstractCentralityAlgorithm, w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
average_centrality(cte::Centrality, w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
asset_phylogeny(w::AbstractVector, X::AbstractMatrix)
asset_phylogeny(cle::Union{<:Network, <:ClusteringEstimator}, w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
```
