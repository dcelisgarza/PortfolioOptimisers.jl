abstract type ClusteringAlgorithm end
abstract type AbstractClusteringEstimator <: PhilogenyEstimator end
abstract type AbstractPortfolioOptimisersClusteringResult <: Clustering.ClusteringResult end
abstract type NumberClustersHeuristic end
function optimal_number_clusters end
function clusterise end
function clusterise(cle::AbstractPortfolioOptimisersClusteringResult, args...; kwargs...)
    return cle
end
struct HierarchicalClustering{T1 <: Symbol} <: ClusteringAlgorithm
    linkage::T1
end

export clusterise
