abstract type ClusteringAlgorithm end
abstract type AbstractClusteringEstimator end
abstract type AbstractPortfolioOptimisersClusteringResult <: Clustering.ClusteringResult end
abstract type NumberClustersHeuristic end
function optimal_number_clusters end
function clusterise end
function clusterise(cle::AbstractPortfolioOptimisersClusteringResult, args...; kwargs...)
    return cle
end

export clusterise
