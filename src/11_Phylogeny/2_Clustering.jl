"""
    AbstractClusteringEstimator <: AbstractPhylogenyEstimator

Abstract supertype for all clustering estimator types in PortfolioOptimisers.jl.

All concrete types implementing clustering-based estimation algorithms should subtype `AbstractClusteringEstimator`. This enables a consistent interface for clustering estimators throughout the package.

# Related

  - [`AbstractClusteringAlgorithm`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClusteringEstimator <: AbstractPhylogenyEstimator end

"""
    AbstractClusteringAlgorithm <: AbstractPhylogenyAlgorithm

Abstract supertype for all clustering algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific clustering algorithms should subtype `AbstractClusteringAlgorithm`. This enables flexible extension and dispatch of clustering routines.

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClusteringAlgorithm <: AbstractPhylogenyAlgorithm end

"""
    AbstractOptimalNumberClustersEstimator <: AbstractEstimator

Abstract supertype for all optimal number of clusters estimator types in PortfolioOptimisers.jl.

All concrete types implementing algorithms to estimate the optimal number of clusters should subtype `AbstractOptimalNumberClustersEstimator`. This enables a consistent interface for cluster number estimation.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end

"""
    AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm

Abstract supertype for all optimal number of clusters algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific algorithms for determining the optimal number of clusters should subtype `AbstractOptimalNumberClustersAlgorithm`. This enables flexible extension and dispatch of cluster number selection routines.

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end

"""
    AbstractClusteringResult <: AbstractPhylogenyResult

Abstract supertype for all clustering result types in PortfolioOptimisers.jl.

All concrete types representing the result of a clustering estimation should subtype `AbstractClusteringResult`. This enables a consistent interface for clustering results throughout the package.

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractClusteringAlgorithm`](@ref)
"""
abstract type AbstractClusteringResult <: AbstractPhylogenyResult end

"""
    HierarchicalClustering{T1, T2, T3, T4} <: AbstractClusteringResult

Result type for hierarchical clustering in PortfolioOptimisers.jl.

`HierarchicalClustering` stores the output of a hierarchical clustering algorithm, including the clustering object, similarity and distance matrices, and the number of clusters.

# Fields

  - `clustering`: The hierarchical clustering object (e.g., `Clustering.Hclust`).
  - `S`: Similarity matrix used for clustering.
  - `D`: Distance matrix used for clustering.
  - `k`: Number of clusters.

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`ClusteringEstimator`](@ref)
"""
struct HierarchicalClustering{T1, T2, T3, T4} <: AbstractClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
end
function HierarchicalClustering(; clustering::Clustering.Hclust, S::AbstractMatrix,
                                D::AbstractMatrix, k::Integer)
    @argcheck(!isempty(S) && !isempty(D) && size(S) == size(D) && k >= one(k),
              AssertionError("The following conditions must hold:\n`S` must be non-empty => $(!isempty(S))\n`D` must be non-empty => $(!isempty(D))\n`S` and `D` must have the same size => $(size(S) == size(D))\nk must be greater than or equal to 1 => $k"))
    return HierarchicalClustering(clustering, S, D, k)
end
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
struct SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm end
struct PredefinedNumberClusters{T1} <: AbstractOptimalNumberClustersAlgorithm
    k::T1
end
function PredefinedNumberClusters(; k::Integer = 1)
    @argcheck(k >= one(k), DomainError("`k` must be greater than or equal to 1:\nk => $k"))
    return PredefinedNumberClusters(k)
end
struct StandardisedSilhouetteScore{T1} <: AbstractOptimalNumberClustersAlgorithm
    metric::T1
end
function StandardisedSilhouetteScore(;
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing)
    return StandardisedSilhouetteScore(metric)
end
struct OptimalNumberClusters{T1, T2} <: AbstractOptimalNumberClustersEstimator
    max_k::T1
    alg::T2
end
function OptimalNumberClusters(; max_k::Union{Nothing, <:Integer} = nothing,
                               alg::AbstractOptimalNumberClustersAlgorithm = SecondOrderDifference())
    if !isnothing(max_k)
        @argcheck(max_k >= one(max_k),
                  DomainError("`max_k` must be greater than or equal to 1:\nmax_k => $max_k"))
    end
    return OptimalNumberClusters(max_k, alg)
end
struct HClustAlgorithm{T1} <: AbstractClusteringAlgorithm
    linkage::T1
end
function HClustAlgorithm(; linkage::Symbol = :ward)
    return HClustAlgorithm(linkage)
end
struct ClusteringEstimator{T1, T2, T3, T4} <: AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    onc::T4
end
function ClusteringEstimator(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             de::AbstractDistanceEstimator = Distance(;
                                                                      alg = CanonicalDistance()),
                             alg::AbstractClusteringAlgorithm = HClustAlgorithm(),
                             onc::Union{<:Integer,
                                        <:AbstractOptimalNumberClustersEstimator} = OptimalNumberClusters())
    return ClusteringEstimator(ce, de, alg, onc)
end

export clusterise, SecondOrderDifference, PredefinedNumberClusters,
       StandardisedSilhouetteScore, OptimalNumberClusters, HClustAlgorithm,
       ClusteringEstimator
