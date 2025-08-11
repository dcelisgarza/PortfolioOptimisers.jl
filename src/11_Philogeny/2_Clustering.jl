abstract type AbstractClusteringEstimator <: AbstractPhilogenyEstimator end
abstract type AbstractClusteringAlgorithm <: AbstractPhilogenyAlgorithm end
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
abstract type AbstractClusteringResult <: AbstractPhilogenyResult end
struct HierarchicalClustering{T1, T2, T3, T4} <: AbstractClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
end
function HierarchicalClustering(; clustering::Clustering.Hclust, S::AbstractMatrix,
                                D::AbstractMatrix, k::Integer)
    @assert(!isempty(S) && !isempty(D) && size(S) == size(D) && k >= one(k),
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
    @assert(k >= one(k), DomainError("`k` must be greater than or equal to 1:\nk => $k"))
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
        @assert(max_k >= one(max_k),
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
