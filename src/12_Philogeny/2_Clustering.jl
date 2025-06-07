abstract type AbstractClusteringEstimator <: AbstractPhilogenyEstimator end
abstract type AbstractClusteringAlgorithm <: AbstractPhilogenyAlgorithm end
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
abstract type AbstractClusteringResult <: AbstractPhilogenyResult end
struct HierarchicalClusteringResult{T1 <: Clustering.Hclust, T2 <: AbstractMatrix,
                                    T3 <: AbstractMatrix, T4 <: Integer} <:
       AbstractClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
end
function HierarchicalClusteringResult(; clustering::Clustering.Hclust, S::AbstractMatrix,
                                      D::AbstractMatrix, k::Integer)
    @smart_assert(!isempty(S) && !isempty(D))
    @smart_assert(size(S) == size(D))
    @smart_assert(k >= one(k))
    return HierarchicalClusteringResult{typeof(clustering), typeof(S), typeof(D),
                                        typeof(k)}(clustering, S, D, k)
end
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
struct SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm end
struct PredefinedNumberClusters{T1 <: Integer} <: AbstractOptimalNumberClustersAlgorithm
    k::T1
end
function PredefinedNumberClusters(; k::Integer = 1)
    @smart_assert(k >= one(k))
    return PredefinedNumberClusters{typeof(k)}(k)
end
struct StandardisedSilhouetteScore{T1 <: Union{Nothing, <:Distances.SemiMetric}} <:
       AbstractOptimalNumberClustersAlgorithm
    metric::T1
end
function StandardisedSilhouetteScore(;
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing)
    return StandardisedSilhouetteScore{typeof(metric)}(metric)
end
struct OptimalNumberClusters{T1 <: Union{Nothing, <:Integer},
                             T2 <: AbstractOptimalNumberClustersAlgorithm} <:
       AbstractOptimalNumberClustersEstimator
    max_k::T1
    alg::T2
end
function OptimalNumberClusters(; max_k::Union{Nothing, <:Integer} = nothing,
                               alg::AbstractOptimalNumberClustersAlgorithm = SecondOrderDifference())
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return OptimalNumberClusters{typeof(max_k), typeof(alg)}(max_k, alg)
end
struct HierarchicalClustering{T1 <: Symbol} <: AbstractClusteringAlgorithm
    linkage::T1
end
function HierarchicalClustering(; linkage::Symbol = :ward)
    return HierarchicalClustering{typeof(linkage)}(linkage)
end
struct ClusteringEstimator{T1 <: StatsBase.CovarianceEstimator,
                           T2 <: AbstractDistanceEstimator,
                           T3 <: AbstractClusteringAlgorithm,
                           T4 <: Union{<:Integer, <:AbstractOptimalNumberClustersEstimator}} <:
       AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    onc::T4
end
function ClusteringEstimator(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             de::AbstractDistanceEstimator = Distance(;
                                                                      alg = CanonicalDistance()),
                             alg::AbstractClusteringAlgorithm = HierarchicalClustering(),
                             onc::Union{<:Integer,
                                        <:AbstractOptimalNumberClustersEstimator} = OptimalNumberClusters())
    return ClusteringEstimator{typeof(ce), typeof(de), typeof(alg), typeof(onc)}(ce, de,
                                                                                 alg, onc)
end

export clusterise, SecondOrderDifference, PredefinedNumberClusters,
       StandardisedSilhouetteScore, OptimalNumberClusters, HierarchicalClustering,
       ClusteringEstimator
