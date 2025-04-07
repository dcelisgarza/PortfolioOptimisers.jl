abstract type AbstractClusteringEstimator <: AbstractPhilogenyEstimator end
abstract type AbstractClusteringAlgorithm <: AbstractPhilogenyAlgorithm end
abstract type AbstractOptimalNumberClusters <: AbstractPhilogenyAlgorithm end
abstract type AbstractClusteringResult <: AbstractPhilogenyResult end

function clusterise end
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
struct HierarchicalClustering{T1 <: Symbol} <: AbstractClusteringAlgorithm
    linkage::T1
end
function HierarchicalClustering(; linkage::Symbol = :ward)
    return HierarchicalClustering{typeof(linkage)}(linkage)
end

struct PredefinedNumberClusters{T1 <: Integer, T2 <: Union{Nothing, <:Integer}} <:
       AbstractOptimalNumberClusters
    k::T1
    max_k::T2
end
function PredefinedNumberClusters(; k::Integer = 1,
                                  max_k::Union{Nothing, <:Integer} = nothing)
    @smart_assert(k >= one(k))
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return PredefinedNumberClusters{typeof(k), typeof(max_k)}(k, max_k)
end
struct SecondOrderDifference{T1 <: Union{Nothing, <:Integer}} <:
       AbstractOptimalNumberClusters
    max_k::T1
end
function SecondOrderDifference(; max_k::Union{Nothing, <:Integer} = nothing)
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return SecondOrderDifference{typeof(max_k)}(max_k)
end
struct StandardisedSilhouetteScore{T1 <: Union{Nothing, <:Integer},
                                   T2 <: Union{Nothing, <:Distances.SemiMetric}} <:
       AbstractOptimalNumberClusters
    max_k::T1
    metric::T2
end
function StandardisedSilhouetteScore(; max_k::Union{Nothing, <:Integer} = nothing,
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing)
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return StandardisedSilhouetteScore{typeof(max_k), typeof(metric)}(max_k, metric)
end
struct ClusteringEstimator{T1 <: StatsBase.CovarianceEstimator,
                           T2 <: AbstractDistanceEstimator,
                           T3 <: AbstractClusteringAlgorithm,
                           T4 <: Union{<:Integer, AbstractOptimalNumberClusters}} <:
       AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    nch::T4
end
function ClusteringEstimator(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             de::AbstractDistanceEstimator = CanonicalDistance(),
                             alg::AbstractClusteringAlgorithm = HierarchicalClustering(),
                             nch::Union{<:Integer, AbstractOptimalNumberClusters} = SecondOrderDifference())
    return ClusteringEstimator{typeof(ce), typeof(de), typeof(alg), typeof(nch)}(ce, de,
                                                                                 alg, nch)
end
