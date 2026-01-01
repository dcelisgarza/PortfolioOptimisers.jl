"""
    abstract type AbstractClusteringEstimator <: AbstractPhylogenyEstimator end

Abstract supertype for all clustering estimator types in PortfolioOptimisers.jl.

All concrete types implementing clustering-based estimation algorithms should subtype `AbstractClusteringEstimator`. This enables a consistent interface for clustering estimators throughout the package.

# Related

  - [`AbstractClusteringAlgorithm`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClusteringEstimator <: AbstractPhylogenyEstimator end
abstract type AbstractHierarchicalClusteringEstimator <: AbstractClusteringEstimator end
abstract type AbstractNonHierarchicalClusteringEstimator <: AbstractClusteringEstimator end
"""
    abstract type AbstractClusteringAlgorithm <: AbstractPhylogenyAlgorithm end

Abstract supertype for all clustering algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific clustering algorithms should subtype `AbstractClusteringAlgorithm`. This enables flexible extension and dispatch of clustering routines.

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClusteringAlgorithm <: AbstractPhylogenyAlgorithm end
"""
"""
abstract type AbstractHierarchicalClusteringAlgorithm <: AbstractClusteringAlgorithm end
"""
"""
abstract type AbstractNonHierarchicalClusteringAlgorithm <: AbstractClusteringAlgorithm end
"""
    abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end

Abstract supertype for all optimal number of clusters estimator types in PortfolioOptimisers.jl.

All concrete types implementing algorithms to estimate the optimal number of clusters should subtype `AbstractOptimalNumberClustersEstimator`. This enables a consistent interface for cluster number estimation.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
"""
    abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end

Abstract supertype for all optimal number of clusters algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific algorithms for determining the optimal number of clusters should subtype `AbstractOptimalNumberClustersAlgorithm`. This enables flexible extension and dispatch of cluster number selection routines.

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}
"""
    abstract type AbstractClusteringResult <: AbstractPhylogenyResult end

Abstract supertype for all clustering result types in PortfolioOptimisers.jl.

All concrete types representing the result of a clustering estimation should subtype `AbstractClusteringResult`. This enables a consistent interface for clustering results throughout the package.

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractClusteringAlgorithm`](@ref)
"""
abstract type AbstractClusteringResult <: AbstractPhylogenyResult end
"""
"""
abstract type AbstractHierarchicalClusteringResult <: AbstractClusteringResult end
"""
"""
abstract type AbstractNonHierarchicalClusteringResult <: AbstractClusteringResult end
const HClE_HCl = Union{<:AbstractHierarchicalClusteringEstimator,
                       <:AbstractHierarchicalClusteringResult}
const ClE_Cl = Union{<:AbstractClusteringEstimator, <:AbstractClusteringResult}
"""
    struct HierarchicalClustering{T1, T2, T3, T4} <: AbstractHierarchicalClusteringResult
        clustering::T1
        S::T2
        D::T3
        k::T4
    end

Result type for hierarchical clustering in PortfolioOptimisers.jl.

`HierarchicalClustering` stores the output of a hierarchical clustering algorithm, including the clustering object, similarity and distance matrices, and the number of clusters.

# Fields

  - `clustering`: The hierarchical clustering object.
  - `S`: Similarity matrix used for clustering.
  - `D`: Distance matrix used for clustering.
  - `k`: Number of clusters.

# Constructor

    HierarchicalClustering(; clustering::Clustering.Hclust, S::MatNum,
                           D::MatNum, k::Integer)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(S)`.
  - `!isempty(D)`.
  - `size(S) == size(D)`.
  - `k ≥ 1`.

# Related

  - [`AbstractHierarchicalClusteringResult`](@ref)
  - [`HierarchicalClusteringEstimator`](@ref)
"""
struct HierarchicalClustering{T1, T2, T3, T4} <: AbstractHierarchicalClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
    function HierarchicalClustering(clustering::Clustering.Hclust, S::MatNum, D::MatNum,
                                    k::Integer)
        @argcheck(!isempty(S), IsEmptyError)
        @argcheck(!isempty(D), IsEmptyError)
        @argcheck(size(S) == size(D), DimensionMismatch)
        @argcheck(one(k) <= k, DomainError)
        return new{typeof(clustering), typeof(S), typeof(D), typeof(k)}(clustering, S, D, k)
    end
end
function HierarchicalClustering(; clustering::Clustering.Hclust, S::MatNum, D::MatNum,
                                k::Integer)
    return HierarchicalClustering(clustering, S, D, k)
end
"""
    clusterise(cle::AbstractClusteringResult, args...; kwargs...)

Return the clustering result as-is.

This function provides a generic interface for extracting or processing clustering results. By default, it simply returns the provided clustering result object unchanged. This allows for consistent downstream handling of clustering results in PortfolioOptimisers.jl workflows.

# Arguments

  - `cle::AbstractClusteringResult`: The clustering result object.
  - `args...`: Additional positional arguments, ignored.
  - `kwargs...`: Additional keyword arguments, ignored.

# Returns

  - The input `cle` object.

# Related

  - [`AbstractClusteringResult`](@ref)
"""
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
"""
    struct SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm end

Algorithm type for estimating the optimal number of clusters using the second-order difference method.

The `SecondOrderDifference` algorithm selects the optimal number of clusters by maximizing the second-order difference of a clustering evaluation metric (such as within-cluster sum of squares or silhouette score) across different cluster counts. This approach helps identify the "elbow" point in the metric curve.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
struct SecondOrderDifference{T1} <: AbstractOptimalNumberClustersAlgorithm
    alg::T1
    function SecondOrderDifference(alg::VectorToScalarMeasure)
        return new{typeof(alg)}(alg)
    end
end
function SecondOrderDifference(; alg::VectorToScalarMeasure = StandardisedValue())
    return SecondOrderDifference(alg)
end
"""
    struct SilhouetteScore{T1} <: AbstractOptimalNumberClustersAlgorithm
        metric::T1
    end

Algorithm type for estimating the optimal number of clusters using the standardised silhouette score.

`SilhouetteScore` selects the optimal number of clusters by maximizing the silhouette score, which measures how well each object lies within its cluster compared to other clusters. The score can be computed using different distance metrics.

# Fields

  - `metric`: The distance metric used for silhouette calculation from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl), or `nothing` for the default.

# Constructor

    SilhouetteScore(; metric::Option{<:Distances.SemiMetric} = nothing)

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> SilhouetteScore()
SilhouetteScore
  metric ┴ nothing
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)
"""
struct SilhouetteScore{T1, T2} <: AbstractOptimalNumberClustersAlgorithm
    alg::T1
    metric::T2
    function SilhouetteScore(alg::VectorToScalarMeasure,
                             metric::Option{<:Distances.SemiMetric})
        return new{typeof(alg), typeof(metric)}(alg, metric)
    end
end
function SilhouetteScore(; alg::VectorToScalarMeasure = StandardisedValue(),
                         metric::Option{<:Distances.SemiMetric} = nothing)
    return SilhouetteScore(alg, metric)
end
"""
    struct OptimalNumberClusters{T1, T2} <: AbstractOptimalNumberClustersEstimator
        max_k::T1
        alg::T2
    end

Estimator type for selecting the optimal number of clusters in PortfolioOptimisers.jl.

`OptimalNumberClusters` encapsulates the configuration for determining the optimal number of clusters, including the maximum allowed clusters and the algorithm used for selection.

# Fields

  - `max_k`: Maximum number of clusters to consider. If `nothing`, computed as the `sqrt(N)`, where `N` is the number of assets.
  - `alg`: Algorithm for selecting the optimal number of clusters. If an integer, defines the number of clusters directly.

# Constructor

    OptimalNumberClusters(; max_k::Option{<:Integer} = nothing,
                          alg::Int_ONC = SecondOrderDifference())

Keyword arguments correspond to the fields above.

## Validation

  - `max_k >= 1`.
  - If `alg` is an integer, `alg >= 1`.

# Examples

```jldoctest
julia> OptimalNumberClusters(; max_k = 10)
OptimalNumberClusters
  max_k ┼ Int64: 10
    alg ┴ SecondOrderDifference()
```

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
struct OptimalNumberClusters{T1, T2} <: AbstractOptimalNumberClustersEstimator
    max_k::T1
    alg::T2
    function OptimalNumberClusters(max_k::Option{<:Integer}, alg::Int_ONC)
        if !isnothing(max_k)
            @argcheck(one(max_k) <= max_k, DomainError)
        end
        if isa(alg, Integer)
            @argcheck(one(alg) <= alg, DomainError)
        end
        return new{typeof(max_k), typeof(alg)}(max_k, alg)
    end
end
function OptimalNumberClusters(; max_k::Option{<:Integer} = nothing,
                               alg::Int_ONC = SecondOrderDifference())
    return OptimalNumberClusters(max_k, alg)
end
"""
    struct HClustAlgorithm{T1} <: AbstractHierarchicalClusteringAlgorithm
        linkage::T1
    end

Algorithm type for hierarchical clustering in PortfolioOptimisers.jl.

`HClustAlgorithm` specifies the linkage method used for hierarchical clustering, such as `:ward`, `:single`, `:complete`, or `:average`.

# Fields

  - `linkage`: Linkage method for hierarchical clustering from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html).

# Constructor

    HClustAlgorithm(; linkage::Symbol = :ward)

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> HClustAlgorithm(; linkage = :average)
HClustAlgorithm
  linkage ┴ Symbol: :average
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`HierarchicalClusteringEstimator`](@ref)
"""
struct HClustAlgorithm{T1} <: AbstractHierarchicalClusteringAlgorithm
    linkage::T1
    function HClustAlgorithm(linkage::Symbol)
        return new{typeof(linkage)}(linkage)
    end
end
function HClustAlgorithm(; linkage::Symbol = :ward)
    return HClustAlgorithm(linkage)
end
"""
    struct HierarchicalClusteringEstimator{T1, T2, T3, T4} <: AbstractClusteringEstimator
        ce::T1
        de::T2
        alg::T3
        onc::T4
    end

Estimator type for clustering in PortfolioOptimisers.jl.

`HierarchicalClusteringEstimator` encapsulates all configuration required for clustering, including the covariance estimator, distance estimator, clustering algorithm, and optimal number of clusters estimator.

# Fields

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Clustering algorithm.
  - `onc`: Optimal number of clusters estimator.

# Constructor

    HierarchicalClusteringEstimator(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
                        alg::AbstractHierarchicalClusteringAlgorithm = HClustAlgorithm(),
                        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> HierarchicalClusteringEstimator()
HierarchicalClusteringEstimator
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┴ Full()
      │   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │      │       pdm ┼ Posdef
      │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │   denoise ┼ nothing
      │      │    detone ┼ nothing
      │      │       alg ┼ nothing
      │      │     order ┴ DenoiseDetoneAlg()
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  alg ┼ HClustAlgorithm
      │   linkage ┴ Symbol: :ward
  onc ┼ OptimalNumberClusters
      │   max_k ┼ nothing
      │     alg ┴ SecondOrderDifference()
```

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
struct HierarchicalClusteringEstimator{T1, T2, T3, T4} <:
       AbstractHierarchicalClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    onc::T4
    function HierarchicalClusteringEstimator(ce::StatsBase.CovarianceEstimator,
                                             de::AbstractDistanceEstimator,
                                             alg::AbstractHierarchicalClusteringAlgorithm,
                                             onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(onc)}(ce, de, alg, onc)
    end
end
function HierarchicalClusteringEstimator(;
                                         ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                         de::AbstractDistanceEstimator = Distance(;
                                                                                  alg = CanonicalDistance()),
                                         alg::AbstractHierarchicalClusteringAlgorithm = HClustAlgorithm(),
                                         onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return HierarchicalClusteringEstimator(ce, de, alg, onc)
end

export HierarchicalClustering, clusterise, SecondOrderDifference, SilhouetteScore,
       OptimalNumberClusters, HClustAlgorithm, HierarchicalClusteringEstimator
