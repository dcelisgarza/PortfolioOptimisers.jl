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

# Constructor

    HierarchicalClustering(; clustering::Clustering.Hclust, S::AbstractMatrix,
                            D::AbstractMatrix, k::Integer)

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
"""
    HierarchicalClustering(; clustering::Clustering.Hclust, S::AbstractMatrix,
                            D::AbstractMatrix, k::Integer)

Construct a [`HierarchicalClustering`](@ref) result for hierarchical clustering.

Creates a `HierarchicalClustering` object from the given clustering result, similarity and distance matrices, and number of clusters. Validates that the matrices are non-empty, of matching size, and that `k ≥ 1`.

# Arguments

  - `clustering`: The hierarchical clustering object.
  - `S`: Similarity matrix.
  - `D`: Distance matrix.
  - `k`: Number of clusters.

# Returns

  - `HierarchicalClustering`: A result object encapsulating the clustering output.

# Validation

  - Throws an error if `S` or `D` are empty, if their sizes do not match, or if `k < 1`.

# Related

  - [`AbstractClusteringResult`](@ref)
"""
function HierarchicalClustering(; clustering::Clustering.Hclust, S::AbstractMatrix,
                                D::AbstractMatrix, k::Integer)
    @argcheck(!isempty(S) && !isempty(D) && size(S) == size(D) && k >= one(k),
              AssertionError("The following conditions must hold:\n`S` must be non-empty => $(!isempty(S))\n`D` must be non-empty => $(!isempty(D))\n`S` and `D` must have the same size => $(size(S) == size(D))\nk must be greater than or equal to 1 => $k"))
    return HierarchicalClustering(clustering, S, D, k)
end
"""
    clusterise(cle::AbstractClusteringResult, args...; kwargs...)

Return the clustering result as-is.

This function provides a generic interface for extracting or processing clustering results. By default, it simply returns the provided clustering result object unchanged. This allows for consistent downstream handling of clustering results in PortfolioOptimisers.jl workflows.

# Arguments

  - `cle::AbstractClusteringResult`: The clustering result object.
  - `args...; kwargs...`: Additional arguments (ignored by default).

# Returns

  - The input `cle` object.

# Related

  - [`AbstractClusteringResult`](@ref)
"""
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
"""
    SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm

Algorithm type for estimating the optimal number of clusters using the second-order difference method.

The `SecondOrderDifference` algorithm selects the optimal number of clusters by maximizing the second-order difference of a clustering evaluation metric (such as within-cluster sum of squares or silhouette score) across different cluster counts. This approach helps identify the "elbow" point in the metric curve.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
struct SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm end

"""
    PredefinedNumberClusters{T1} <: AbstractOptimalNumberClustersAlgorithm

Algorithm type for specifying a fixed, user-defined number of clusters.

`PredefinedNumberClusters` allows the user to set the number of clusters directly, bypassing any automatic selection algorithm. This is useful when the desired number of clusters is known in advance or dictated by external requirements.

# Fields

  - `k`: The fixed number of clusters.

# Constructor

    PredefinedNumberClusters(; k::Integer = 1)

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
struct PredefinedNumberClusters{T1} <: AbstractOptimalNumberClustersAlgorithm
    k::T1
end
"""
    PredefinedNumberClusters(; k::Integer = 1)

Construct a [`PredefinedNumberClusters`](@ref) algorithm with a fixed number of clusters.

# Arguments

  - `k``: The number of clusters (must be ≥ 1).

# Returns

  - `PredefinedNumberClusters`: An algorithm object specifying the fixed number of clusters.

# Validation

  - Throws an error if `k < 1`.

# Examples

```jldoctest
julia> PredefinedNumberClusters(; k = 3)
PredefinedNumberClusters
  k | Int64: 3
```

# Related

  - [`PredefinedNumberClusters`](@ref)
"""
function PredefinedNumberClusters(; k::Integer = 1)
    @argcheck(k >= one(k), DomainError("`k` must be greater than or equal to 1:\nk => $k"))
    return PredefinedNumberClusters(k)
end

"""
    StandardisedSilhouetteScore{T1} <: AbstractOptimalNumberClustersAlgorithm

Algorithm type for estimating the optimal number of clusters using the standardised silhouette score.

`StandardisedSilhouetteScore` selects the optimal number of clusters by maximizing the silhouette score, which measures how well each object lies within its cluster compared to other clusters. The score can be computed using different distance metrics.

# Fields

  - `metric`: The distance metric used for silhouette calculation (e.g., from `Distances.jl`), or `nothing` for the default.

# Constructor

    PredefinedNumberClusters(; k::Integer = 1)

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
struct StandardisedSilhouetteScore{T1} <: AbstractOptimalNumberClustersAlgorithm
    metric::T1
end
"""
    StandardisedSilhouetteScore(; metric::Union{Nothing, <:Distances.SemiMetric} = nothing)

Construct a [`StandardisedSilhouetteScore`](@ref) algorithm for optimal cluster number selection.

# Arguments

  - `metric`: The distance metric to use for silhouette calculation (optional). If `nothing`, the default metric is used.

# Returns

  - `StandardisedSilhouetteScore`: An algorithm object for silhouette-based cluster selection.

# Examples

```jldoctest
julia> StandardisedSilhouetteScore()
StandardisedSilhouetteScore
  metric | nothing
```

# Related

  - [`StandardisedSilhouetteScore`](@ref)
"""
function StandardisedSilhouetteScore(;
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing)
    return StandardisedSilhouetteScore(metric)
end

"""
    OptimalNumberClusters{T1, T2} <: AbstractOptimalNumberClustersEstimator

Estimator type for selecting the optimal number of clusters in PortfolioOptimisers.jl.

`OptimalNumberClusters` encapsulates the configuration for determining the optimal number of clusters, including the maximum allowed clusters and the algorithm used for selection.

# Fields

  - `max_k`: Maximum number of clusters to consider (can be `nothing` for no limit).
  - `alg`: Algorithm for selecting the optimal number of clusters (e.g., [`SecondOrderDifference`](@ref), [`StandardisedSilhouetteScore`](@ref), [`PredefinedNumberClusters`](@ref)).

# Constructor

    OptimalNumberClusters(; max_k::Union{Nothing, <:Integer} = nothing,
                          alg::AbstractOptimalNumberClustersAlgorithm = SecondOrderDifference())

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
struct OptimalNumberClusters{T1, T2} <: AbstractOptimalNumberClustersEstimator
    max_k::T1
    alg::T2
end
"""
    OptimalNumberClusters(; max_k::Union{Nothing, <:Integer} = nothing,
                          alg::AbstractOptimalNumberClustersAlgorithm = SecondOrderDifference())

Construct an [`OptimalNumberClusters`](@ref) estimator for optimal cluster number selection.

# Arguments

  - `max_k`: Maximum number of clusters to consider. If `nothing`, defaults to `ceil(Int, sqrt(N))`, where `N` is the number of assets.
  - `alg`: Algorithm for selecting the optimal number of clusters.

# Returns

  - `OptimalNumberClusters`: An estimator object for optimal cluster number selection.

# Validation

  - Throws an error if `max_k` is provided and less than 1.

# Examples

```jldoctest
julia> OptimalNumberClusters(; max_k = 10)
OptimalNumberClusters
  max_k | Int64: 10
    alg | SecondOrderDifference()
```

# Related

  - [`OptimalNumberClusters`](@ref)
"""
function OptimalNumberClusters(; max_k::Union{Nothing, <:Integer} = nothing,
                               alg::AbstractOptimalNumberClustersAlgorithm = SecondOrderDifference())
    if !isnothing(max_k)
        @argcheck(max_k >= one(max_k),
                  DomainError("`max_k` must be greater than or equal to 1:\nmax_k => $max_k"))
    end
    return OptimalNumberClusters(max_k, alg)
end

"""
    HClustAlgorithm{T1} <: AbstractClusteringAlgorithm

Algorithm type for hierarchical clustering in PortfolioOptimisers.jl.

`HClustAlgorithm` specifies the linkage method used for hierarchical clustering, such as `:ward`, `:single`, `:complete`, or `:average`.

# Fields

  - `linkage`: Linkage method for hierarchical clustering from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html).

# Constructor

    HClustAlgorithm(; linkage::Symbol = :ward)

# Related

  - [`AbstractClusteringAlgorithm`](@ref)
  - [`ClusteringEstimator`](@ref)
"""
struct HClustAlgorithm{T1} <: AbstractClusteringAlgorithm
    linkage::T1
end
"""
    HClustAlgorithm(; linkage::Symbol = :ward)

Construct an [`HClustAlgorithm`](@ref) for hierarchical clustering.

# Arguments

  - `linkage`: Linkage method to use for hierarchical clustering from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html).

# Returns

  - `HClustAlgorithm`: An algorithm object for hierarchical clustering.

# Examples

```jldoctest
julia> HClustAlgorithm(; linkage = :average)
HClustAlgorithm
  linkage | Symbol: :average
```

# Related

  - [`HClustAlgorithm`](@ref)
"""
function HClustAlgorithm(; linkage::Symbol = :ward)
    return HClustAlgorithm(linkage)
end

"""
    ClusteringEstimator{T1, T2, T3, T4} <: AbstractClusteringEstimator

Estimator type for clustering in PortfolioOptimisers.jl.

`ClusteringEstimator` encapsulates all configuration required for clustering, including the covariance estimator, distance estimator, clustering algorithm, and optimal number of clusters estimator.

# Fields

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Clustering algorithm.
  - `onc`: Optimal number of clusters estimator.

# Constructor

    ClusteringEstimator(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                         de::AbstractDistanceEstimator = Distance(alg = CanonicalDistance()),
                         alg::AbstractClusteringAlgorithm = HClustAlgorithm(),
                         onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())

# Related

  - [`AbstractClusteringEstimator`](@ref)
  - [`AbstractClusteringAlgorithm`](@ref)
  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
struct ClusteringEstimator{T1, T2, T3, T4} <: AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    onc::T4
end
"""
    ClusteringEstimator(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                         de::AbstractDistanceEstimator = Distance(alg = CanonicalDistance()),
                         alg::AbstractClusteringAlgorithm = HClustAlgorithm(),
                         onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())

Construct a [`ClusteringEstimator`](@ref) for clustering.

Creates a clustering estimator using the specified covariance estimator, distance estimator, clustering algorithm, and optimal number of clusters estimator.

# Arguments

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Clustering algorithm.
  - `onc`: Optimal number of clusters estimator.

# Returns

  - `ClusteringEstimator`: An estimator object for clustering.

# Examples

```jldoctest
julia> ClusteringEstimator()
ClusteringEstimator
   ce | PortfolioOptimisersCovariance
      |   ce | Covariance
      |      |    me | SimpleExpectedReturns
      |      |       |   w | nothing
      |      |    ce | GeneralWeightedCovariance
      |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      |      |       |    w | nothing
      |      |   alg | Full()
      |   mp | DefaultMatrixProcessing
      |      |       pdm | Posdef
      |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
      |      |   denoise | nothing
      |      |    detone | nothing
      |      |       alg | nothing
   de | Distance
      |   alg | CanonicalDistance()
  alg | HClustAlgorithm
      |   linkage | Symbol: :ward
  onc | OptimalNumberClusters
      |   max_k | nothing
      |     alg | SecondOrderDifference()
```

# Related

  - [`ClusteringEstimator`](@ref)
"""
function ClusteringEstimator(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             de::AbstractDistanceEstimator = Distance(;
                                                                      alg = CanonicalDistance()),
                             alg::AbstractClusteringAlgorithm = HClustAlgorithm(),
                             onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return ClusteringEstimator(ce, de, alg, onc)
end

export HierarchicalClustering, clusterise, SecondOrderDifference, PredefinedNumberClusters,
       StandardisedSilhouetteScore, OptimalNumberClusters, HClustAlgorithm,
       ClusteringEstimator
