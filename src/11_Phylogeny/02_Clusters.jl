"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing clustering-based estimation algorithms should be subtypes of `AbstractClustersEstimator`.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersEstimator <: AbstractPhylogenyEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific clustering algorithms should be subtypes of `AbstractClustersAlgorithm`.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersAlgorithm <: AbstractPhylogenyAlgorithm end
function factory(alg::AbstractClustersAlgorithm, args...; kwargs...)
    return alg
end
abstract type AbstractHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
abstract type AbstractNonHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing algorithms to estimate the optimal number of clusters should be subtypes of `AbstractOptimalNumberClustersEstimator`.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific algorithms for determining the optimal number of clusters should be subtypes of `AbstractOptimalNumberClustersAlgorithm`.

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering result types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing the result of a clustering estimation should be subtypes of `AbstractClusteringResult`.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClustersAlgorithm`](@ref)
"""
abstract type AbstractClusteringResult <: AbstractPhylogenyResult end
const ClTypes = Union{<:Clustering.ClusteringResult, <:Clustering.Hclust}
"""
$(DocStringExtensions.TYPEDEF)

Result type for hierarchical clustering in `PortfolioOptimisers.jl`.

`Clusters` stores the output of a hierarchical clustering algorithm, including the clustering object, similarity and distance matrices, and the number of clusters.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Clusters(;
        res::ClTypes,
        S::MatNum,
        D::MatNum,
        k::Integer
    ) -> Clusters

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:S])
  - $(val_dict[:D])
  - $(val_dict[:S_D])
  - $(val_dict[:ck])

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`ClustersEstimator`](@ref)
"""
@concrete struct Clusters <: AbstractClusteringResult
    "$(field_dict[:cres])"
    res
    "$(field_dict[:S])"
    S
    "$(field_dict[:D])"
    D
    "$(field_dict[:ck])"
    k
    function Clusters(res::ClTypes, S::MatNum, D::MatNum, k::Integer)
        @argcheck(!isempty(S), IsEmptyError)
        @argcheck(!isempty(D), IsEmptyError)
        @argcheck(size(S) == size(D), DimensionMismatch)
        @argcheck(one(k) <= k, DomainError)
        return new{typeof(res), typeof(S), typeof(D), typeof(k)}(res, S, D, k)
    end
end
function Clusters(; res::ClTypes, S::MatNum, D::MatNum, k::Integer)
    return Clusters(res, S, D, k)
end
"""
    clusterise(cle::AbstractClusteringResult, args...; kwargs...)

Return the clustering result as-is.

This function provides a generic interface for extracting or processing clustering results. By default, it simply returns the provided clustering result object unchanged. This allows for consistent downstream handling of clustering results in `PortfolioOptimisers.jl` workflows.

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
$(DocStringExtensions.TYPEDEF)

Algorithm type for estimating the optimal number of clusters using the second-order difference method.

The `SecondOrderDifference` algorithm selects the optimal number of clusters by maximizing the second-order difference of a clustering evaluation metric (such as within-cluster sum of squares or silhouette score) across different cluster counts. This approach helps identify the "elbow" point in the metric curve.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SecondOrderDifference(;
        alg::VectorToScalarMeasure = StandardisedValue()
    ) -> SecondOrderDifference

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> SecondOrderDifference()
SecondOrderDifference
  alg ┼ StandardisedValue
      │   mv ┼ MeanValue
      │      │   w ┴ nothing
      │   sv ┼ StdValue
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`VectorToScalarMeasure`](@ref)
"""
@concrete struct SecondOrderDifference <: AbstractOptimalNumberClustersAlgorithm
    "$(field_dict[:vsalg])"
    alg
    function SecondOrderDifference(alg::VectorToScalarMeasure)
        return new{typeof(alg)}(alg)
    end
end
function SecondOrderDifference(; alg::VectorToScalarMeasure = StandardisedValue())
    return SecondOrderDifference(alg)
end
function factory(alg::SecondOrderDifference, w::StatsBase.AbstractWeights)
    return SecondOrderDifference(; alg = factory(alg.alg, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for estimating the optimal number of clusters using the standardised silhouette score.

`SilhouetteScore` selects the optimal number of clusters by maximizing the silhouette score, which measures how well each object lies within its cluster compared to other clusters. The score can be computed using different distance metrics.

# Fields

$(DocStringExtensions.FIELDS)

  - `alg`: The vector-to-scalar measure used to evaluate clustering quality.
  - `metric`: The distance metric used for silhouette calculation from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl), or `nothing` for the default.

# Constructors

    SilhouetteScore(;
        alg::VectorToScalarMeasure = StandardisedValue(),
        metric::Option{<:Distances.SemiMetric} = nothing
    ) -> SilhouetteScore

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> SilhouetteScore()
SilhouetteScore
     alg ┼ StandardisedValue
         │   mv ┼ MeanValue
         │      │   w ┴ nothing
         │   sv ┼ StdValue
         │      │           w ┼ nothing
         │      │   corrected ┴ Bool: true
  metric ┴ nothing
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)
"""
@concrete struct SilhouetteScore <: AbstractOptimalNumberClustersAlgorithm
    "$(field_dict[:vsalg])"
    alg
    metric
    function SilhouetteScore(alg::VectorToScalarMeasure,
                             metric::Option{<:Distances.SemiMetric})
        return new{typeof(alg), typeof(metric)}(alg, metric)
    end
end
function SilhouetteScore(; alg::VectorToScalarMeasure = StandardisedValue(),
                         metric::Option{<:Distances.SemiMetric} = nothing)
    return SilhouetteScore(alg, metric)
end
function factory(alg::SilhouetteScore, w::StatsBase.AbstractWeights)
    return SilhouetteScore(; alg = factory(alg.alg, w), metric = alg.metric)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for selecting the optimal number of clusters in `PortfolioOptimisers.jl`.

`OptimalNumberClusters` encapsulates the configuration for determining the optimal number of clusters, including the maximum allowed clusters and the algorithm used for selection.

# Fields

  - `max_k`: Maximum number of clusters to consider. If `nothing`, computed as the `sqrt(N)`, where `N` is the number of assets.
  - `alg`: Algorithm for selecting the optimal number of clusters. If an integer, defines the number of clusters directly.

# Constructors

    OptimalNumberClusters(;
        max_k::Option{<:Integer} = nothing,
        alg::Int_ONC = SecondOrderDifference()
    ) -> OptimalNumberClusters

Keywords correspond to the struct's fields.

## Validation

  - `max_k >= 1`.
  - If `alg` is an integer, `alg >= 1`.

# Examples

```jldoctest
julia> OptimalNumberClusters(; max_k = 10)
OptimalNumberClusters
  max_k ┼ Int64: 10
    alg ┼ SecondOrderDifference
        │   alg ┼ StandardisedValue
        │       │   mv ┼ MeanValue
        │       │      │   w ┴ nothing
        │       │   sv ┼ StdValue
        │       │      │           w ┼ nothing
        │       │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
@concrete struct OptimalNumberClusters <: AbstractOptimalNumberClustersEstimator
    max_k
    alg
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
function factory(onc::OptimalNumberClusters, w::StatsBase.AbstractWeights)
    return OptimalNumberClusters(; max_k = onc.max_k, alg = factory(onc.alg, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for hierarchical clustering in `PortfolioOptimisers.jl`.

`HClustAlgorithm` specifies the linkage method used for hierarchical clustering, such as `:ward`, `:single`, `:complete`, or `:average`.

# Fields

  - `linkage`: Linkage method for hierarchical clustering from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html).

# Constructors

    HClustAlgorithm(;
        linkage::Symbol = :ward
    ) -> HClustAlgorithm

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> HClustAlgorithm(; linkage = :average)
HClustAlgorithm
  linkage ┴ Symbol: :average
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`]-(@ref)
  - [`ClustersEstimator`](@ref)
"""
@concrete struct HClustAlgorithm <: AbstractHierarchicalClusteringAlgorithm
    linkage
    function HClustAlgorithm(linkage::Symbol)
        return new{typeof(linkage)}(linkage)
    end
end
function HClustAlgorithm(; linkage::Symbol = :ward)
    return HClustAlgorithm(linkage)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for clustering in `PortfolioOptimisers.jl`.

`ClustersEstimator` encapsulates all configuration required for clustering, including the covariance estimator, distance estimator, res algorithm, and optimal number of clusters estimator.

# Fields

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Clustering algorithm.
  - `onc`: Optimal number of clusters estimator.

# Constructors

    ClustersEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::AbstractClustersAlgorithm = HClustAlgorithm(),
        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters()
    ) -> ClustersEstimator

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> ClustersEstimator()
ClustersEstimator
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │     w ┼ nothing
      │      │       │   idx ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │    ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │     w ┼ nothing
      │      │       │   idx ┴ nothing
      │      │   alg ┴ Full()
      │   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ DenoiseDetoneAlg()
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  alg ┼ HClustAlgorithm
      │   linkage ┴ Symbol: :ward
  onc ┼ OptimalNumberClusters
      │   max_k ┼ nothing
      │     alg ┼ SecondOrderDifference
      │         │   alg ┼ StandardisedValue
      │         │       │   mv ┼ MeanValue
      │         │       │      │   w ┴ nothing
      │         │       │   sv ┼ StdValue
      │         │       │      │           w ┼ nothing
      │         │       │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractHierarchicalClusteringAlgorithm`]-(@ref)
  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
@concrete struct ClustersEstimator <: AbstractClustersEstimator
    ce
    de
    alg
    onc
    function ClustersEstimator(ce::StatsBase.CovarianceEstimator,
                               de::AbstractDistanceEstimator,
                               alg::AbstractClustersAlgorithm,
                               onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(onc)}(ce, de, alg, onc)
    end
end
function ClustersEstimator(;
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           de::AbstractDistanceEstimator = Distance(;
                                                                    alg = CanonicalDistance()),
                           alg::AbstractClustersAlgorithm = HClustAlgorithm(),
                           onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return ClustersEstimator(ce, de, alg, onc)
end
function factory(cle::ClustersEstimator, w::StatsBase.AbstractWeights)
    return ClustersEstimator(; ce = factory(cle.ce, w), de = cle.de, alg = cle.alg,
                             onc = cle.onc)
end
const HClE_HCl = Union{<:ClustersEstimator{<:Any, <:Any,
                                           <:AbstractHierarchicalClusteringAlgorithm,
                                           <:Any},
                       <:Clusters{<:Clustering.Hclust, <:Any, <:Any, <:Any}}
const ClE_Cl = Union{<:AbstractClustersEstimator, <:AbstractClusteringResult}

export Clusters, clusterise, SecondOrderDifference, SilhouetteScore, OptimalNumberClusters,
       HClustAlgorithm, ClustersEstimator
