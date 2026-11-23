"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimators that order the assets of a universe from their distances.

A seriation puts related assets next to each other. [`HierarchicalRiskParity`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) bisect the order that a seriation gives, so the order is an input of their weights. Their field `ser` holds a member of this family, or a seriation result. [`seriate`](@ref) runs a member and returns an [`AbstractSeriationResult`](@ref).

Each member holds the configuration that it reads. [`HierarchicalSeriation`](@ref) reads the leaf order of a dendrogram, and [`SpectralSeriation`](@ref) reads the spectrum of a Laplacian of the distances, with no tree.

# Interfaces

To add a seriation, subtype `AbstractSeriationEstimator` and implement this method:

## `seriate`

  - `seriate(ser::MySeriation, X::MatNum; dims::Int = 1, kwargs...) -> AbstractSeriationResult`: Order the assets of the data matrix `X`. The result holds the order in its property `order`, a permutation of the assets.

### Arguments

  - `ser`: The instance of the subtype.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - `dims`: The axis of the observations.
  - `kwargs...`: The keywords of the estimators that `ser` holds, such as `pr`, `rd` and `iv`.

### Returns

  - `res::AbstractSeriationResult`: The order of the assets.

A member that reads the returns states its source with a method of [`returns_source`](@ref), as [`SpectralSeriation`](@ref) does. Then the method of [`seriate`](@ref) over a prior result picks the returns for it.

# Related

  - [`AbstractSeriationResult`](@ref)
  - [`HierarchicalSeriation`](@ref)
  - [`SpectralSeriation`](@ref)
  - [`seriate`](@ref)
"""
abstract type AbstractSeriationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all results of a seriation.

A result holds the order of the assets in its property `order`, a permutation of the assets. [`HierarchicalRiskParity`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) read `order` and nothing else, so a subtype can hold any other diagnostic. A result in the field `ser` of an optimiser is used as it is, as a [`Clusters`](@ref) in the field `cle` of a clustering optimiser is.

# Interfaces

To add a result, subtype `AbstractSeriationResult` and give it the property `order`.

  - `order::AbstractVector{<:Integer}`: The assets in their order, a permutation of `1:N`.

A result that an optimiser can read over a subset of its assets, under an Investable Mask or inside a [`NestedClustered`](@ref), also implements `port_opt_view(res::MyResult, i, args...)`, which returns the order of the assets `i`. Without that method, the result is used as it is, and the optimiser refuses an order over another universe.

# Related

  - [`AbstractSeriationEstimator`](@ref)
  - [`HierarchicalSeriationResult`](@ref)
  - [`SpectralSeriationResult`](@ref)
  - [`FixedSeriationResult`](@ref)
  - [`seriate`](@ref)
"""
abstract type AbstractSeriationResult <: AbstractResult end
"""
    const SerE_SerR = Union{AbstractSeriationEstimator, AbstractSeriationResult}

Alias for a seriation estimator or result.

The field `ser` of [`HierarchicalRiskParity`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) takes either, as [`ClE_Cl`](@ref) does for a clustering.

# Related

  - [`AbstractSeriationEstimator`](@ref)
  - [`AbstractSeriationResult`](@ref)
"""
const SerE_SerR = Union{AbstractSeriationEstimator, AbstractSeriationResult}
"""
$(DocStringExtensions.TYPEDEF)

Orders the assets by the leaves of a dendrogram.

[`seriate`](@ref) clusters the assets with `cle`, and the order is the leaf order of the dendrogram. The branch order of the clustering algorithm decides the leaf order, see [`AbstractBranchOrder`](@ref). The default clustering gives the exact optimal leaf order of the Ward dendrogram.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalSeriation(;
        cle::HClE_HCl = ClustersEstimator()
    ) -> HierarchicalSeriation

Keywords correspond to the struct's fields.

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `cle`: Recursively indexed through [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> HierarchicalSeriation(;
                             cle = ClustersEstimator(;
                                                     alg = HClustAlgorithm(;
                                                                           branchorder = nothing)))
HierarchicalSeriation
  cle ┼ ClustersEstimator
      │      ce ┼ PortfolioOptimisersCovariance
      │         │   ce ┼ Covariance
      │         │      │    me ┼ SimpleExpectedReturns
      │         │      │       │   w ┴ nothing
      │         │      │    ce ┼ GeneralCovariance
      │         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │         │      │       │    w ┴ nothing
      │         │      │   alg ┼ FullMoment()
      │         │      │     w ┴ nothing
      │         │   mp ┼ MatrixProcessing
      │         │      │     pdm ┼ Posdef
      │         │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │         │      │      dn ┼ nothing
      │         │      │      dt ┼ nothing
      │         │      │     alg ┼ nothing
      │         │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │      de ┼ Distance
      │         │   power ┼ nothing
      │         │     alg ┴ CanonicalDistance()
      │     alg ┼ HClustAlgorithm
      │         │       linkage ┼ WardLinkage
      │         │               │   presquared ┴ Bool: false
      │         │   branchorder ┴ nothing
      │     onc ┼ OptimalNumberClusters
      │         │   max_k ┼ nothing
      │         │     alg ┼ SecondOrderDifference
      │         │         │   alg ┼ StandardisedValue
      │         │         │       │   mv ┼ MeanValue
      │         │         │       │      │   w ┴ nothing
      │         │         │       │   sv ┼ StdValue
      │         │         │       │      │           w ┼ nothing
      │         │         │       │      │   corrected ┴ Bool: true
      │   x_src ┴ nothing
```

# Related

  - [`AbstractSeriationEstimator`](@ref)
  - [`HierarchicalSeriationResult`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`AbstractBranchOrder`](@ref)
  - [`seriate`](@ref)
"""
@propagatable @concrete struct HierarchicalSeriation <: AbstractSeriationEstimator
    """
    The hierarchical clustering estimator, or a precomputed hierarchical clustering result. Its dendrogram gives the order.
    """
    @oprop cle
    function HierarchicalSeriation(cle::HClE_HCl)
        return new{typeof(cle)}(cle)
    end
end
function HierarchicalSeriation(; cle::HClE_HCl = ClustersEstimator())::HierarchicalSeriation
    return HierarchicalSeriation(cle)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the order of a [`HierarchicalSeriation`](@ref) and the clustering that gave it.

`order` is the vector `clr.res.order` itself, not a copy. A plot of the dendrogram reads `clr`, with its number of clusters `clr.k`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalSeriationResult(;
        clr::Clusters{<:Clustering.Hclust}
    ) -> HierarchicalSeriationResult

The constructor reads `order` from `clr`.

# Related

  - [`AbstractSeriationResult`](@ref)
  - [`HierarchicalSeriation`](@ref)
  - [`Clusters`](@ref)
"""
@concrete struct HierarchicalSeriationResult <: AbstractSeriationResult
    """
    $(field_dict[:clr])
    """
    clr
    """
    $(field_dict[:ser_order])
    """
    order
    function HierarchicalSeriationResult(clr::Clusters{<:Clustering.Hclust})
        order = clr.res.order
        return new{typeof(clr), typeof(order)}(clr, order)
    end
end
function HierarchicalSeriationResult(;
                                     clr::Clusters{<:Clustering.Hclust})::HierarchicalSeriationResult
    return HierarchicalSeriationResult(clr)
end
# The inner constructor reads `order` from the tree of `clr`, so a lens write leaves it out.
@forward_properties HierarchicalSeriationResult begin
    derived(order)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds an order of the assets that the caller states.

The order can come from outside the library, such as an order by sector, or the order of another implementation in a parity test. In the field `ser` of [`HierarchicalRiskParity`](@ref) or [`SchurComplementHierarchicalRiskParity`](@ref), it fixes the order of the bisection.

When the optimiser reads a subset of the assets, under an Investable Mask or inside a [`NestedClustered`](@ref), [`port_opt_view`](@ref) keeps the assets of the subset in their stated order.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FixedSeriationResult(;
        order::AbstractVector{<:Integer}
    ) -> FixedSeriationResult

Keywords correspond to the struct's fields.

## Validation

  - `isperm(order)`. The constructor throws an `ArgumentError` otherwise.

# Examples

```jldoctest
julia> FixedSeriationResult(; order = [3, 1, 2])
FixedSeriationResult
  order ┴ Vector{Int64}: [3, 1, 2]
```

# Related

  - [`AbstractSeriationResult`](@ref)
  - [`HierarchicalSeriationResult`](@ref)
  - [`SpectralSeriationResult`](@ref)
"""
@concrete struct FixedSeriationResult <: AbstractSeriationResult
    """
    $(field_dict[:ser_order])
    """
    order
    function FixedSeriationResult(order::AbstractVector{<:Integer})
        @argcheck(isperm(order),
                  ArgumentError("`FixedSeriationResult.order` is $order, which is not a permutation of `1:$(length(order))`. State each asset once."))
        return new{typeof(order)}(order)
    end
end
function FixedSeriationResult(; order::AbstractVector{<:Integer})::FixedSeriationResult
    return FixedSeriationResult(order)
end
"""
    seriate(ser::HierarchicalSeriation, X::MatNum; dims::Int = 1, kwargs...)
    seriate(ser::AbstractSeriationResult, args...; kwargs...)

Order the assets with a seriation estimator, or return a seriation result unchanged.

# Algorithm

 1. A result: return `ser` itself. The method runs no seriation, and it discards `args` and `kwargs`.
 2. A [`HierarchicalSeriation`](@ref): cluster `X` with `ser.cle` through [`clusterise`](@ref), and return the [`HierarchicalSeriationResult`](@ref) of the clustering.

# Arguments

  - `ser`: The seriation estimator or result.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for [`clusterise`](@ref).

# Returns

  - `res::AbstractSeriationResult`: The order of the assets.

# Related

  - [`AbstractSeriationEstimator`](@ref)
  - [`AbstractSeriationResult`](@ref)
  - [`clusterise`](@ref)
"""
function seriate(ser::HierarchicalSeriation, X::MatNum; dims::Int = 1, kwargs...)
    return HierarchicalSeriationResult(clusterise(ser.cle, X; dims = dims, kwargs...))
end
function seriate(ser::AbstractSeriationResult, args...; kwargs...)
    return ser
end
function returns_source(ser::HierarchicalSeriation)
    return returns_source(ser.cle)
end
function reader_frames(ser::HierarchicalSeriation, path::AbstractString, refit::Bool)
    return field_frames(ser, path, refit)
end
"""
    assert_seriation_universe(ser::AbstractSeriationResult, N::Integer)::Nothing

Assert that a seriation orders exactly the assets that the optimisation allocates over.

A result that the caller states is used as it is, as [`assert_clustering_universe`](@ref) states for a clustering. Under an Investable Mask, or inside a subset or a nested view, the optimisation runs on fewer assets than the caller ordered. The order would then index the wrong columns. The order of a [`HierarchicalSeriationResult`](@ref) is the leaf order of its clustering, so it has one entry for each asset of the clustering.

# Arguments

  - `ser`: The seriation result.
  - `N`: Number of assets the optimisation runs on, after the reduction.

# Validation

  - `length(ser.order) == N`, else a `DimensionMismatch` is raised.

# Returns

  - `nothing`.

# Related

  - [`assert_clustering_universe`](@ref)
  - [`AbstractSeriationResult`](@ref)
"""
function assert_seriation_universe(ser::AbstractSeriationResult, N::Integer)::Nothing
    Ns = length(ser.order)
    @argcheck(Ns == N,
              DimensionMismatch("the seriation orders $(Ns) assets, but the optimisation runs on $(N). A seriation result is used exactly as stated, so it must be stated over the universe the optimisation runs on. State a seriation estimator instead when the universe changes."))
    return nothing
end
"""
    port_opt_view(ser::FixedSeriationResult, i, args...) -> FixedSeriationResult

Keep the assets `i` of a stated order, in the order that it states.

The assets of `i` are numbered by their position in `i`, so the view is a permutation of `1:length(i)`.

# Arguments

  - `ser`: The stated order.
  - `i`: The indices of the assets that the view keeps.
  - `args...`: Not read.

# Returns

  - `ser::FixedSeriationResult`: The order of the assets `i`.

# Related

  - [`FixedSeriationResult`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(ser::FixedSeriationResult, i, args...)
    return FixedSeriationResult(seriation_view_order(ser.order, i, length(ser.order)))
end
"""
    seriation_view_order(order::AbstractVector{<:Integer}, i, N::Integer) -> Vector{Int}

The order of the assets `i` in an order `order` of some assets of a universe of `N` assets, with each asset numbered by its position in `i`.

An asset of `i` that `order` does not hold is not in the view. A stated order holds every asset, so its view is a permutation of `1:length(i)`. The memory of a spectral fit holds the assets that the fit ordered alone, so its view can hold fewer.

# Arguments

  - `order`: Distinct assets of the universe, by their index in the universe.
  - `i`: The indices of the assets that the view keeps.
  - `N`: The number of assets of the universe.

# Returns

  - `o::Vector{Int}`: The assets of `i` that `order` holds, in its order, numbered by their position in `i`.

# Related

  - [`port_opt_view`](@ref)
  - [`FixedSeriationResult`](@ref)
"""
function seriation_view_order(order::AbstractVector{<:Integer}, i, N::Integer)
    pos = zeros(Int, N)
    pos[i] .= 1:length(i)
    return [pos[k] for k in order if !iszero(pos[k])]
end

export HierarchicalSeriation, HierarchicalSeriationResult, FixedSeriationResult, seriate
# The abstract types of two Open Families. The verb of their interface is `seriate`.
public AbstractSeriationEstimator, AbstractSeriationResult
