"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all network estimator types.

All concrete and/or abstract types implementing network-based estimation algorithms should be subtypes of `AbstractNetworkEstimator`.

# Related

  - [`NetworkEstimator`](@ref)
  - [`AbstractCentralityEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.
"""
abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end
"""
    const NwE_Pl_ClE_Cl = Union{<:AbstractNetworkEstimator, <:PhylogenyResult, <:ClE_Cl}

Alias for a network estimator, phylogeny result, or clustering estimator/result.

Used internally for dispatch in phylogeny and network estimation workflows that accept any of these forms.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`ClE_Cl`](@ref)
"""
const NwE_Pl_ClE_Cl = Union{<:AbstractNetworkEstimator, <:PhylogenyResult, <:ClE_Cl}
"""
    const NwE_ClE = Union{<:AbstractNetworkEstimator, <:AbstractClustersEstimator}

Alias for a phylogeny **source**: a network estimator or a clustering estimator, and nothing precomputed.

This is the bound of the `pl` slot on [`SemiDefinitePhylogenyEstimator`](@ref) and [`IntegerPhylogenyEstimator`](@ref), and the exclusion is the point. A constraint *estimator* answers "how do I build this constraint for whatever universe I am given"; a precomputed `PhylogenyResult` or `Clusters` in that slot answers a different question — "here is the answer for the universe I was built on" — and the two are only interchangeable while the universe never changes.

They stopped being interchangeable the moment a meta-optimiser handed a subproblem a subset of the assets. `phylogeny_matrix` returns a precomputed result unchanged, so the estimator emitted a full-universe constraint matrix for a three-asset subproblem, and every guard aimed at precomputed constraints missed it because the object presented as an estimator. The exclusion therefore lives in the **type**: the shape is not constructible, so there is no runtime check to write, to forget, or to route around. The only runtime guard left on this path is [`assert_external_optimiser`](@ref), which now has just one remaining case to catch — a precomputed constraint *result*.

Precomputed structure has a home already: build the constraint once and pass the **result** — [`SemiDefinitePhylogeny`](@ref) or [`IntegerPhylogeny`](@ref), whose `A` field takes a `PhylogenyResult` or a bare matrix — which is exactly what `phylogeny_constraints(est, X)` returns. Nothing is lost, and the guards that exist for results then apply.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractClustersEstimator`](@ref)
  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
"""
const NwE_ClE = Union{<:AbstractNetworkEstimator, <:AbstractClustersEstimator}
"""
    const NwE_ClE_Cl = Union{<:AbstractNetworkEstimator, <:ClE_Cl}

Alias for a network estimator or clustering estimator/result.

Used for dispatch in phylogeny workflows that accept either a network estimator or a clustering estimator/result.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`ClE_Cl`](@ref)
"""
const NwE_ClE_Cl = Union{<:AbstractNetworkEstimator, <:ClE_Cl}
"""
$(DocStringExtensions.TYPEDEF)

Builds an asset network from a covariance estimate, and says which pairs of assets it relates.

`NetworkEstimator` encapsulates the configuration for constructing a network from asset data, including the covariance estimator, distance estimator, tree or similarity algorithm, and the separation algorithm that says how far apart two assets sit in the resulting graph.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Tree_SimMat = KruskalTree(),
        sep::AbstractSeparationAlgorithm = HopCount()
    ) -> NetworkEstimator

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `de`: Recursively updated via [`factory`](@ref).

# The separation lives here, not on the consumer

`sep` says which pairs the network relates, and every consumer that reads a *closure* of this graph needs that answer: [`phylogeny_matrix`](@ref) and the phylogeny constraint families, both [`clusterise`](@ref) methods, and [`Proximity`](@ref). It therefore sits on the estimator that builds the graph rather than on any one of them — a rule visible only to the feature producer would be structurally invisible to the constraint path, which receives nothing but this estimator.

The one exception is a consumer that reads the **structure** rather than a closure of it, and `sep` is **inert** there: the weighted routes of [`centrality_graph`](@ref) take the weighted graph itself, because a closure is a sum of matrix powers and a power of a weighted matrix sums *products* of distances. So a [`HopCount`](@ref) of `n = 2` moves a [`DegreeCentrality`](@ref) and leaves a [`ClosenessCentrality`](@ref) where it was. At the default `HopCount(; n = 1)` nothing is visible, since the closure of a graph at one hop is the graph.

The budget rides on the member: `HopCount(; n = 2)` rather than a bare `n = 2` beside `sep`. A budget stated apart from the rule that measures it has no statable unit, and becomes a dead field the moment a member measures something other than hops — which [`PathLength`](@ref) does, budgeting in the distance estimator's units instead.

Only [`HopCount`](@ref) is admitted by every consumer, and the split falls on whether the consumer walks a **matrix power**. Both [`clusterise`](@ref) methods accumulate ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)``, so they read `sep.n` as a power count and refuse [`PathLength`](@ref) at dispatch: a radius has no analogue of a matrix power. [`phylogeny_matrix`](@ref) and [`Proximity`](@ref) take either, each through a method of its own — a hop ball is a clamped power sum, a radius ball is a threshold on [`separation_matrix`](@ref).

# Examples

```jldoctest
julia> NetworkEstimator()
NetworkEstimator
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┼ FullMoment()
      │      │     w ┴ nothing
      │   mp ┼ MatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  alg ┼ KruskalTree
      │     args ┼ Tuple{}: ()
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
  sep ┼ HopCount
      │   n ┴ Int64: 1
```

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.4.
  - $(ref_dict[:mantegna1999])
  - $(ref_dict[:tumminello2005])
  - $(ref_dict[:PMFG])
"""
@propagatable @concrete struct NetworkEstimator <: AbstractNetworkEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    """
    $(field_dict[:de])
    """
    @fprop de
    """
    $(field_dict[:ntalg])
    """
    alg <: Tree_SimMat
    """
    $(field_dict[:ntsep])
    """
    sep
    function NetworkEstimator(ce::StatsBase.CovarianceEstimator,
                              de::AbstractDistanceEstimator, alg::Tree_SimMat,
                              sep::AbstractSeparationAlgorithm)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(sep)}(ce, de, alg, sep)
    end
end
function NetworkEstimator(;
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          de::AbstractDistanceEstimator = Distance(;
                                                                   alg = CanonicalDistance()),
                          alg::Tree_SimMat = KruskalTree(),
                          sep::AbstractSeparationAlgorithm = HopCount())
    return NetworkEstimator(ce, de, alg, sep)
end
"""
$(DocStringExtensions.TYPEDEF)

Clusters assets by the pseudo-distances that a network's structure induces.

`NetworkClustersEstimator` encapsulates the configuration for clustering assets from a network, pairing the [`NetworkEstimator`](@ref) that builds the graph with the clustering algorithm and the optimal-number-of-clusters estimator applied to the pseudo-distance matrix it induces.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkClustersEstimator(;
        nte::AbstractNetworkEstimator = NetworkEstimator(),
        alg::AbstractClustersAlgorithm = HClustAlgorithm(),
        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters()
    ) -> NetworkClustersEstimator

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `nte`: Recursively updated via [`factory`](@ref).

The power sums both [`clusterise`](@ref) methods accumulate are indexed by `nte.sep.n`, so the separation budget reaches this estimator through its network estimator rather than being restated here. That also fixes which separations this estimator accepts: `nte.sep` must be a [`HopCount`](@ref), since a power count is what the sums are indexed by. A [`PathLength`](@ref) is constructible here but has no [`clusterise`](@ref) method.

# Examples

```jldoctest
julia> NetworkClustersEstimator()
NetworkClustersEstimator
  nte ┼ NetworkEstimator
      │    ce ┼ PortfolioOptimisersCovariance
      │       │   ce ┼ Covariance
      │       │      │    me ┼ SimpleExpectedReturns
      │       │      │       │   w ┴ nothing
      │       │      │    ce ┼ GeneralCovariance
      │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │      │       │    w ┴ nothing
      │       │      │   alg ┼ FullMoment()
      │       │      │     w ┴ nothing
      │       │   mp ┼ MatrixProcessing
      │       │      │     pdm ┼ Posdef
      │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      │      dn ┼ nothing
      │       │      │      dt ┼ nothing
      │       │      │     alg ┼ nothing
      │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │    de ┼ Distance
      │       │   power ┼ nothing
      │       │     alg ┴ CanonicalDistance()
      │   alg ┼ KruskalTree
      │       │     args ┼ Tuple{}: ()
      │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │   sep ┼ HopCount
      │       │   n ┴ Int64: 1
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

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct NetworkClustersEstimator <: AbstractClustersEstimator
    """
    $(field_dict[:nte])
    """
    @fprop nte
    """
    $(field_dict[:clalg])
    """
    alg
    """
    $(field_dict[:onc])
    """
    onc
    function NetworkClustersEstimator(nte::AbstractNetworkEstimator,
                                      alg::AbstractClustersAlgorithm,
                                      onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(nte), typeof(alg), typeof(onc)}(nte, alg, onc)
    end
end
function NetworkClustersEstimator(; nte::AbstractNetworkEstimator = NetworkEstimator(),
                                  alg::AbstractClustersAlgorithm = HClustAlgorithm(),
                                  onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return NetworkClustersEstimator(nte, alg, onc)
end

export NetworkEstimator, NetworkClustersEstimator
