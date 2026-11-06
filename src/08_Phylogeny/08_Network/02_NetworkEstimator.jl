"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that build a network of assets.

Every concrete or abstract type that builds a network of assets from data must subtype `AbstractNetworkEstimator`. [`phylogeny_matrix`](@ref), [`centrality_graph`](@ref) and the separation kernels dispatch on it.

# Related

  - [`NetworkEstimator`](@ref)
  - [`AbstractCentralityEstimator`](@ref)
"""
abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end
"""
    const NwE_Pl_ClE_Cl = Union{AbstractNetworkEstimator, PhylogenyResult, ClE_Cl}

Union of a network estimator, a phylogeny result, a clustering estimator and a clustering result.

It is the bound of the `pl` argument of [`average_centrality`](@ref), which takes a precomputed [`PhylogenyResult`](@ref) as well as the estimators.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`ClE_Cl`](@ref)
"""
const NwE_Pl_ClE_Cl = Union{AbstractNetworkEstimator, PhylogenyResult, ClE_Cl}
"""
    const NwE_ClE = Union{AbstractNetworkEstimator, AbstractClustersEstimator}

Union of a network estimator and a clustering estimator, with no precomputed result.

It is the bound of the `pl` field of [`SemiDefinitePhylogenyEstimator`](@ref), [`IntegerPhylogenyEstimator`](@ref), [`CentralityEstimator`](@ref) and [`PhylogenyPanel`](@ref). The union leaves out a precomputed result on purpose. A constraint estimator builds its constraint for the set of assets of each call. A precomputed `PhylogenyResult` or `Clusters` holds the answer for the set of assets that made it. The two give the same constraint only while the set of assets does not change.

A meta-optimiser changes the set of assets, because it gives each subproblem a subset of them. [`phylogeny_matrix`](@ref) returns a precomputed result unchanged. A precomputed result in this field therefore gave a subproblem of three assets a constraint matrix over all the assets. The guards for precomputed constraints did not find it, because the object was an estimator. Now the type refuses the shape, so no runtime check is necessary. [`assert_external_optimiser`](@ref) is the one runtime guard on this path, and it catches a precomputed constraint result.

To use a precomputed structure, build the constraint one time and pass the result. `phylogeny_constraints(est, X)` returns a [`SemiDefinitePhylogeny`](@ref) or an [`IntegerPhylogeny`](@ref). The `A` field of each takes a `PhylogenyResult` or a matrix, and the guards for results apply to it.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractClustersEstimator`](@ref)
  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyPanel`](@ref)
"""
const NwE_ClE = Union{AbstractNetworkEstimator, AbstractClustersEstimator}
"""
    const NwE_ClE_Cl = Union{AbstractNetworkEstimator, ClE_Cl}

Union of a network estimator, a clustering estimator and a clustering result.

It is the bound of the `pl` argument of [`centrality_vector`](@ref), of [`asset_phylogeny`](@ref), of [`plot_network`](@ref), and of [`phylogeny_matrix`](@ref) on a Prior Result. A precomputed [`PhylogenyResult`](@ref) goes to a separate method of each function that takes one.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`ClE_Cl`](@ref)
"""
const NwE_ClE_Cl = Union{AbstractNetworkEstimator, ClE_Cl}
"""
$(DocStringExtensions.TYPEDEF)

Builds an asset network from a covariance estimate, and says which pairs of assets it relates.

`ce` and `de` give the distance matrix of the assets. `alg` selects a minimum spanning tree or a PMFG over that matrix. `sep` measures how far apart two assets are in the graph, and it states the budget beyond which two assets count as unrelated.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Tree_SimMat = KruskalTree(),
        sep::AbstractSeparationAlgorithm = HopCount(),
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> NetworkEstimator

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ce_cov]) The tree route under [`VariationInfoDistance`](@ref) reads no correlation, and the constructor refuses a variance estimator on it too, because `ce` holds a covariance estimator on every route.
  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). The constructor throws an `ArgumentError` otherwise.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `de`: Recursively updated through [`factory`](@ref).

# The separation lives here, not on the consumer

`sep` states which pairs the network relates. Every consumer that reads a closure of this graph needs that answer. These consumers are [`phylogeny_matrix`](@ref) and the phylogeny constraint families, both [`clusterise`](@ref) methods, and [`Proximity`](@ref). `sep` is therefore a field of the estimator that builds the graph, and not of one consumer. The constraint path receives only this estimator, so it cannot read a rule that a different consumer holds.

A consumer that reads the structure, and not a closure of it, does not read `sep`. The weighted routes of [`centrality_graph`](@ref) take the weighted graph. A closure is a sum of matrix powers, and a power of a weighted matrix sums products of distances. For example, a [`HopCount`](@ref) of `n = 2` changes a [`DegreeCentrality`](@ref) and does not change a [`ClosenessCentrality`](@ref). At the default `HopCount(; n = 1)` the two routes agree, because the closure of a graph at one hop is the graph.

The budget is a field of the member, as in `HopCount(; n = 2)`, and not a separate field of this estimator. A budget apart from the rule that measures it has no unit. Such a field has no use for a member that does not count hops. [`PathLength`](@ref) is such a member, and it states its budget in the units of the distance estimator.

Every consumer accepts a [`HopCount`](@ref). A consumer that sums matrix powers does not accept a [`PathLength`](@ref). Both [`clusterise`](@ref) methods add up ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)``, so they read `sep.n` as a count of powers. They refuse a [`PathLength`](@ref) at dispatch, because a radius has no equivalent of a matrix power. [`phylogeny_matrix`](@ref) and [`Proximity`](@ref) take either member, each through a method of its own. A hop ball is a clamped sum of powers, and a radius ball is a threshold on [`separation_matrix`](@ref).

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
  x_src ┴ nothing
```

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:mantegna1999]) Page 194, for the minimum spanning tree of a tree `alg`.
  - $(ref_dict[:PMFG]) Section III.A, Algorithm 1, for the graph of a similarity `alg`.
  - $(ref_dict[:graphpo1]) Sections 3.1 and 3.2, for the two networks of a portfolio.
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
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function NetworkEstimator(ce::StatsBase.CovarianceEstimator,
                              de::AbstractDistanceEstimator, alg::Tree_SimMat,
                              sep::AbstractSeparationAlgorithm,
                              x_src::Option{<:AbstractReturnsSource})
        assert_covariance_estimator(ce, :ce)
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :NetworkEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(sep), typeof(x_src)}(ce, de,
                                                                                    alg,
                                                                                    sep,
                                                                                    x_src)
    end
end
function NetworkEstimator(;
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          de::AbstractDistanceEstimator = Distance(;
                                                                   alg = CanonicalDistance()),
                          alg::Tree_SimMat = KruskalTree(),
                          sep::AbstractSeparationAlgorithm = HopCount(),
                          x_src::Option{<:AbstractReturnsSource} = nothing)
    return NetworkEstimator(ce, de, alg, sep, x_src)
end
"""
$(DocStringExtensions.TYPEDEF)

Clusters assets by the pseudo-distances that a network's structure induces.

`nte` builds the network. [`clusterise`](@ref) turns the network into a matrix of pseudo-distances, `alg` clusters that matrix, and `onc` selects the number of clusters.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkClustersEstimator(;
        nte::AbstractNetworkEstimator = NetworkEstimator(),
        alg::AbstractClustersAlgorithm = HClustAlgorithm(),
        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters(),
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> NetworkClustersEstimator

Keywords correspond to the struct's fields.

## Validation

  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). The constructor throws an `ArgumentError` otherwise.
  - `returns_source(nte) === nothing`. The network estimator reads the matrix that this estimator selects, so it states no source of its own, see [`assert_inner_reader`](@ref). The constructor throws an `ArgumentError` otherwise.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `nte`: Recursively updated through [`factory`](@ref).

Both [`clusterise`](@ref) methods index their sums of powers by `nte.sep.n`. The separation budget comes from the network estimator, and this estimator does not state it again. For the same reason, `nte.sep` must be a [`HopCount`](@ref). The constructor accepts a [`PathLength`](@ref), but no [`clusterise`](@ref) method takes it.

# Examples

```jldoctest
julia> NetworkClustersEstimator()
NetworkClustersEstimator
    nte ┼ NetworkEstimator
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
        │     alg ┼ KruskalTree
        │         │     args ┼ Tuple{}: ()
        │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │     sep ┼ HopCount
        │         │   n ┴ Int64: 1
        │   x_src ┴ nothing
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
  x_src ┴ nothing
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
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function NetworkClustersEstimator(nte::AbstractNetworkEstimator,
                                      alg::AbstractClustersAlgorithm,
                                      onc::AbstractOptimalNumberClustersEstimator,
                                      x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :NetworkClustersEstimator)
        assert_inner_reader(nte, :nte, :NetworkClustersEstimator)
        return new{typeof(nte), typeof(alg), typeof(onc), typeof(x_src)}(nte, alg, onc,
                                                                         x_src)
    end
end
function NetworkClustersEstimator(; nte::AbstractNetworkEstimator = NetworkEstimator(),
                                  alg::AbstractClustersAlgorithm = HClustAlgorithm(),
                                  onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters(),
                                  x_src::Option{<:AbstractReturnsSource} = nothing)
    return NetworkClustersEstimator(nte, alg, onc, x_src)
end
function returns_source(x::Union{NetworkEstimator, NetworkClustersEstimator})
    return x.x_src
end
function reader_frames(x::Union{NetworkEstimator, NetworkClustersEstimator},
                       path::AbstractString, ::Bool)
    return source_frames(x.x_src, path, true)
end

export NetworkEstimator, NetworkClustersEstimator
