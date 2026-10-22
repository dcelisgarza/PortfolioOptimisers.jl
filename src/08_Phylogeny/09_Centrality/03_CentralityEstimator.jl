"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that score the centrality of each asset.

Every concrete or abstract type that estimates a centrality vector must subtype `AbstractCentralityEstimator`.

# Related

  - [`CentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
abstract type AbstractCentralityEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Bundles a network source with the centrality algorithm that scores its assets.

`pl` builds the network or the partition, and `ct` scores each asset on it.

The network has weights where it can have them. [`centrality_polarity`](@ref) returns the quantity that `ct` needs, and [`centrality_graph`](@ref) gives that quantity from `pl`. The shortest path algorithms need distances, and [`EigenvectorCentrality`](@ref) needs similarities.

The centrality vector is the input of the average centrality ``\\mathbf{C}_n^{\\prime} \\boldsymbol{w}`` of Equation 6 of [graphpo1](@cite). That work computes ``\\mathbf{C}_n`` on the unweighted adjacency matrix of the tree or of the TMFG. The weights are a choice of this library, and they follow the rule of [pozzi2013](@cite).

The estimator has no override of its own. For the centrality over the topology alone, put [`TopologyOnly`](@ref) in the `ov` field of `ct`. In both cases, this estimator only holds `pl` and `ct`.

!!! warning

    Five cases run on the unweighted graph, and none of them raises an error. A caller names a configured algorithm and never asks for weights. A source and an algorithm that cannot use weights have no request to refuse. [`TopologyOnly`](@ref) asks for no weights, and every source can give that, so it is not one of the five cases.

     1. A clustering estimator as `pl`. A precomputed [`Clusters`](@ref) or [`PhylogenyResult`](@ref) passed directly to [`centrality_vector`](@ref) is the same case. Neither a partition nor a precomputed phylogeny matrix has edge weights, and neither borrows weights.
     2. [`DegreeCentrality`](@ref). `Graphs.jl` does not read the weights.
     3. [`Pagerank`](@ref). `Graphs.jl` does not read the weights.
     4. [`KatzCentrality`](@ref). `Graphs.katz_centrality` reads `adjacency_matrix(g, Bool)`, and raises an `InexactError` on a weighted graph.
     5. [`EigenvectorCentrality`](@ref) on a tree branch. The branch carries no similarity for it to read.

    The weighted routes do not read the `sep` field of a [`NetworkEstimator`](@ref). They read the structure, and not the separation closure that [`phylogeny_matrix`](@ref) builds. At the default `HopCount(; n = 1)` the two agree, because the closure of a graph at one hop is the graph.

[`BetweennessCentrality`](@ref) and [`StressCentrality`](@ref) read the weights, but the weights do not change them on a tree. A tree has exactly one path between two vertices, so the set of shortest paths is the same at all weights. This is a property of the tree, and it does not hold on the similarity branch.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CentralityEstimator(;
        pl::NwE_ClE = NetworkEstimator(),
        ct::AbstractCentralityAlgorithm = DegreeCentrality(),
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> CentralityEstimator

Keywords correspond to the struct's fields.

## Validation

  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). The constructor throws an `ArgumentError` otherwise.
  - `returns_source(pl) === nothing`. The network or clustering estimator reads the matrix that this estimator selects, so it states no source of its own, see [`assert_inner_reader`](@ref). The constructor throws an `ArgumentError` otherwise.

# Examples

```jldoctest
julia> CentralityEstimator()
CentralityEstimator
     pl ┼ NetworkEstimator
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
     ct ┼ DegreeCentrality
        │     kind ┼ Int64: 0
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
  x_src ┴ nothing
```

# Related

  - [`AbstractCentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)

# References

  - $(ref_dict[:graphpo1]) Section 2.4.1, Equation 6, and Section 4, Equation 8.
  - $(ref_dict[:pozzi2013]) Section "Methods", "Centrality and Peripherality measures".
"""
@concrete struct CentralityEstimator <: AbstractCentralityEstimator
    """
    Network estimator or clustering estimator that builds the structure. The bound [`NwE_ClE`](@ref) refuses a precomputed `PhylogenyResult` or `Clusters`.
    """
    pl
    """
    $(field_dict[:cta])
    """
    ct
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function CentralityEstimator(pl::NwE_ClE, ct::AbstractCentralityAlgorithm,
                                 x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :CentralityEstimator)
        assert_inner_reader(pl, :pl, :CentralityEstimator)
        return new{typeof(pl), typeof(ct), typeof(x_src)}(pl, ct, x_src)
    end
end
function CentralityEstimator(; pl::NwE_ClE = NetworkEstimator(),
                             ct::AbstractCentralityAlgorithm = DegreeCentrality(),
                             x_src::Option{<:AbstractReturnsSource} = nothing)
    return CentralityEstimator(pl, ct, x_src)
end
function returns_source(cte::CentralityEstimator)
    return cte.x_src
end
function reader_frames(cte::CentralityEstimator, path::AbstractString, ::Bool)
    return source_frames(cte.x_src, path, true)
end

export CentralityEstimator
