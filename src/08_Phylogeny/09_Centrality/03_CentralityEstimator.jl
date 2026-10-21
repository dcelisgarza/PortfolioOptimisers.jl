"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all centrality estimator types.

All concrete and/or abstract types implementing centrality-based estimation algorithms should be subtypes of `AbstractCentralityEstimator`.

# Related

  - [`CentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
abstract type AbstractCentralityEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Bundles a network source with the centrality algorithm that scores its assets.

`CentralityEstimator` encapsulates the configuration for computing centrality measures on a network, including the network estimator and the centrality algorithm.

The network is weighted where it can be. [`centrality_polarity`](@ref) answers which quantity `ct` needs — distances for the shortest-path algorithms, similarities for [`EigenvectorCentrality`](@ref) — and [`centrality_graph`](@ref) supplies it from `pl`.

The centrality vector feeds the average centrality ``\\mathbf{C}_n^{\\prime} \\boldsymbol{w}`` of [graphpo1](@cite), Equation 6. That work computes ``\\mathbf{C}_n`` on the unweighted adjacency matrix of the tree or of the TMFG. The weights are this library's choice, and they follow the rule of [pozzi2013](@cite).

The estimator carries no override of its own. A caller who wants the centrality over the network's topology alone configures `ct` itself, with [`TopologyOnly`](@ref) in its `ov` field, and this estimator is a pure bundle of `pl` and `ct` either way.

!!! warning

    Five cases run on the **unweighted** graph, and none of them raises. A caller names a configured algorithm and never asks *for* weights, so an unweightable pairing has not been handed a request it cannot serve. [`TopologyOnly`](@ref) asks *away* from them, which every source can serve, so it adds no case to this list and is not one of the five.

     1. A clustering estimator or a precomputed [`Clusters`](@ref) as `pl`, or a precomputed [`PhylogenyResult`](@ref) passed to [`centrality_vector`](@ref) directly. A partition has no edge weights, and does not borrow any.
     2. [`DegreeCentrality`](@ref). `Graphs.jl` ignores weights.
     3. [`Pagerank`](@ref). `Graphs.jl` ignores weights.
     4. [`KatzCentrality`](@ref). `Graphs.katz_centrality` binarises through `adjacency_matrix(g, Bool)`.
     5. [`EigenvectorCentrality`](@ref) on a tree branch. The branch carries no similarity for it to read.

    On the weighted routes the `sep` field of a [`NetworkEstimator`](@ref) is **inert**: they read the structure itself rather than the separation closure [`phylogeny_matrix`](@ref) builds. At the default `HopCount(; n = 1)` the two agree, because the closure of a graph at one hop is the graph.

[`BetweennessCentrality`](@ref) and [`StressCentrality`](@ref) do read the weights, and are nonetheless unchanged by them on a tree: a tree has exactly one path between any two vertices, so the shortest-path set is the same at any weights. That is a theorem about the graph rather than a limitation of the algorithm, and it does not hold on the similarity branch.

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

  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). An `ArgumentError` is thrown otherwise.
  - `returns_source(pl) === nothing`: the network or clustering estimator reads the matrix that this estimator picks, so it states no source of its own, see [`assert_inner_reader`](@ref). An `ArgumentError` is thrown otherwise.

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
    $(field_dict[:pler])
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
