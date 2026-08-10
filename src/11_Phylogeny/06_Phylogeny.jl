"""
$(DocStringExtensions.TYPEDEF)

Container type for phylogeny matrix or vector results.

`PhylogenyResult` stores the output of phylogeny-based estimation routines, such as network or clustering-based phylogeny matrices, or centrality vectors. It is used throughout the package to represent validated phylogeny structures for constraint generation, centrality analysis, and related workflows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyResult(;
        X::ArrNum
    ) -> PhylogenyResult

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:Xe]).
  - $(val_dict[:phX_Xv])

# Examples

```jldoctest
julia> PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0])
PhylogenyResult
  X ┴ 3×3 Matrix{Int64}

julia> PhylogenyResult(; X = [0.2, 0.5, 0.3])
PhylogenyResult
  X ┴ Vector{Float64}: [0.2, 0.5, 0.3]
```

# Related

  - [`AbstractPhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`centrality_vector`](@ref)
"""
@concrete struct PhylogenyResult <: AbstractPhylogenyResult
    """
    $(field_dict[:phX_Xv])
    """
    X
    function PhylogenyResult(X::ArrNum)
        @argcheck(!isempty(X), IsEmptyError)
        if isa(X, MatNum)
            @argcheck(LinearAlgebra.issymmetric(X) && all(iszero, LinearAlgebra.diag(X)),
                      ArgumentError("phylogeny needs a distance matrix (symmetric, zero diagonal). Got a $(ifelse(LinearAlgebra.issymmetric(X), "symmetric", "non-symmetric")) $(ifelse(all(iszero, LinearAlgebra.diag(X)), "zero diagonal", "non-zero diagonal")) matrix."))
        end
        return new{typeof(X)}(X)
    end
end
function PhylogenyResult(; X::ArrNum)::PhylogenyResult
    return PhylogenyResult(X)
end
"""
    phylogeny_matrix(plr::PhylogenyResult{<:MatNum}, args...; kwargs...)

Fallback no-op for returning a validated phylogeny matrix result as-is.

This method provides a generic interface for handling precomputed phylogeny matrices wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for constraint generation and analysis.

# Arguments

  - `plr::PhylogenyResult{<:MatNum}`: Phylogeny matrix result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - The input `plr` object.

# Examples

```jldoctest
julia> plr = PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0]);

julia> phylogeny_matrix(plr)
PhylogenyResult
  X ┴ 3×3 Matrix{Int64}
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(plr::PhylogenyResult{<:MatNum}, args...; kwargs...)
    return plr
end
"""
    centrality_vector(plr::PhylogenyResult{<:VecNum}, args...; kwargs...)

Fallback no-op for returning a validated centrality vector result as-is.

This method provides a generic interface for handling precomputed centrality vectors wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for centrality-based analysis and constraint generation.

# Arguments

  - `plr::PhylogenyResult{<:VecNum}`: Centrality vector result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - The input `plr` object.

# Examples

```jldoctest
julia> plr = PhylogenyResult(; X = [0.2, 0.5, 0.3]);

julia> centrality_vector(plr)
PhylogenyResult
  X ┴ Vector{Float64}: [0.2, 0.5, 0.3]
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(plr::PhylogenyResult{<:VecNum}, args...; kwargs...)
    return plr
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all centrality algorithm types in `PortfolioOptimisers.jl` from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/).

All concrete and/or abstract types implementing specific centrality algorithms (e.g., betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, stress) should be subtypes of `AbstractCentralityAlgorithm`.

# Each member declares the weights it needs

A member says which quantity its edge weights must be, through [`centrality_polarity`](@ref), and [`centrality_graph`](@ref) supplies it. The declaration is about **correctness** — a shortest path over similarities is backwards — and never about capability: a member that declares nothing, and a source that carries no weights, both run on the plain graph rather than raising. The fallback declares `nothing`, so a new member is unweighted until it opts in.

# Related

  - [`centrality_polarity`](@ref)
  - [`centrality_graph`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`DegreeCentrality`](@ref)
  - [`EigenvectorCentrality`](@ref)
  - [`KatzCentrality`](@ref)
  - [`Pagerank`](@ref)
  - [`RadialityCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
abstract type AbstractCentralityAlgorithm <: AbstractPhylogenyAlgorithm end
"""
    assert_centrality_args(::Type{T}, args::Tuple) where {T}

Refuse a matrix inside a centrality algorithm's `args`.

`args` is splatted straight into the `Graphs.jl` centrality function, so a matrix in it is a `distmx` — a second, undeclared way to weight the graph. [`centrality_polarity`](@ref) is the declared one, and it picks the weights the algorithm's own mathematics needs, from the structure that was actually built. Two channels answering the same question is one too many, and this one was never safe:

  - `Graphs.betweenness_centrality`'s `distmx` is its **third** positional argument, so a matrix in `args` binds to `vs` instead and the call **overflows the stack** inside `Graphs.degree`. That takes the session with it, not merely the call.
  - `Graphs.closeness_centrality`'s is its second, so that one worked — silently overriding the polarity, and reporting a wrong-sized matrix as a `BoundsError` rather than a `DimensionMismatch`.
  - `Graphs.stress_centrality` has no `distmx` at all.

Non-matrix entries are untouched: a vertex list or a sample count is a genuine positional argument of those functions and says nothing about weights.

# Arguments

  - `T`: Centrality algorithm type, named in the error message.
  - `args`: Positional arguments destined for the `Graphs.jl` centrality function.

# Returns

  - `nothing`.

# Validation

  - Throws a [`ConflictingArgumentError`](@ref) if any entry of `args` is an `AbstractMatrix`.

# Related

  - [`centrality_polarity`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
function assert_centrality_args(::Type{T}, args::Tuple) where {T}
    idx = findfirst(a -> isa(a, AbstractMatrix), args)
    @argcheck(isnothing(idx),
              ConflictingArgumentError("`args` of a $(T) must not contain a matrix: a weight matrix reaches the centrality algorithm through `centrality_polarity`, not through `args`. Got\nargs[$(idx)] => $(isnothing(idx) ? nothing : typeof(args[idx]))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the centrality vector from a matrix [`PhylogenyResult`](@ref) using the specified centrality algorithm.

Builds a graph from the phylogeny matrix and applies `ct` to compute node centrality scores.

The graph is **always unweighted**, whatever polarity `ct` declares. A precomputed [`PhylogenyResult`](@ref) is a matrix of `0`s and `1`s, so it is one of the weightless sources listed on [`centrality_vector`](@ref)'s warning, and the weights it does not carry cannot be recovered from it. Pass the estimator instead of its result to get the weighted answer.

# Related

  - [`PhylogenyResult`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
  - [`calc_centrality`](@ref)
  - [`centrality_graph`](@ref)
"""
function centrality_vector(plr::PhylogenyResult{<:MatNum}, ct::AbstractCentralityAlgorithm,
                           args...; kwargs...)
    G = Graphs.SimpleGraph(plr.X)
    return PhylogenyResult(; X = calc_centrality(ct, G))
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for betweenness centrality.

`BetweennessCentrality` computes the [betweenness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality) of nodes in a graph, measuring the extent to which a node lies on shortest paths between other nodes.

Declares [`DistancePolarity`](@ref): it is defined over shortest paths, so its weights must be distances. On a tree the weighted answer equals the unweighted one — a tree has exactly one path between any two vertices, so no weighting can change the shortest-path set — which is a theorem about the graph rather than a limitation, and it does not hold on the similarity branch.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BetweennessCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> BetweennessCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BetweennessCentrality()
BetweennessCentrality
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`Graphs.betweenness_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality)
"""
@concrete struct BetweennessCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    function BetweennessCentrality(args::Tuple, kwargs::NamedTuple)
        assert_centrality_args(BetweennessCentrality, args)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BetweennessCentrality(; args::Tuple = (),
                               kwargs::NamedTuple = (;))::BetweennessCentrality
    return BetweennessCentrality(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for closeness centrality.

`ClosenessCentrality` computes the [closeness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality) of nodes in a graph, measuring how close a node is to all other nodes.

Declares [`DistancePolarity`](@ref): it sums shortest-path lengths, so its weights must be distances. It reads them on **both** branches, so its answer on a [`NetworkEstimator`](@ref) source differs from the unweighted one — measured over twenty assets, a maximum absolute change of `0.713` on a triangulated maximally filtered graph and `0.538` on a tree.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClosenessCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> ClosenessCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> ClosenessCentrality()
ClosenessCentrality
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`Graphs.closeness_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality)
"""
@concrete struct ClosenessCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    function ClosenessCentrality(args::Tuple, kwargs::NamedTuple)
        assert_centrality_args(ClosenessCentrality, args)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function ClosenessCentrality(; args::Tuple = (),
                             kwargs::NamedTuple = (;))::ClosenessCentrality
    return ClosenessCentrality(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for degree centrality.

`DegreeCentrality` computes the [degree centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D) of nodes in a graph, measuring the number of edges connected to each node. The `kind` parameter specifies the type of degree (0: total, 1: in-degree, 2: out-degree).

Declares no polarity and runs on the plain graph: `Graphs.degree_centrality` counts edges and ignores what they weigh. It is therefore one of the algorithms for which the estimator's `sep` stays **live** — the unweighted route reads the separation closure [`phylogeny_matrix`](@ref) builds, so `HopCount(; n = 2)` does change this answer.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DegreeCentrality(;
        kind::Integer = 0,
        kwargs::NamedTuple = (;)
    ) -> DegreeCentrality

Keywords correspond to the struct's fields.

## Validation

  - `0 <= kind <= 2`.

# Examples

```jldoctest
julia> DegreeCentrality(; kind = 1)
DegreeCentrality
    kind ┼ Int64: 1
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`Graphs._degree_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D)
"""
@concrete struct DegreeCentrality <: AbstractCentralityAlgorithm
    """
    Degree type (0: total, 1: in-degree, 2: out-degree).
    """
    kind
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    function DegreeCentrality(kind::Integer, kwargs::NamedTuple)
        @argcheck(kind in 0:2, DomainError(kind, "kind must be in 0:2"))
        return new{typeof(kind), typeof(kwargs)}(kind, kwargs)
    end
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))::DegreeCentrality
    return DegreeCentrality(kind, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for [eigenvector centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D).

`EigenvectorCentrality` computes the eigenvector centrality of nodes in a graph, measuring the influence of a node based on the centrality of its neighbors.

Declares [`SimilarityPolarity`](@ref), the only member that does: it is the leading eigenvector of the adjacency matrix itself, so a stronger link must contribute a larger entry. It therefore reads weights on the similarity branch alone. A tree is selected by minimising a distance and carries no similarity, so this algorithm runs unweighted there rather than being handed the wrong quantity.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`Graphs.eigenvector_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D)
"""
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for Katz centrality.

`KatzCentrality` computes the [Katz centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality) of nodes in a graph, measuring the influence of a node based on the number and length of walks between nodes, controlled by the attenuation factor `alpha`.

Declares no polarity and runs on the plain graph: `Graphs.katz_centrality` binarises its input through `adjacency_matrix(g, Bool)`, and throws an `InexactError` when the graph is weighted. The unweighted route is real code here rather than an absent check.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KatzCentrality(;
        alpha::Number = 0.3
    ) -> KatzCentrality

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:katz_alpha])

# Examples

```jldoctest
julia> KatzCentrality(; alpha = 0.5)
KatzCentrality
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`Graphs.katz_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality)
"""
@concrete struct KatzCentrality <: AbstractCentralityAlgorithm
    """
    Attenuation factor for Katz centrality.
    """
    alpha
    function KatzCentrality(alpha::Number)
        @argcheck(zero(alpha) < alpha, DomainError(alpha, "`alpha` must be positive"))
        return new{typeof(alpha)}(alpha)
    end
end
function KatzCentrality(; alpha::Number = 0.3)::KatzCentrality
    return KatzCentrality(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for PageRank.

`Pagerank` computes the [PageRank](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer) of nodes in a graph, measuring the importance of nodes based on the structure of incoming links. The algorithm is controlled by the damping factor `alpha`, number of iterations `n`, and convergence tolerance `epsilon`.

Declares no polarity and runs on the plain graph: `Graphs.pagerank` ignores edge weights. Like [`DegreeCentrality`](@ref) it therefore keeps the estimator's `sep` live, reading the separation closure rather than the structure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Pagerank(;
        alpha::Number = 0.85,
        n::Integer = 100,
        epsilon::Number = 1e-6
    ) -> Pagerank

Keywords correspond to the struct's fields.

## Validation

  - `n > 0`.
  - `0 < alpha < 1`.
  - `epsilon > 0`.

# Examples

```jldoctest
julia> Pagerank(; alpha = 0.9, n = 200, epsilon = 1e-8)
Pagerank
        n ┼ Int64: 200
    alpha ┼ Float64: 0.9
  epsilon ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`Graphs.pagerank`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer)
"""
@concrete struct Pagerank <: AbstractCentralityAlgorithm
    """
    Number of iterations.
    """
    n
    """
    Damping factor.
    """
    alpha
    """
    Convergence threshold.
    """
    epsilon
    function Pagerank(n::Integer, alpha::Number, epsilon::Number)
        @argcheck(0 < n, DomainError)
        assert_unit_interval(alpha, :alpha)
        @argcheck(zero(epsilon) < epsilon, DomainError)
        return new{typeof(n), typeof(alpha), typeof(epsilon)}(n, alpha, epsilon)
    end
end
function Pagerank(; n::Integer = 100, alpha::Number = 0.85,
                  epsilon::Number = 1e-6)::Pagerank
    return Pagerank(n, alpha, epsilon)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for [radiality centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D).

`RadialityCentrality` computes the radiality centrality of nodes in a graph, measuring how close a node is to all other nodes, adjusted for the maximum possible distance.

Declares [`DistancePolarity`](@ref): it reads shortest-path lengths against the graph's diameter, so its weights must be distances. It reads them on both branches, and its answer moves when they arrive — measured over twenty assets, a maximum absolute change of `0.248` on a triangulated maximally filtered graph and `0.234` on a tree.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`Graphs.radiality_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D)
"""
struct RadialityCentrality <: AbstractCentralityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for [stress centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.stress_centrality).

`StressCentrality` computes the stress centrality of nodes in a graph, measuring the number of shortest paths passing through each node.

Declares [`DistancePolarity`](@ref): it counts shortest paths, so its weights must be distances. Like [`BetweennessCentrality`](@ref) it is unchanged by them on a tree, where the shortest-path set is fixed by the structure alone, and does move on the similarity branch.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StressCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> StressCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> StressCentrality()
StressCentrality
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`Graphs.stress_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.stress_centrality)
"""
@concrete struct StressCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    function StressCentrality(args::Tuple, kwargs::NamedTuple)
        assert_centrality_args(StressCentrality, args)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))::StressCentrality
    return StressCentrality(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the polarity of the edge weights a centrality algorithm reads.

A weighted network carries one of two opposite quantities on its edges. A **distance** runs small-is-close; a **similarity** runs large-is-close. Which one an algorithm needs is a fact about its own mathematics and not about the graph it is handed: on one and the same triangulated maximally filtered graph, closeness wants the distances and eigenvector centrality wants the similarities. So the polarity is declared per algorithm, by [`centrality_polarity`](@ref), and the builder supplies the matching quantity.

# Polarity never decides whether the call succeeds

It selects **which** weights an algorithm receives, and nothing else. An algorithm that declares no polarity, and a source that carries no weights, both run on the plain unweighted graph rather than raising — see the warning on [`centrality_vector`](@ref) for the full list. Weightedness is a property of the source, not of the request: there is no flag, so a caller names an algorithm and never asks for weights in the first place.

# Related

  - [`DistancePolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`centrality_polarity`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_graph`](@ref)
"""
abstract type AbstractCentralityPolarity <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Declares that an algorithm's edge weights must be **distances**: small means closely related.

Every algorithm that walks a shortest path needs this polarity, because a shortest path minimises the sum of the weights along it. Over similarities the same routine seeks the route through the *weakest* links and returns a backwards answer without raising.

Supplied by [`calc_distance_weighted_graph`](@ref), which carries distances on both branches.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`centrality_polarity`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
"""
struct DistancePolarity <: AbstractCentralityPolarity end
"""
$(DocStringExtensions.TYPEDEF)

Declares that an algorithm's edge weights must be **similarities**: large means closely related.

An algorithm that reads the weighted adjacency matrix directly, rather than walking a path, needs the entry to grow with relatedness — a stronger link must contribute more.

Supplied by [`calc_weighted_adjacency_graph`](@ref), and only on its similarity branch. The tree branch is selected by [`calc_mst`](@ref) minimising a distance and holds no similarity, so an algorithm declaring this polarity runs unweighted there.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`centrality_polarity`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
"""
struct SimilarityPolarity <: AbstractCentralityPolarity end
"""
    centrality_polarity(ct::AbstractCentralityAlgorithm)
    centrality_polarity(ct::Union{<:BetweennessCentrality, <:ClosenessCentrality,
                                  <:RadialityCentrality, <:StressCentrality})
    centrality_polarity(ct::EigenvectorCentrality)

Declare which quantity a centrality algorithm's edge weights must be.

The extension contract of [`AbstractCentralityPolarity`](@ref). [`centrality_graph`](@ref) reads it to decide what to weight the network with.

# The fallback declares nothing, so opting in is explicit

The method on [`AbstractCentralityAlgorithm`](@ref) returns `nothing`, which routes to the plain unweighted graph. A new algorithm therefore runs unweighted until it says otherwise, which is the safe default: a wrong polarity does not raise, it silently reverses the ordering the algorithm is reading.

# What the shipped members declare, and why

  - [`DistancePolarity`](@ref) — [`BetweennessCentrality`](@ref), [`ClosenessCentrality`](@ref), [`RadialityCentrality`](@ref), [`StressCentrality`](@ref). All four are defined over shortest paths.
  - [`SimilarityPolarity`](@ref) — [`EigenvectorCentrality`](@ref). It is the leading eigenvector of the adjacency matrix itself, so a larger entry must mean a stronger link.
  - `nothing` — [`DegreeCentrality`](@ref), [`Pagerank`](@ref), [`KatzCentrality`](@ref). `Graphs.jl` cannot use weights in any of the three: the first two ignore them, and `Graphs.katz_centrality` binarises through `adjacency_matrix(g, Bool)` and throws an `InexactError` when handed a weighted graph.

The line between the first two groups and the third is `Graphs.jl`'s own. The declaration is about correctness — which weights — and the absence of one is about capability.

# Arguments

  - $(field_dict[:cta])

# Returns

  - `polarity::Option{<:AbstractCentralityPolarity}`: The declared polarity, or `nothing` for an algorithm that cannot read weights.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`centrality_graph`](@ref)
  - [`calc_centrality`](@ref)
"""
function centrality_polarity end
function centrality_polarity(::AbstractCentralityAlgorithm)::Option{<:AbstractCentralityPolarity}
    return nothing
end
function centrality_polarity(::Union{<:BetweennessCentrality, <:ClosenessCentrality,
                                     <:RadialityCentrality, <:StressCentrality})::Option{<:AbstractCentralityPolarity}
    return DistancePolarity()
end
function centrality_polarity(::EigenvectorCentrality)::Option{<:AbstractCentralityPolarity}
    return SimilarityPolarity()
end
"""
    calc_centrality(ct::AbstractCentralityAlgorithm, g::Graphs.AbstractGraph)

Compute node centrality scores for a graph using the specified centrality algorithm.

This function dispatches to the appropriate centrality computation from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/) based on the type of `ct`. Supported algorithms include betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, and stress centrality.

`g` may be weighted or unweighted, and nothing here inspects which. `Graphs.jl` weights implicitly — the `distmx` of every routine that takes one defaults to `weights(g)` — so the choice is made once, by [`centrality_graph`](@ref), and this function only forwards. Handing a weighted graph to an algorithm that declares no polarity is what [`centrality_graph`](@ref) exists to prevent: `Graphs.katz_centrality` throws an `InexactError` on one.

# Arguments

  - `ct`: Centrality algorithm to use.

      + `ct::BetweennessCentrality`: Computes betweenness centrality.
      + `ct::ClosenessCentrality`: Computes closeness centrality.
      + `ct::DegreeCentrality`: Computes degree centrality.
      + `ct::EigenvectorCentrality`: Computes eigenvector centrality.
      + `ct::KatzCentrality`: Computes Katz centrality.
      + `ct::Pagerank`: Computes PageRank.
      + `ct::RadialityCentrality`: Computes radiality centrality.
      + `ct::StressCentrality`: Computes stress centrality.

  - `g`: Graph to compute centrality on.

# Returns

  - `ct::VecNum`: Centrality scores for each node in the graph.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`DegreeCentrality`](@ref)
  - [`EigenvectorCentrality`](@ref)
  - [`KatzCentrality`](@ref)
  - [`Pagerank`](@ref)
  - [`RadialityCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
function calc_centrality(ct::BetweennessCentrality, g::Graphs.AbstractGraph)
    return Graphs.betweenness_centrality(g, ct.args...; ct.kwargs...)
end
function calc_centrality(ct::ClosenessCentrality, g::Graphs.AbstractGraph)
    return Graphs.closeness_centrality(g, ct.args...; ct.kwargs...)
end
function calc_centrality(ct::DegreeCentrality, g::Graphs.AbstractGraph)
    return Graphs._degree_centrality(g, ct.kind; ct.kwargs...)
end
function calc_centrality(::EigenvectorCentrality, g::Graphs.AbstractGraph)
    return Graphs.eigenvector_centrality(g::Graphs.AbstractGraph)
end
function calc_centrality(ct::KatzCentrality, g::Graphs.AbstractGraph)
    return Graphs.katz_centrality(g, ct.alpha)
end
function calc_centrality(ct::Pagerank, g::Graphs.AbstractGraph)
    return Graphs.pagerank(g, ct.alpha, ct.n, ct.epsilon)
end
function calc_centrality(::RadialityCentrality, g::Graphs.AbstractGraph)
    return Graphs.radiality_centrality(g::Graphs.AbstractGraph)
end
function calc_centrality(ct::StressCentrality, g::Graphs.AbstractGraph)
    return Graphs.stress_centrality(g, ct.args...; ct.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all minimum spanning tree (MST) algorithm types.

All concrete and/or abstract types implementing specific MST algorithms (e.g., Kruskal, Boruvka, Prim) should be subtypes of `AbstractTreeType`.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
"""
    const Tree_SimMat = Union{<:AbstractNonNegativeSimilarityMatrixAlgorithm,
                              <:AbstractTreeType}

Alias for a tree or similarity matrix algorithm.

Matches either an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) or an [`AbstractTreeType`](@ref). Used for dispatch in phylogeny estimation where either a spanning tree or a similarity matrix approach may be used.

The similarity half is the **narrow** family, not [`AbstractSimilarityMatrixAlgorithm`](@ref): the similarity branch builds a PMFG, whose consumers cannot take a negative weight. So [`MaximumDistanceSimilarity`](@ref), [`ExponentialSimilarity`](@ref), [`GeneralExponentialSimilarity`](@ref) and [`ComplementSimilarity`](@ref) match, and [`AngularSimilarity`](@ref) does not.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`NetworkEstimator`](@ref)
"""
const Tree_SimMat = Union{<:AbstractNonNegativeSimilarityMatrixAlgorithm,
                          <:AbstractTreeType}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for Kruskal's minimum spanning tree (MST).

`KruskalTree` specifies the use of [Kruskal's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst) for constructing a minimum spanning tree from a graph.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KruskalTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> KruskalTree

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> KruskalTree()
KruskalTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.kruskal_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst)
"""
@concrete struct KruskalTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function KruskalTree(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))::KruskalTree
    return KruskalTree(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for Boruvka's minimum spanning tree (MST).

`BoruvkaTree` specifies the use of [Boruvka's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst) for constructing a minimum spanning tree from a graph.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoruvkaTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> BoruvkaTree

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BoruvkaTree()
BoruvkaTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.boruvka_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst)
"""
@concrete struct BoruvkaTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function BoruvkaTree(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))::BoruvkaTree
    return BoruvkaTree(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for Prim's minimum spanning tree (MST).

`PrimTree` specifies the use of [Prim's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst) for constructing a minimum spanning tree from a graph.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PrimTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> PrimTree

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PrimTree()
PrimTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.prim_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst)
"""
@concrete struct PrimTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function PrimTree(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree(args, kwargs)
end
"""
    calc_mst(alg::AbstractTreeType, g::Graphs.AbstractGraph)

Compute the minimum spanning tree (MST) of a graph using the specified algorithm.

This function dispatches to the appropriate MST computation from `Graphs.jl` based on the type of `alg`. Supported algorithms include Kruskal, Boruvka, and Prim.

# Arguments

  - `alg`: MST algorithm to use.

      + `alg::KruskalTree`: Computes the MST using Kruskal's algorithm.
      + `alg::BoruvkaTree`: Computes the MST using Boruvka's algorithm.
      + `alg::PrimTree`: Computes the MST using Prim's algorithm.

  - `g::Graphs.AbstractGraph`: Graph to compute the MST on.

# Returns

  - `tree::Vector`: Vector of edges representing the MST.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
function calc_mst(ct::KruskalTree, g::Graphs.AbstractGraph)
    return Graphs.kruskal_mst(g, ct.args...; ct.kwargs...)
end
function calc_mst(ct::BoruvkaTree, g::Graphs.AbstractGraph)
    return Graphs.boruvka_mst(g, ct.args...; ct.kwargs...)[1]
end
function calc_mst(ct::PrimTree, g::Graphs.AbstractGraph)
    return Graphs.prim_mst(g, ct.args...; ct.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all network estimator types.

All concrete and/or abstract types implementing network-based estimation algorithms should be subtypes of `AbstractNetworkEstimator`.

# Related

  - [`NetworkEstimator`](@ref)
  - [`AbstractCentralityEstimator`](@ref)
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

Estimator type for network-based phylogeny analysis.

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
      │      │   alg ┴ FullMoment()
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

Estimator type for network-based clustering.

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
      │       │      │   alg ┴ FullMoment()
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
"""
@propagatable @concrete struct NetworkClustersEstimator <: AbstractClustersEstimator
    """
    Network estimator.
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

Estimator type for centrality-based analysis.

`CentralityEstimator` encapsulates the configuration for computing centrality measures on a network, including the network estimator and the centrality algorithm.

The network is weighted where it can be. [`centrality_polarity`](@ref) declares which quantity `ct` needs — distances for the shortest-path algorithms, similarities for [`EigenvectorCentrality`](@ref) — and [`centrality_graph`](@ref) supplies it from `pl`.

!!! warning

    Five cases run on the **unweighted** graph, and none of them raises. A caller names an algorithm and never asks for weights, so an unweightable pairing has not been handed a request it cannot serve.

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
        ct::AbstractCentralityAlgorithm = DegreeCentrality()
    ) -> CentralityEstimator

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CentralityEstimator()
CentralityEstimator
  pl ┼ NetworkEstimator
     │    ce ┼ PortfolioOptimisersCovariance
     │       │   ce ┼ Covariance
     │       │      │    me ┼ SimpleExpectedReturns
     │       │      │       │   w ┴ nothing
     │       │      │    ce ┼ GeneralCovariance
     │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │      │       │    w ┴ nothing
     │       │      │   alg ┴ FullMoment()
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
  ct ┼ DegreeCentrality
     │     kind ┼ Int64: 0
     │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
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
    function CentralityEstimator(pl::NwE_ClE, ct::AbstractCentralityAlgorithm)
        return new{typeof(pl), typeof(ct)}(pl, ct)
    end
end
function CentralityEstimator(; pl::NwE_ClE = NetworkEstimator(),
                             ct::AbstractCentralityAlgorithm = DegreeCentrality())
    return CentralityEstimator(pl, ct)
end
"""
    graph_weight_matrix(D::MatNum)

Return `D` as a matrix whose off-diagonal entries are representable as `SimpleWeightedGraphs` edge weights.

A distance matrix and a weighted graph disagree about what `0` means. In the distance codomain `0` is the *floor* — two assets as close as they can be. In the graph representation `0` is the reserved value meaning *absent*: `SimpleWeightedGraph` sparsifies its input, and `add_edge!` with a zero weight refuses outright. Handing a zero distance straight to the constructor therefore deletes exactly the edge the minimum spanning tree most wants, and the two assets come out non-adjacent — the most related pair in the universe reported as unrelated, with no error raised.

A zero is not a symptom of bad data. `SimpleAbsoluteDistance` and `LogDistance` are defined on `abs(rho)`, so a perfectly *anti*-correlated pair — a long/short leg, an inverse ETF, a pairs trade — is at distance zero and is genuinely maximally related. The square-root algorithms reach zero from the other side, since their `clamp!` maps any `rho >= 1` to exactly zero.

So the zero is *repaired*, not rejected: each off-diagonal zero moves to `nextfloat(zero(eltype(D)))`, the smallest representable positive value. That is the nearest value the representation can carry, it is orders of magnitude below any distance a caller could mean, and it is absorbed exactly by any sum it enters. `D` itself is returned untouched when no entry needs moving, so the copy is only paid for when it buys something.

Negative and `NaN` entries have no such nearest representable value and are rejected. A negative distance inverts the ordering it expresses and is *unsound* rather than merely wrong under the shortest-path routines that consume these weights — they return an answer instead of raising. A `NaN` — which a zero-variance asset produces, via a `NaN` correlation — silently fails every comparison the tree algorithms make.

`Inf` is left alone: it is the honest distance between uncorrelated assets under [`LogDistance`](@ref), the graph accepts it, and a spanning tree simply takes those edges last.

# Arguments

  - `D`: Symmetric distance matrix.

# Returns

  - `W::MatNum`: `D` itself, or a repaired copy of it.

# Validation

  - Throws a `DomainError` if any off-diagonal entry is negative or `NaN`.

# Related

  - [`calc_weighted_adjacency_graph`](@ref)
  - [`clusterise`](@ref)
"""
function graph_weight_matrix(D::MatNum)
    z = zero(eltype(D))
    repair = false
    for j in axes(D, 2), i in axes(D, 1)
        if i == j
            continue
        end
        d = D[i, j]
        @argcheck(!isnan(d) && d >= z,
                  DomainError(d,
                              "off-diagonal distances must be non-negative and not NaN, because a graph edge weight is. Got\nD[$i, $j] => $d"))
        repair |= iszero(d)
    end
    if !repair
        return D
    end
    W = copy(D)
    tiny = nextfloat(z)
    for j in axes(W, 2), i in axes(W, 1)
        if i != j && iszero(W[i, j])
            W[i, j] = tiny
        end
    end
    return W
end
"""
    calc_weighted_adjacency_graph(alg::AbstractTreeType, D::MatNum)
    calc_weighted_adjacency_graph(alg::AbstractNonNegativeSimilarityMatrixAlgorithm,
                                  S::MatNum)
    calc_weighted_adjacency_graph(nte::NetworkEstimator, X::MatNum; dims::Int = 1,
                                  kwargs...)

Build the weighted graph whose edges are the network structure.

This is the **one construction site** of that structure. [`calc_weighted_adjacency`](@ref) and [`calc_adjacency`](@ref) are each a single operation on the graph returned here, so how the structure is selected is decided in this function and nowhere else.

# Polarity is per branch, and the two branches are not interchangeable

Each branch keeps **the quantity that selected its own edges**, and the two quantities run in opposite directions.

  - `alg::AbstractTreeType`: [`calc_mst`](@ref) minimises the distance, so the weights are **distances**. Small means closely related.
  - `alg::AbstractNonNegativeSimilarityMatrixAlgorithm`: [`PMFG_T2s`](@ref) maximises the gain over the similarity, so the weights are **similarities**. Large means closely related.

Re-weighting either branch with the other quantity would weight a structure by the quantity that did not select it, so neither is converted. The result carries no polarity tag, because the polarity is recoverable by dispatch on the algorithm.

A consumer that walks a path must therefore branch on `nte.alg` first. A shortest path over similarities inverts the ordering it is meant to express and **returns an answer instead of raising**, so the two graphs are interchangeable in shape but not in meaning.

# Two entry points, because the selecting quantity is not always cheap

The two-argument methods take **the selecting quantity itself** — the distance on the tree branch, the similarity on the PMFG branch — and the three-argument method derives it from `X` and forwards. Which branch is which is decided by the same dispatch either way, so the polarity above is a property of the algorithm and not of the entry point.

The two-argument form exists for a caller that already holds that matrix. [`clusterise`](@ref) is one: it needs `D` and `S` for its own power sum and for the [`Clusters`](@ref) it returns, so re-deriving them here would compute the same correlation twice. That is not a rounding error — under [`VariationInfoDistance`](@ref) the derivation is `98%` of `clusterise`'s runtime, so the second one would almost double it.

# The weights

  - Tree branch: strictly positive, and finite or `Inf`. [`graph_weight_matrix`](@ref) moves every zero distance off the value the representation reserves for *absent*, and rejects a negative or a `NaN`. `Inf` is legal — it is the honest [`LogDistance`](@ref) between two uncorrelated assets.
  - PMFG branch: non-negative and finite. [`PMFG_T2s`](@ref) checks its input for non-negativity, and declines an edge whose gain is zero, so no stored zero reaches the graph.

# Arguments

  - `alg`: Tree or similarity matrix algorithm.
  - $(arg_dict[:D])
  - $(arg_dict[:S])
  - `nte`: Network estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `G::SimpleWeightedGraphs.SimpleWeightedGraph`: The network structure, carrying its branch's own weights.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_adjacency`](@ref)
  - [`graph_weight_matrix`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`clusterise`](@ref)
"""
function calc_weighted_adjacency_graph(alg::AbstractTreeType, D::MatNum)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(graph_weight_matrix(D))
    return G[calc_mst(alg, G)]
end
function calc_weighted_adjacency_graph(::AbstractNonNegativeSimilarityMatrixAlgorithm,
                                       S::MatNum)
    return SimpleWeightedGraphs.SimpleWeightedGraph(PMFG_T2s(S)[1])
end
function calc_weighted_adjacency_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                             <:AbstractTreeType}, X::MatNum;
                                       dims::Int = 1, kwargs...)
    return calc_weighted_adjacency_graph(nte.alg,
                                         distance(nte.de, nte.ce, X; dims = dims,
                                                  kwargs...))
end
function calc_weighted_adjacency_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                             <:AbstractNonNegativeSimilarityMatrixAlgorithm},
                                       X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(nte.de, nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.alg, nte.de, D)
    return calc_weighted_adjacency_graph(nte.alg,
                                         distance_to_similarity(nte.alg; S = S, D = D))
end
"""
    calc_weighted_adjacency(alg::Tree_SimMat, W::MatNum)
    calc_weighted_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the weighted adjacency matrix of the network structure.

`Graphs.adjacency_matrix` of a *weighted* graph returns the **weights**, not `0`/`1`, so this is the matrix form of [`calc_weighted_adjacency_graph`](@ref) and inherits that function's per-branch polarity unchanged: distances on the tree branch, similarities on the PMFG branch. Read the polarity section of [`calc_weighted_adjacency_graph`](@ref) before consuming the values.

The sparsity pattern is the structure itself, so it is identical to [`calc_adjacency`](@ref)'s on the same input. Only the stored values differ.

The two entry points are [`calc_weighted_adjacency_graph`](@ref)'s two entry points, one `Graphs.adjacency_matrix` call further on. `W` is the selecting quantity — the distance on the tree branch, the similarity on the PMFG branch — and [`clusterise`](@ref) supplies it directly, having already paid for it.

# Arguments

  - `alg`: Tree or similarity matrix algorithm.
  - `W`: Selecting quantity of `alg`'s branch: a distance matrix under an [`AbstractTreeType`](@ref), a similarity matrix under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref).
  - `nte`: Network estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC`: Weighted adjacency matrix of the network, in its branch's own polarity.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`calc_adjacency`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`clusterise`](@ref)
"""
function calc_weighted_adjacency(alg::Tree_SimMat, W::MatNum)
    return Graphs.adjacency_matrix(calc_weighted_adjacency_graph(alg, W))
end
function calc_weighted_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return Graphs.adjacency_matrix(calc_weighted_adjacency_graph(nte, X; dims = dims,
                                                                 kwargs...))
end
"""
    calc_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the binary adjacency matrix for a network estimator.

The structure comes from [`calc_weighted_adjacency_graph`](@ref); this function is the round trip through `Graphs.SimpleGraph` that discards the weights. Both branches share the one body, because the branch is decided in the tier below.

Consumers that need the weights call [`calc_weighted_adjacency`](@ref) instead. They must then observe the per-branch polarity documented on [`calc_weighted_adjacency_graph`](@ref). The binarisation here is what exempts this function from it.

# Arguments

  - `nte`: Network estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC{Int, Int}`: Binary adjacency matrix representing the network.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
"""
function calc_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(calc_weighted_adjacency_graph(nte, X;
                                                                                    dims = dims,
                                                                                    kwargs...)))
end
"""
    calc_distance_weighted_graph(nte::NetworkEstimator, X::MatNum; dims::Int = 1,
                                 kwargs...)

Build the network structure carrying **distances** on its edges, on either branch.

[`calc_weighted_adjacency_graph`](@ref) gives each branch the quantity that *selected* its edges, so its two branches hold opposite polarities. This function gives both branches the same one. The structure is unchanged — it is the same edge set, vertex for vertex — and only the weights differ, on the PMFG branch alone.

# Why the PMFG branch may be re-weighted here, and may not be there

Re-weighting a structure with a quantity that did not select it is what [`calc_weighted_adjacency_graph`](@ref) refuses. This is not that. Every [`AbstractSimilarityMatrixAlgorithm`](@ref) is a strictly decreasing function of the distance, so the similarity that selected the PMFG's edges is a **monotone image of `D`**, not a foreign quantity: `D` is the selecting quantity's preimage, and the same PMFG comes out of it.

What must not happen is a path taken over the similarities themselves. A shortest path *minimises* the sum of its edge weights, so over similarities it seeks the route through the **weakest** links — the ordering it produces is backwards. It is also quiet about it: measured over the four similarity algorithms, the backwards answer correlates `0.95` to `0.97` with the right one, which is close enough to pass a glance and not close enough to be usable.

# Arguments

  - `nte`: Network estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `G::SimpleWeightedGraphs.SimpleWeightedGraph`: The network structure, weighted by distance on both branches.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`graph_weight_matrix`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
"""
function calc_distance_weighted_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                            <:AbstractTreeType}, X::MatNum;
                                      dims::Int = 1, kwargs...)
    # The tree branch already weights its edges with `D`, so the distance-weighted structure
    # and the selecting-quantity one are the same graph.
    return calc_weighted_adjacency_graph(nte, X; dims = dims, kwargs...)
end
function calc_distance_weighted_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                            <:AbstractNonNegativeSimilarityMatrixAlgorithm},
                                      X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(nte.de, nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.alg, nte.de, D)
    S = distance_to_similarity(nte.alg; S = S, D = D)
    # `PMFG_T2s` selects the edges from the similarity; `D` then supplies their lengths. The
    # repair is `calc_weighted_adjacency_graph`'s tree-branch one, needed here for the same
    # reason -- a zero distance is the value the representation reserves for *absent*.
    W = graph_weight_matrix(D)
    r, c, _ = SparseArrays.findnz(PMFG_T2s(S)[1])
    v = [W[i, j] for (i, j) in zip(r, c)]
    return SimpleWeightedGraphs.SimpleWeightedGraph(SparseArrays.sparse(r, c, v,
                                                                        size(W)...))
end
"""
    separation_matrix(sep::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)
    separation_matrix(sep::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)

Compute the dense `assets × assets` matrix of separations under a separation algorithm.

Half of the extension contract of [`AbstractSeparationAlgorithm`](@ref); [`separation_budget`](@ref) is the other half.

# The unreachable sentinel

An unreachable pair carries whatever sentinel the underlying routine uses, **not** a repaired value: `Graphs.gdistances` reports `typemax(Int)` for [`HopCount`](@ref), and `Graphs.floyd_warshall_shortest_paths` reports `typemax(T)` for [`PathLength`](@ref), which on the `Float64` weights it is handed is `Inf`. Callers must compare against the budget before doing anything else with the entry, and that comparison has to **short-circuit** — an `ifelse` evaluates both branches, and [`ReciprocalDecay`](@ref) overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError` rather than a discarded number.

# The two shipped members read the same structure differently

[`HopCount`](@ref) counts the edges of [`calc_adjacency`](@ref); [`PathLength`](@ref) sums the distances along them, over [`calc_distance_weighted_graph`](@ref). All-pairs shortest paths come from one `floyd_warshall_shortest_paths` call rather than a Dijkstra per vertex — measured about **7 times faster** on this shape, and within about `1.3` times of the breadth-first loop the hop count uses.

# Arguments

  - `sep`: Separation algorithm.
  - `nte`: Network estimator. The graph is rebuilt from `X` on every call, through [`calc_adjacency`](@ref) or [`calc_distance_weighted_graph`](@ref).
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments forwarded to the underlying phylogeny routines.

# Returns

  - `d::Matrix`: Square matrix of separations. `d[i, k]` is the separation between assets `i` and `k`, `d[i, i]` is zero, and an unreachable pair carries the sentinel above.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_budget`](@ref)
  - [`calc_adjacency`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
  - [`Proximity`](@ref)
"""
function separation_matrix end
function separation_matrix(::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                           dims::Int = 1, kwargs...)
    g = Graphs.SimpleGraph(calc_adjacency(nte, X; dims = dims, kwargs...))
    d = Matrix{Int}(undef, Graphs.nv(g), Graphs.nv(g))
    for v in Graphs.vertices(g)
        @inbounds d[:, v] = Graphs.gdistances(g, v)
    end
    return d
end
function separation_matrix(::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                           dims::Int = 1, kwargs...)
    g = calc_distance_weighted_graph(nte, X; dims = dims, kwargs...)
    return Graphs.floyd_warshall_shortest_paths(g).dists
end
"""
    separation_budget(sep::HopCount, nte::AbstractNetworkEstimator, d::MatNum)
    separation_budget(sep::PathLength, nte::AbstractNetworkEstimator, d::MatNum)

Resolve the separation budget in scope: the separation beyond which a pair counts as unrelated.

Half of the extension contract of [`AbstractSeparationAlgorithm`](@ref); [`separation_matrix`](@ref) is the other half. Split from `separation_matrix` because a consumer needs the budget on its own — to probe a decay before entering the `assets × assets` loop, or to threshold a matrix it already holds.

# The separations are passed in, not recomputed

`d` is the matrix [`separation_matrix`](@ref) produced, so a member whose budget is *observed* rather than configured — the diameter of what the graph actually turned out to be — can read it without a second traversal. That is why the argument is the **matrix** and not a diameter: finding the largest finite entry is itself the `assets²` reduction, so passing a diameter would charge [`HopCount`](@ref) for one it ignores. Handing over `d` pushes the reduction into [`PathLength`](@ref), the member that wants it.

`nte` is inert for what ships: it is the channel through which an extension budget can see the estimator that owns it. Inert arguments have precedent here — [`separation_decay`](@ref)'s `dmax` is read by only one of five members.

# The observed diameter is a ceiling, not only a default

[`PathLength`](@ref) clamps a chosen `dmax` to the observed diameter as well as substituting the diameter for `nothing`. The clamp **truncates nothing** — no pair sits beyond the diameter — so it is a scale-top correction and is visible only through [`LinearDecay`](@ref), the one decay reading the budget. Without it, `dmax = 100` on a graph of diameter `3.5` would flatten the scores towards a constant while forbidding no pair at all.

# Arguments

  - `sep`: Separation algorithm.
  - `nte`: Network estimator that owns `sep`. **Inert** for the shipped members.
  - `d`: Separation matrix from [`separation_matrix`](@ref). **Inert** for [`HopCount`](@ref), whose budget is configured rather than observed; read by [`PathLength`](@ref), whose budget is capped by what the graph turned out to be.

# Returns

  - `dmax::Number`: Separation budget. Stated in the units `sep` measures in — hops for [`HopCount`](@ref), the distance estimator's units for [`PathLength`](@ref) — so it is only ever compared against entries of `d`.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
function separation_budget end
function separation_budget(sep::HopCount, ::AbstractNetworkEstimator, ::MatNum)::Number
    return sep.n
end
function separation_budget(sep::PathLength, ::AbstractNetworkEstimator, d::MatNum)::Number
    # The diameter of what the graph turned out to be. The sentinel is excluded rather than
    # repaired: an unreachable pair is not a long one, and taking it as the diameter would
    # make the budget `Inf`, which `LinearDecay` scores `Inf` at every separation.
    delta = zero(eltype(d))
    for dij in d
        if isfinite(dij) && dij > delta
            delta = dij
        end
    end
    return isnothing(sep.dmax) ? delta : min(sep.dmax, delta)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Internal dispatch helper for constructing a [`Clusters`](@ref) result within a network-based clustering workflow.

Selects the appropriate clustering routine based on `alg`, determines the optimal number of clusters, and returns a [`Clusters`](@ref) result encapsulating all relevant outputs.

# Arguments

  - `alg`: Clustering algorithm.

      + `alg::HClustAlgorithm`: Applies hierarchical clustering via `Clustering.hclust` on the pseudo-distance matrix `P`.
      + `alg::DBHT`: Applies Direct Bubble Hierarchical Tree clustering via [`DBHTs`](@ref) on `P` and `S`.
      + `alg::AbstractNonHierarchicalClusteringAlgorithm`: Applies non-hierarchical clustering via [`optimal_number_clusters`](@ref) on `P`.

  - $(arg_dict[:onc])

  - $(arg_dict[:S])

  - $(arg_dict[:D])

  - `P::MatNum`: Symmetric pseudo-distance matrix derived from the network or similarity structure.

  - `branchorder`: Branch ordering strategy for hierarchical clustering.

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

# Related

  - [`Clusters`](@ref)
  - [`HClustAlgorithm`](@ref)
  - [`DBHT`](@ref)
  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`clusterise`](@ref)
"""
function _clusterise(alg::HClustAlgorithm, onc::AbstractOptimalNumberClustersEstimator,
                     S::MatNum, D::MatNum, P::MatNum; branchorder::Symbol = :optimal)
    res = Clustering.hclust(P; linkage = alg.linkage, branchorder = branchorder)
    k = optimal_number_clusters(onc, res, P)
    return Clusters(; res = res, S = S, D = D, P = P, k = k)
end
function _clusterise(alg::DBHT, onc::AbstractOptimalNumberClustersEstimator, S::MatNum,
                     D::MatNum, P::MatNum; branchorder::Symbol = :optimal)
    res = DBHTs(P, S; branchorder = branchorder, root = alg.root)[end]
    k = optimal_number_clusters(onc, res, P)
    return Clusters(; res = res, S = S, D = D, P = P, k = k)
end
function _clusterise(alg::AbstractNonHierarchicalClusteringAlgorithm,
                     onc::AbstractOptimalNumberClustersEstimator, S::MatNum, D::MatNum,
                     P::MatNum; kwargs...)
    res, k = optimal_number_clusters(onc, alg, P)
    return Clusters(; res = res, S = S, D = D, P = P, k = k)
end
"""
    clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                <:AbstractTreeType,
                                                                <:HopCount}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster assets using a minimum spanning tree (MST) network structure and return a [`Clusters`](@ref) result.

Builds the MST from the distance matrix, accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

``\\mathbf{A}`` is [`calc_weighted_adjacency`](@ref)'s, so this method reads the same structure as every other consumer of a network and carries **weights**, not `0`/`1` — the tree branch's polarity is the distance, which is what ``\\mathbf{D}^i - \\mathbf{A}^i`` subtracts a like quantity from. The two-argument entry point is the one used, because `D` is already in hand.

# Only a hop count is admitted

The fourth type parameter is narrowed to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at **dispatch**. The power sum is indexed by `nte.nte.sep.n`, and a matrix power counts edges: there is no radius analogue of ``\\mathbf{D}^i - \\mathbf{A}^i``, so the refusal is the honest answer rather than a gap. [`phylogeny_matrix`](@ref) does have a radius method, so the two consumers of a network differ here on purpose.

# Arguments

  - `nte`: Network clustering estimator configured with an MST-based [`NetworkEstimator`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

# Related

  - [`NetworkClustersEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`Clusters`](@ref)
  - [`_clusterise`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_mst`](@ref)
  - [`HopCount`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractTreeType,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    P = zeros(eltype(D), size(D))
    # The distance is the tree branch's selecting quantity, and it is in hand already for
    # the power sum below, so the shared routine is entered at its two-argument form.
    A = calc_weighted_adjacency(nte.nte.alg, D)
    # `nte.nte.sep.n` is read as a matrix-power count rather than as a budget: a separation
    # member that measures something other than hops has no `n`, and fails here rather than
    # silently truncating a power sum it cannot index.
    for i in 0:(nte.nte.sep.n)
        P .+= D^i - A^i
    end
    P .-= LinearAlgebra.Diagonal(P)
    return _clusterise(nte.alg, nte.onc, S, D, LinearAlgebra.Symmetric(P);
                       branchorder = branchorder)
end
"""
    clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                <:HopCount}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster assets using a Planar Maximally Filtered Graph (PMFG) network structure and return a [`Clusters`](@ref) result.

Builds the PMFG from the similarity matrix via [`PMFG_T2s`](@ref), accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{S}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

``\\mathbf{A}`` is [`calc_weighted_adjacency`](@ref)'s, as on the tree method, and this branch's polarity is the **similarity** — so ``\\mathbf{S}^i - \\mathbf{A}^i`` again subtracts a like quantity. The two-argument entry point is the one used, because `S` is already in hand.

# Only a hop count is admitted

The fourth type parameter is narrowed to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at **dispatch**. See the tree method: a matrix power counts edges, and there is no radius analogue of the power sum.

# Arguments

  - `nte`: Network clustering estimator configured with a similarity-matrix-based [`NetworkEstimator`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

# Related

  - [`NetworkClustersEstimator`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`Clusters`](@ref)
  - [`_clusterise`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`HopCount`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.nte.alg, nte.nte.de, D)
    P = zeros(eltype(D), size(D))
    S = distance_to_similarity(nte.nte.alg; S = S, D = D)
    # The similarity is the PMFG branch's selecting quantity. See the tree method.
    Rpm = calc_weighted_adjacency(nte.nte.alg, S)
    # See the tree method: a matrix-power count, not a budget.
    for i in 0:(nte.nte.sep.n)
        P .+= S^i - Rpm^i
    end
    P .-= LinearAlgebra.Diagonal(P)
    return _clusterise(nte.alg, nte.onc, S, D, LinearAlgebra.Symmetric(P);
                       branchorder = branchorder)
end
"""
    const HClE_HCl = Union{<:ClustersEstimator{<:Any, <:Any,
                                               <:AbstractHierarchicalClusteringAlgorithm,
                                               <:Any},
                           <:Clusters{<:Clustering.Hclust, <:Any, <:Any, <:Any},
                           <:NetworkClustersEstimator{<:Any,
                                                  <:AbstractHierarchicalClusteringAlgorithm}}

Alias for a hierarchical clustering estimator or result.

Matches either a [`ClustersEstimator`](@ref) parameterised with a hierarchical clustering algorithm, or a [`Clusters`](@ref) result wrapping a `Clustering.Hclust`. Used internally for dispatch in hierarchical clustering workflows.

# Related

  - [`ClustersEstimator`](@ref)
  - [`NetworkClustersEstimator`](@ref)
  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`Clusters`](@ref)
"""
const HClE_HCl = Union{<:ClustersEstimator{<:Any, <:Any,
                                           <:AbstractHierarchicalClusteringAlgorithm,
                                           <:Any},
                       <:Clusters{<:Clustering.Hclust, <:Any, <:Any, <:Any},
                       <:NetworkClustersEstimator{<:Any,
                                                  <:AbstractHierarchicalClusteringAlgorithm}}
"""
    _phylogeny_matrix(sep::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)
    _phylogeny_matrix(sep::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)

Internal dispatch helper carrying [`phylogeny_matrix`](@ref)'s per-separation body.

The neighbourhood [`phylogeny_matrix`](@ref) selects is a question about the separation, not about the estimator, so the split lives here rather than on the public method's argument. Dispatching on the estimator instead would pin the choice to `NetworkEstimator` and leave every other [`AbstractNetworkEstimator`](@ref) on one branch — and this family's other kernels, [`separation_matrix`](@ref) and [`separation_budget`](@ref), already take the separation first for the same reason.

# The two balls

  - [`HopCount`](@ref): the **hop ball**, `sum(A^i for i in 0:n)` clamped to `0` or `1`. `sep.n` is read directly as a **matrix-power count** rather than through [`separation_budget`](@ref), which is what makes it a power count and not a budget.
  - [`PathLength`](@ref): the **radius ball**, [`separation_matrix`](@ref) thresholded at [`separation_budget`](@ref). No second traversal.

# Arguments

  - `sep`: Separation algorithm, taken from `nte.sep` by the public method.
  - `nte`: Network estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix. `1` for a related pair, `0` otherwise, `0` on the diagonal.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`calc_adjacency`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
"""
function _phylogeny_matrix end
function _phylogeny_matrix(sep::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                           dims::Int = 1, kwargs...)
    A = calc_adjacency(nte, X; dims = dims, kwargs...)
    P = zeros(Int, size(A))
    # A matrix-power count, hence `sep.n` directly rather than `separation_budget`: this is
    # the hop branch, and a separation measuring anything else needs its own method.
    for i in 0:(sep.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - LinearAlgebra.I
    return P
end
function _phylogeny_matrix(sep::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                           dims::Int = 1, kwargs...)
    d = separation_matrix(sep, nte, X; dims = dims, kwargs...)
    dmax = separation_budget(sep, nte, d)
    # An unreachable pair carries `separation_matrix`'s `Inf` sentinel, so the same
    # comparison that applies the budget also rejects it. The diagonal is zero and therefore
    # always inside the budget, which `- I` then clears, matching the hop branch exactly.
    return Int.(d .<= dmax) - LinearAlgebra.I
end
"""
    phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the phylogeny matrix for a network estimator.

Builds the network from `X` and returns the binary matrix of the pairs `nte.sep` counts as related, with self-loops removed. Which neighbourhood that is comes from the separation, through [`_phylogeny_matrix`](@ref): [`HopCount`](@ref) gives the **hop ball**, the clamped power sum `sum(A^i for i in 0:n)` the network family has always used; [`PathLength`](@ref) gives the **radius ball**, [`separation_matrix`](@ref) thresholded at [`separation_budget`](@ref).

# The result is `Int` under either separation

Selection changes; the values do not. [`PhylogenyResult`](@ref)'s matrix is `Int` here as everywhere else, because no consumer of one wants a number: [`SemiDefinitePhylogeny`](@ref) is weight-inert (`A ⊙ W == 0` is the same constraint at any magnitude), [`IntegerPhylogeny`](@ref) counts an integer cardinality, and [`centrality_vector`](@ref) binarises before any centrality algorithm runs. The graded reading of a separation lives on [`Proximity`](@ref) instead.

# What the radius ball buys, measured

**It barely re-ranks, and on the PMFG not at all.** Compare a hop shell against the equal-cardinality prefix of the path-length ordering: on a 20-asset PMFG the two sets are **identical** at every shell — `0` pairs differ out of `54`, `121`, `165` and `186`. On the minimum spanning tree they are identical at the shells of `19` and `48`, and differ by `1`, `1`, `3` and `2` pairs at the shells of `84`, `115`, `144` and `170`. Both structures are selected by distance in the first place, so a path length **refines** a hop count rather than rivalling it. A reader who takes the radius ball for a conceptually different neighbourhood will be wrong.

What it buys is **intermediate cardinalities between the shells**. Over the same PMFG the hop knob relates `54`, then `121`, then `165` of the `190` pairs; a caller wanting about `100` cannot ask for it. Sweeping `dmax` across the same graph reaches `36`, `55`, `100`, `122`, `151` and `179`. That is the whole gain, and it is real for [`SemiDefinitePhylogeny`](@ref) and [`IntegerPhylogeny`](@ref), whose constraint strength is that cardinality.

# `PathLength`'s default budget relates everything reachable

`PathLength()` leaves `dmax = nothing`, which [`separation_budget`](@ref) resolves to the **observed diameter** — so no reachable pair sits outside it and the matrix is all ones off the diagonal. Measured: `190` of `190` pairs on both branches. This is the honest reading of an unstated budget rather than a fall-back, but it is the *opposite* end of the dial from [`HopCount`](@ref)'s default `n = 1`: a caller who swaps one separation for the other and changes nothing else gets the maximal ball where they had the minimal one. State a numeric `dmax` to select anything narrower.

# Arguments

  - `nte`: NetworkEstimator estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult{<:Matrix{Int}}`: Phylogeny matrix representing asset relationships. `1` for a related pair, `0` otherwise, `0` on the diagonal.

# Related

  - [`NetworkEstimator`](@ref)
  - [`_phylogeny_matrix`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`calc_adjacency`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
"""
function phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1,
                          kwargs...)
    return PhylogenyResult(; X = _phylogeny_matrix(nte.sep, nte, X; dims = dims, kwargs...))
end
"""
    phylogeny_matrix(cle::ClE_Cl,
                     X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1,
                     kwargs...)

Compute the phylogeny matrix for a clustering estimator or result.

This function clusterises the data, cuts the tree into the optimal number of clusters, and constructs a binary phylogeny matrix indicating shared cluster membership, with self-loops removed.

# Arguments

  - `cle`: Clustering estimator or result.
  - `X`: Data matrix (observations × assets).
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing cluster relationships.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`clusterise`](@ref)
"""
function phylogeny_matrix(cle::ClE_Cl, X::MatNum; branchorder::Symbol = :optimal,
                          dims::Int = 1, kwargs...)
    res = clusterise(cle, X; branchorder = branchorder, dims = dims, kwargs...)
    clusters = assignments(res)
    P = zeros(Int, size(X, 2), res.k)
    for i in axes(P, 2)
        idx = clusters .== i
        P[idx, i] .= one(eltype(P))
    end
    return PhylogenyResult(; X = P * transpose(P) - LinearAlgebra.I)
end
"""
    centrality_graph(pl::ClE_Cl, ct::AbstractCentralityAlgorithm, X::MatNum;
                     dims::Int = 1, kwargs...)
    centrality_graph(nte::AbstractNetworkEstimator, ct::AbstractCentralityAlgorithm,
                     X::MatNum; dims::Int = 1, kwargs...)
    centrality_graph(polarity::Option{<:AbstractCentralityPolarity},
                     nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Build the graph [`calc_centrality`](@ref) runs on, weighted in the polarity `ct` declares.

The one place where the source and the algorithm are both in scope, so it is the one place the pairing can be resolved. [`centrality_polarity`](@ref) says which quantity `ct` needs; the source says which quantities it has.

# The routing

| source                                              | polarity                     | graph                                                                              |
|:--------------------------------------------------- |:---------------------------- |:---------------------------------------------------------------------------------- |
| [`AbstractNetworkEstimator`](@ref)                  | [`DistancePolarity`](@ref)   | [`calc_distance_weighted_graph`](@ref) — distances, on either branch               |
| [`NetworkEstimator`](@ref) on the similarity branch | [`SimilarityPolarity`](@ref) | [`calc_weighted_adjacency_graph`](@ref) — the similarities that selected the edges |
| any source                                          | `nothing`                    | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                           |
| a clustering estimator or [`Clusters`](@ref)        | any                          | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                           |
| [`AbstractNetworkEstimator`](@ref) on a tree branch | [`SimilarityPolarity`](@ref) | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                           |

The similarity route is narrower than the distance route on purpose. [`calc_distance_weighted_graph`](@ref) carries distances on both branches, but only the similarity branch is *selected* by a similarity — a tree is selected by [`calc_mst`](@ref) minimising a distance, and manufacturing a similarity from it would weight the structure with a quantity that did not choose it.

# A partition carries no weights, and does not borrow any

A clustering source could reach a distance estimator through its own `de`, and does not. The triangulated maximally filtered graph selects each edge by a *pairwise* quantity, so a distance orders that selection; a partition selects by a dendrogram and a cut, and two assets in the same cluster may sit far apart in the distance. Co-membership is not ordered by the distance, so there is no quantity to borrow.

# The separation is read on the unweighted route only

The unweighted route goes through [`phylogeny_matrix`](@ref), so it sees the [`AbstractSeparationAlgorithm`](@ref) on the estimator — a [`HopCount`](@ref) of `n = 2` gives centrality on the two-hop closure. The weighted routes bypass it and read the structure itself, because a closure is built by summing matrix powers and a power of a weighted matrix sums *products* of distances, which is not a separation. So the `sep` field is **inert on the weighted routes**. At the default `HopCount(; n = 1)` there is nothing to notice: the closure of a graph at one hop is the graph.

# Two entry points, because the polarity is resolved once

The three-argument methods taking `ct` resolve [`centrality_polarity`](@ref) and forward to the methods taking the polarity itself, which is the same shape [`separation_matrix`](@ref) uses: the deciding algorithm comes first, and the estimator only supplies the graph.

# Arguments

  - $(field_dict[:pler])
  - $(field_dict[:cta])
  - `polarity`: Declared polarity of `ct`, from [`centrality_polarity`](@ref).
  - `nte`: Network estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `g::Graphs.AbstractGraph`: A `SimpleWeightedGraphs.SimpleWeightedGraph` on a weighted route, a `Graphs.SimpleGraph` otherwise.

# Related

  - [`centrality_polarity`](@ref)
  - [`AbstractCentralityPolarity`](@ref)
  - [`calc_centrality`](@ref)
  - [`centrality_vector`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function centrality_graph end
function centrality_graph(pl::ClE_Cl, ::AbstractCentralityAlgorithm, X::MatNum;
                          dims::Int = 1, kwargs...)
    return Graphs.SimpleGraph(phylogeny_matrix(pl, X; dims = dims, kwargs...).X)
end
function centrality_graph(nte::AbstractNetworkEstimator, ct::AbstractCentralityAlgorithm,
                          X::MatNum; dims::Int = 1, kwargs...)
    return centrality_graph(centrality_polarity(ct), nte, X; dims = dims, kwargs...)
end
function centrality_graph(::Nothing, nte::AbstractNetworkEstimator, X::MatNum;
                          dims::Int = 1, kwargs...)
    return Graphs.SimpleGraph(phylogeny_matrix(nte, X; dims = dims, kwargs...).X)
end
function centrality_graph(::DistancePolarity, nte::AbstractNetworkEstimator, X::MatNum;
                          dims::Int = 1, kwargs...)
    return calc_distance_weighted_graph(nte, X; dims = dims, kwargs...)
end
function centrality_graph(::SimilarityPolarity,
                          nte::NetworkEstimator{<:Any, <:Any,
                                                <:AbstractNonNegativeSimilarityMatrixAlgorithm},
                          X::MatNum; dims::Int = 1, kwargs...)
    return calc_weighted_adjacency_graph(nte, X; dims = dims, kwargs...)
end
function centrality_graph(::SimilarityPolarity, nte::AbstractNetworkEstimator, X::MatNum;
                          dims::Int = 1, kwargs...)
    # A tree is selected by minimising a distance, so it carries no similarity to read.
    return Graphs.SimpleGraph(phylogeny_matrix(nte, X; dims = dims, kwargs...).X)
end
"""
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      X::MatNum; dims::Int = 1, kwargs...)

Compute the centrality vector for a network and centrality algorithm.

This function builds the graph with [`centrality_graph`](@ref) — weighted in the polarity `ct` declares, where the source can supply it — and computes node centrality scores with [`calc_centrality`](@ref).

!!! warning

    Five cases run on the **unweighted** graph, and none of them raises. A caller names an algorithm and never asks for weights, so an unweightable pairing has not been handed a request it cannot serve.

     1. A clustering estimator, a precomputed [`Clusters`](@ref), or a precomputed [`PhylogenyResult`](@ref) as the source. A partition has no edge weights, and does not borrow any.
     2. [`DegreeCentrality`](@ref). `Graphs.jl` ignores weights.
     3. [`Pagerank`](@ref). `Graphs.jl` ignores weights.
     4. [`KatzCentrality`](@ref). `Graphs.katz_centrality` binarises through `adjacency_matrix(g, Bool)`.
     5. [`EigenvectorCentrality`](@ref) on a tree branch. The branch carries no similarity for it to read.

    On the weighted routes the estimator's `sep` field is **inert**: they read the structure itself rather than the separation closure [`phylogeny_matrix`](@ref) builds. At the default `HopCount(; n = 1)` the two agree, because the closure of a graph at one hop is the graph.

# Arguments

  - `pl`: Phylogeny estimator.
  - `ct`: Centrality algorithm.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::VecNum`: Centrality scores for each asset.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_graph`](@ref)
  - [`centrality_polarity`](@ref)
  - [`calc_centrality`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, X::MatNum;
                           dims::Int = 1, kwargs...)
    return PhylogenyResult(;
                           X = calc_centrality(ct,
                                               centrality_graph(pl, ct, X; dims = dims,
                                                                kwargs...)))
end
"""
    centrality_vector(cte::CentralityEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the centrality vector for a centrality estimator.

This function applies the centrality algorithm in the estimator to the network constructed from the data.

# Arguments

  - `cte`: Centrality estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::VecNum`: Centrality scores for each asset.

# Related

  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::CentralityEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return centrality_vector(cte.pl, cte.ct, X; dims = dims, kwargs...)
end
"""
    average_centrality(pl::NwE_Pl_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum, X::MatNum;
                       dims::Int = 1, kwargs...)

Compute the weighted average centrality for a network and centrality algorithm.

This function computes the centrality vector and returns the weighted average using the provided weights.

# Arguments

  - `pl`: NetworkEstimator estimator.
  - `ct`: Centrality algorithm.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
"""
function average_centrality(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            X::MatNum; dims::Int = 1, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, X; dims = dims, kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, X::MatNum;
                       dims::Int = 1, kwargs...)

Compute the weighted average centrality for a centrality estimator.

This function applies the centrality algorithm in the estimator to the network and returns the weighted average using the provided weights.

# Arguments

  - `cte`: Centrality estimator.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::VecNum, X::MatNum; dims::Int = 1,
                            kwargs...)
    return average_centrality(cte.pl, cte.ct, w, X; dims = dims, kwargs...)
end
"""
    asset_phylogeny(w::VecNum, X::MatNum)

Compute the asset phylogeny score for a set of weights and a phylogeny matrix.

This function computes the weighted sum of the phylogeny matrix, normalised by the sum of absolute weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Arguments

  - `w`: Weights vector.
  - `X`: Phylogeny matrix.

# Returns

  - `p::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(w::VecNum, X::MatNum)
    aw = abs.(w * transpose(w))
    c = LinearAlgebra.dot(X, aw)
    c /= sum(aw)
    return c
end
"""
    asset_phylogeny(pl::PhylogenyResult{<:MatNum}, w::VecNum, args...;
                    kwargs...)

Compute the asset phylogeny score for a set of portfolio weights and a phylogeny matrix result, forwarding additional arguments.

This method provides compatibility with workflows that pass extra positional or keyword arguments. It extracts the phylogeny matrix from the `PhylogenyResult` and delegates to `asset_phylogeny(w, pl)`, ignoring any additional arguments.

# Arguments

  - `pl::PhylogenyResult{<:MatNum}`: Phylogeny matrix result object.
  - `w::VecNum`: Portfolio weights vector.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `score::Number`: Asset phylogeny score.

# Related

  - [`PhylogenyResult`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(pl::PhylogenyResult{<:MatNum}, w::VecNum, args...; kwargs...)
    return asset_phylogeny(w, pl.X)
end
"""
    asset_phylogeny(cle::NwE_ClE_Cl,
                    w::VecNum, X::MatNum; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a set of weights and a network or clustering estimator.

This function computes the phylogeny matrix using the estimator and data, then computes the asset phylogeny score using the weights.

# Arguments

  - `cle`: NetworkEstimator or clustering estimator.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `p::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(cle::NwE_ClE_Cl, w::VecNum, X::MatNum; dims::Int = 1, kwargs...)
    return asset_phylogeny(phylogeny_matrix(cle, X; dims = dims, kwargs...), w)
end

export PhylogenyResult, BetweennessCentrality, ClosenessCentrality, DegreeCentrality,
       EigenvectorCentrality, KatzCentrality, Pagerank, RadialityCentrality,
       StressCentrality, KruskalTree, BoruvkaTree, PrimTree, NetworkEstimator,
       phylogeny_matrix, average_centrality, asset_phylogeny, AbstractCentralityAlgorithm,
       CentralityEstimator, centrality_vector, NetworkClustersEstimator, separation_matrix,
       separation_budget, AbstractCentralityPolarity, DistancePolarity, SimilarityPolarity,
       centrality_polarity
