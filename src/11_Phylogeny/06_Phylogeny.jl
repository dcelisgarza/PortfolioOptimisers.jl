"""
$(DocStringExtensions.TYPEDEF)

Container type for phylogeny matrix or vector results in `PortfolioOptimisers.jl`.

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
            @argcheck(LinearAlgebra.issymmetric(X))
            @argcheck(all(iszero, LinearAlgebra.diag(X)))
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

# Related

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
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the centrality vector from a matrix [`PhylogenyResult`](@ref) using the specified centrality algorithm.

Builds a graph from the phylogeny matrix and applies `ct` to compute node centrality scores.

# Related

  - [`PhylogenyResult`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
  - [`calc_centrality`](@ref)
"""
function centrality_vector(plr::PhylogenyResult{<:MatNum}, ct::AbstractCentralityAlgorithm,
                           args...; kwargs...)
    G = Graphs.SimpleGraph(plr.X)
    return PhylogenyResult(; X = calc_centrality(ct, G))
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for betweenness centrality in `PortfolioOptimisers.jl`.

`BetweennessCentrality` computes the [betweenness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality) of nodes in a graph, measuring the extent to which a node lies on shortest paths between other nodes.

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
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BetweennessCentrality(; args::Tuple = (),
                               kwargs::NamedTuple = (;))::BetweennessCentrality
    return BetweennessCentrality(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for closeness centrality in `PortfolioOptimisers.jl`.

`ClosenessCentrality` computes the [closeness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality) of nodes in a graph, measuring how close a node is to all other nodes.

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
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function ClosenessCentrality(; args::Tuple = (),
                             kwargs::NamedTuple = (;))::ClosenessCentrality
    return ClosenessCentrality(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for degree centrality in `PortfolioOptimisers.jl`.

`DegreeCentrality` computes the [degree centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D) of nodes in a graph, measuring the number of edges connected to each node. The `kind` parameter specifies the type of degree (0: total, 1: in-degree, 2: out-degree).

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
        @argcheck(kind in 0:2)
        return new{typeof(kind), typeof(kwargs)}(kind, kwargs)
    end
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))::DegreeCentrality
    return DegreeCentrality(kind, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for [eigenvector centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D) in `PortfolioOptimisers.jl`.

`EigenvectorCentrality` computes the eigenvector centrality of nodes in a graph, measuring the influence of a node based on the centrality of its neighbors.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`Graphs.eigenvector_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D)
"""
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for Katz centrality in `PortfolioOptimisers.jl`.

`KatzCentrality` computes the [Katz centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality) of nodes in a graph, measuring the influence of a node based on the number and length of walks between nodes, controlled by the attenuation factor `alpha`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KatzCentrality(;
        alpha::Number = 0.3
    ) -> KatzCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> KatzCentrality(; alpha = 0.5)
KatzCentrality
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`Graphs.katz_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality)
"""
@concrete struct KatzCentrality <: AbstractCentralityAlgorithm
    """
    Attenuation factor for Katz centrality.
    """
    alpha
    function KatzCentrality(alpha::Number)
        return new{typeof(alpha)}(alpha)
    end
end
function KatzCentrality(; alpha::Number = 0.3)::KatzCentrality
    return KatzCentrality(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for PageRank in `PortfolioOptimisers.jl`.

`Pagerank` computes the [PageRank](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer) of nodes in a graph, measuring the importance of nodes based on the structure of incoming links. The algorithm is controlled by the damping factor `alpha`, number of iterations `n`, and convergence tolerance `epsilon`.

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
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
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

Centrality algorithm type for [radiality centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D) in `PortfolioOptimisers.jl`.

`RadialityCentrality` computes the radiality centrality of nodes in a graph, measuring how close a node is to all other nodes, adjusted for the maximum possible distance.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`Graphs.radiality_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D)
"""
struct RadialityCentrality <: AbstractCentralityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centrality algorithm type for [stress centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.stress_centrality) in `PortfolioOptimisers.jl`.

`StressCentrality` computes the stress centrality of nodes in a graph, measuring the number of shortest paths passing through each node.

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
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))::StressCentrality
    return StressCentrality(args, kwargs)
end
"""
    calc_centrality(ct::AbstractCentralityAlgorithm, g::Graphs.AbstractGraph)

Compute node centrality scores for a graph using the specified centrality algorithm.

This function dispatches to the appropriate centrality computation from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/) based on the type of `ct`. Supported algorithms include betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, and stress centrality.

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

Abstract supertype for all minimum spanning tree (MST) algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific MST algorithms (e.g., Kruskal, Boruvka, Prim) should be subtypes of `AbstractTreeType`.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
"""
    const Tree_SimMat = Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType}

Alias for a tree or similarity matrix algorithm.

Matches either an [`AbstractSimilarityMatrixAlgorithm`](@ref) or an [`AbstractTreeType`](@ref). Used for dispatch in phylogeny estimation where either a spanning tree or a similarity matrix approach may be used.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`NetworkEstimator`](@ref)
"""
const Tree_SimMat = Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm type for Kruskal's minimum spanning tree (MST) in `PortfolioOptimisers.jl`.

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

Algorithm type for Boruvka's minimum spanning tree (MST) in `PortfolioOptimisers.jl`.

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

Algorithm type for Prim's minimum spanning tree (MST) in `PortfolioOptimisers.jl`.

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

Abstract supertype for all network estimator types in `PortfolioOptimisers.jl`.

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
    const NwE_PlM_ClE_Cl = Union{<:AbstractNetworkEstimator,
                                 <:PhylogenyResult{<:AbstractMatrix}, <:ClE_Cl}

Alias for a network estimator, matrix-phylogeny result, or clustering estimator/result.

Similar to [`NwE_Pl_ClE_Cl`](@ref) but restricts `PhylogenyResult` to those wrapping an `AbstractMatrix`.

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`ClE_Cl`](@ref)
"""
const NwE_PlM_ClE_Cl = Union{<:AbstractNetworkEstimator,
                             <:PhylogenyResult{<:AbstractMatrix}, <:ClE_Cl}
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

Estimator type for network-based phylogeny analysis in `PortfolioOptimisers.jl`.

`NetworkEstimator` encapsulates the configuration for constructing a network from asset data, including the covariance estimator, distance estimator, tree or similarity algorithm, and the network depth parameter.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Tree_SimMat = KruskalTree(),
        n::Integer = 1
    ) -> NetworkEstimator

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ntn])

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
      │      │   alg ┴ Full()
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
    n ┴ Int64: 1
```

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
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
    alg
    """
    $(field_dict[:ntn])
    """
    n
    function NetworkEstimator(ce::StatsBase.CovarianceEstimator,
                              de::AbstractDistanceEstimator, alg::Tree_SimMat, n::Integer)
        @argcheck(n >= one(n))
        return new{typeof(ce), typeof(de), typeof(alg), typeof(n)}(ce, de, alg, n)
    end
end
function NetworkEstimator(;
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          de::AbstractDistanceEstimator = Distance(;
                                                                   alg = CanonicalDistance()),
                          alg::Tree_SimMat = KruskalTree(), n::Integer = 1)
    return NetworkEstimator(ce, de, alg, n)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for network-based phylogeny analysis in `PortfolioOptimisers.jl`.

`NetworkEstimator` encapsulates the configuration for constructing a network from asset data, including the covariance estimator, distance estimator, tree or similarity algorithm, and the network depth parameter.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NetworkEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Tree_SimMat = KruskalTree(),
        n::Integer = 1
    ) -> NetworkEstimator

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ntn])

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
      │       │      │   alg ┴ Full()
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
      │     n ┴ Int64: 1
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
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
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

Abstract supertype for all centrality estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing centrality-based estimation algorithms should be subtypes of `AbstractCentralityEstimator`.

# Related

  - [`CentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
abstract type AbstractCentralityEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Estimator type for centrality-based analysis in `PortfolioOptimisers.jl`.

`CentralityEstimator` encapsulates the configuration for computing centrality measures on a network, including the network estimator and the centrality algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CentralityEstimator(;
        pl::NwE_Pl_ClE_Cl = NetworkEstimator(),
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
     │       │      │   alg ┴ Full()
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
     │     n ┴ Int64: 1
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
    function CentralityEstimator(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm)
        return new{typeof(pl), typeof(ct)}(pl, ct)
    end
end
function CentralityEstimator(; pl::NwE_Pl_ClE_Cl = NetworkEstimator(),
                             ct::AbstractCentralityAlgorithm = DegreeCentrality())
    return CentralityEstimator(pl, ct)
end
"""
    calc_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the adjacency matrix for a network estimator.

# Arguments

  - `nte`: NetworkEstimator estimator.

      + `nte::NetworkEstimator{<:Any, <:Any, <:AbstractTreeType, <:Any}`: Constructs a weighted graph from the distance matrix and computes the minimum spanning tree, returning the adjacency matrix of the resulting graph.
      + `nte::NetworkEstimator{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm, <:Any}`: Computes the similarity and distance matrices, applies the [`PMFG_T2s`](@ref) algorithm, and returns the adjacency matrix of the resulting graph..

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::Matrix{Int}`: Adjacency matrix representing the network.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
"""
function calc_adjacency(nte::NetworkEstimator{<:Any, <:Any, <:AbstractTreeType}, X::MatNum;
                        dims::Int = 1, kwargs...)
    D = distance(nte.de, nte.ce, X; dims = dims, kwargs...)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(D)
    tree = calc_mst(nte.alg, G)
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(G[tree]))
end
function calc_adjacency(nte::NetworkEstimator{<:Any, <:Any,
                                              <:AbstractSimilarityMatrixAlgorithm},
                        X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(nte.de, nte.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(nte.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(Rpm))
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
                                                                <:AbstractTreeType, <:Any}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster assets using a minimum spanning tree (MST) network structure and return a [`Clusters`](@ref) result.

Builds the MST from the distance matrix, accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

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
  - [`calc_mst`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractTreeType,
                                                                     <:Any}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    P = zeros(eltype(D), size(D))
    G = SimpleWeightedGraphs.SimpleWeightedGraph(D)
    tree = calc_mst(nte.nte.alg, G)
    A = Graphs.adjacency_matrix(G[tree])
    for i in 0:(nte.nte.n)
        P .+= D^i - A^i
    end
    P .-= LinearAlgebra.Diagonal(P)
    return _clusterise(nte.alg, nte.onc, S, D, LinearAlgebra.Symmetric(P);
                       branchorder = branchorder)
end
"""
    clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                <:AbstractSimilarityMatrixAlgorithm,
                                                                <:Any}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster assets using a Planar Maximally Filtered Graph (PMFG) network structure and return a [`Clusters`](@ref) result.

Builds the PMFG from the similarity matrix via [`PMFG_T2s`](@ref), accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{S}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

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
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`Clusters`](@ref)
  - [`_clusterise`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`dbht_similarity`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractSimilarityMatrixAlgorithm,
                                                                     <:Any}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    P = zeros(eltype(D), size(D))
    S = dbht_similarity(nte.nte.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    for i in 0:(nte.nte.n)
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
    phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the phylogeny matrix for a network estimator.

This function constructs the adjacency matrix for the network, then computes the phylogeny matrix by summing powers of the adjacency matrix up to the network depth parameter `n`, clamping values to 0 or 1, and removing self-loops.

# Arguments

  - `nte`: NetworkEstimator estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing asset relationships.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_adjacency`](@ref)
"""
function phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1,
                          kwargs...)
    A = calc_adjacency(nte, X; dims = dims, kwargs...)
    P = zeros(Int, size(A))
    for i in 0:(nte.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - LinearAlgebra.I
    return PhylogenyResult(; X = P)
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
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      X::MatNum; dims::Int = 1, kwargs...)

Compute the centrality vector for a network and centrality algorithm.

This function constructs the phylogeny matrix for the network, builds a graph, and computes node centrality scores using the specified centrality algorithm.

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
  - [`calc_centrality`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, X::MatNum;
                           dims::Int = 1, kwargs...)
    return centrality_vector(phylogeny_matrix(pl, X; dims = dims, kwargs...), ct;
                             dims = dims, kwargs...)
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
       CentralityEstimator, centrality_vector, NetworkClustersEstimator
