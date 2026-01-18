"""
    struct PhylogenyResult{T} <: AbstractPhylogenyResult
        X::T
    end

Container type for phylogeny matrix or vector results in PortfolioOptimisers.jl.

`PhylogenyResult` stores the output of phylogeny-based estimation routines, such as network or clustering-based phylogeny matrices, or centrality vectors. It is used throughout the package to represent validated phylogeny structures for constraint generation, centrality analysis, and related workflows.

# Fields

  - `X`: The phylogeny matrix or centrality vector.

# Constructor

    PhylogenyResult(; X::ArrNum)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(X)`.

  - If `X` is a `MatNum`:

      + Must be symmetric, `LinearAlgebra.issymmetric(X) == true`.
      + Must have zero diagonal, `all(iszero, LinearAlgebra.diag(X)) == true`.

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
struct PhylogenyResult{T} <: AbstractPhylogenyResult
    X::T
    function PhylogenyResult(X::ArrNum)
        @argcheck(!isempty(X), IsEmptyError)
        if isa(X, MatNum)
            @argcheck(LinearAlgebra.issymmetric(X))
            @argcheck(all(iszero, LinearAlgebra.diag(X)))
        end
        return new{typeof(X)}(X)
    end
end
function PhylogenyResult(; X::ArrNum)
    return PhylogenyResult(X)
end
"""
    phylogeny_matrix(pl::PhylogenyResult{<:MatNum}, args...; kwargs...)

Fallback no-op for returning a validated phylogeny matrix result as-is.

This method provides a generic interface for handling precomputed phylogeny matrices wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for constraint generation and analysis.

# Arguments

  - `pl::PhylogenyResult{<:MatNum}`: Phylogeny matrix result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - The input `pl` object.

# Examples

```jldoctest
julia> pl = PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0]);

julia> phylogeny_matrix(pl)
PhylogenyResult
  X ┴ 3×3 Matrix{Int64}
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(pl::PhylogenyResult{<:MatNum}, args...; kwargs...)
    return pl
end
"""
    centrality_vector(pl::PhylogenyResult{<:VecNum}, args...; kwargs...)

Fallback no-op for returning a validated centrality vector result as-is.

This method provides a generic interface for handling precomputed centrality vectors wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for centrality-based analysis and constraint generation.

# Arguments

  - `pl::PhylogenyResult{<:VecNum}`: Centrality vector result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - The input `pl` object.

# Examples

```jldoctest
julia> pl = PhylogenyResult(; X = [0.2, 0.5, 0.3]);

julia> centrality_vector(pl)
PhylogenyResult
  X ┴ Vector{Float64}: [0.2, 0.5, 0.3]
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(pl::PhylogenyResult{<:VecNum}, args...; kwargs...)
    return pl
end
"""
    abstract type AbstractCentralityAlgorithm <: AbstractPhylogenyAlgorithm end

Abstract supertype for all centrality algorithm types in PortfolioOptimisers.jl from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/).

All concrete types implementing specific centrality algorithms (e.g., betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, stress) should subtype `AbstractCentralityAlgorithm`. This enables flexible extension and dispatch of centrality routines for network and phylogeny analysis.

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

function centrality_vector(pl::PhylogenyResult{<:MatNum}, ct::AbstractCentralityAlgorithm,
                           args...; kwargs...)
    G = Graphs.SimpleGraph(pl.X)
    return PhylogenyResult(; X = calc_centrality(ct, G))
end
"""
    struct BetweennessCentrality{T1, T2} <: AbstractCentralityAlgorithm
        args::T1
        kwargs::T2
    end

Centrality algorithm type for betweenness centrality in PortfolioOptimisers.jl.

`BetweennessCentrality` computes the [betweenness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality) of nodes in a graph, measuring the extent to which a node lies on shortest paths between other nodes.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

    BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct BetweennessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
    function BetweennessCentrality(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality(args, kwargs)
end
"""
    struct ClosenessCentrality{T1, T2} <: AbstractCentralityAlgorithm
        args::T1
        kwargs::T2
    end

Centrality algorithm type for closeness centrality in PortfolioOptimisers.jl.

`ClosenessCentrality` computes the [closeness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality) of nodes in a graph, measuring how close a node is to all other nodes.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

    ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct ClosenessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
    function ClosenessCentrality(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality(args, kwargs)
end
"""
    struct DegreeCentrality{T1, T2} <: AbstractCentralityAlgorithm
        kind::T1
        kwargs::T2
    end

Centrality algorithm type for degree centrality in PortfolioOptimisers.jl.

`DegreeCentrality` computes the [degree centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D) of nodes in a graph, measuring the number of edges connected to each node. The `kind` parameter specifies the type of degree (0: total, 1: in-degree, 2: out-degree).

# Fields

  - `kind`: Degree type (0: total, 1: in-degree, 2: out-degree).
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

    DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct DegreeCentrality{T1, T2} <: AbstractCentralityAlgorithm
    kind::T1
    kwargs::T2
    function DegreeCentrality(kind::Integer, kwargs::NamedTuple)
        @argcheck(kind in 0:2)
        return new{typeof(kind), typeof(kwargs)}(kind, kwargs)
    end
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
    return DegreeCentrality(kind, kwargs)
end
"""
    struct EigenvectorCentrality <: AbstractCentralityAlgorithm end

Centrality algorithm type for [eigenvector centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D) in PortfolioOptimisers.jl.

`EigenvectorCentrality` computes the eigenvector centrality of nodes in a graph, measuring the influence of a node based on the centrality of its neighbors.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`Graphs.eigenvector_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D)
"""
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
"""
    struct KatzCentrality{T1} <: AbstractCentralityAlgorithm
        alpha::T1
    end

Centrality algorithm type for Katz centrality in PortfolioOptimisers.jl.

`KatzCentrality` computes the [Katz centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality) of nodes in a graph, measuring the influence of a node based on the number and length of walks between nodes, controlled by the attenuation factor `alpha`.

# Fields

  - `alpha`: Attenuation factor for Katz centrality.

# Constructor

    KatzCentrality(; alpha::Number = 0.3)

Keyword arguments correspond to the fields above.

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
struct KatzCentrality{T1} <: AbstractCentralityAlgorithm
    alpha::T1
    function KatzCentrality(alpha::Number)
        return new{typeof(alpha)}(alpha)
    end
end
function KatzCentrality(; alpha::Number = 0.3)
    return KatzCentrality(alpha)
end
"""
    struct Pagerank{T1, T2, T3} <: AbstractCentralityAlgorithm
        n::T1
        alpha::T2
        epsilon::T3
    end

Centrality algorithm type for PageRank in PortfolioOptimisers.jl.

`Pagerank` computes the [PageRank](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer) of nodes in a graph, measuring the importance of nodes based on the structure of incoming links. The algorithm is controlled by the damping factor `alpha`, number of iterations `n`, and convergence tolerance `epsilon`.

# Fields

  - `n`: Number of iterations (must be > 0).
  - `alpha`: Damping factor (must be in (0, 1)).
  - `epsilon`: Convergence tolerance (must be > 0).

# Constructor

    Pagerank(; alpha::Number = 0.85, n::Integer = 100, epsilon::Number = 1e-6)

Keyword arguments correspond to the fields above.

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
struct Pagerank{T1, T2, T3} <: AbstractCentralityAlgorithm
    n::T1
    alpha::T2
    epsilon::T3
    function Pagerank(n::Integer, alpha::Number, epsilon::Number)
        @argcheck(0 < n, DomainError)
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
        @argcheck(zero(epsilon) < epsilon, DomainError)
        return new{typeof(n), typeof(alpha), typeof(epsilon)}(n, alpha, epsilon)
    end
end
function Pagerank(; n::Integer = 100, alpha::Number = 0.85, epsilon::Number = 1e-6)
    return Pagerank(n, alpha, epsilon)
end
"""
    struct RadialityCentrality <: AbstractCentralityAlgorithm end

Centrality algorithm type for [radiality centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D) in PortfolioOptimisers.jl.

`RadialityCentrality` computes the radiality centrality of nodes in a graph, measuring how close a node is to all other nodes, adjusted for the maximum possible distance.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`Graphs.radiality_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D)
"""
struct RadialityCentrality <: AbstractCentralityAlgorithm end
"""
    struct StressCentrality{T1, T2} <: AbstractCentralityAlgorithm
        args::T1
        kwargs::T2
    end

Centrality algorithm type for [stress centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.stress_centrality) in PortfolioOptimisers.jl.

`StressCentrality` computes the stress centrality of nodes in a graph, measuring the number of shortest paths passing through each node.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

    StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct StressCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
    function StressCentrality(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
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
    abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end

Abstract supertype for all minimum spanning tree (MST) algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific MST algorithms (e.g., Kruskal, Boruvka, Prim) should subtype `AbstractTreeType`. This enables flexible extension and dispatch of tree-based routines for network and phylogeny analysis.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
const Tree_SimMat = Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType}
"""
    struct KruskalTree{T1, T2} <: AbstractTreeType
        args::T1
        kwargs::T2
    end

Algorithm type for Kruskal's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`KruskalTree` specifies the use of [Kruskal's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst) for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

    KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct KruskalTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
    function KruskalTree(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree(args, kwargs)
end
"""
    struct BoruvkaTree{T1, T2} <: AbstractTreeType
        args::T1
        kwargs::T2
    end

Algorithm type for Boruvka's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`BoruvkaTree` specifies the use of [Boruvka's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst) for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

    BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct BoruvkaTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
    function BoruvkaTree(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree(args, kwargs)
end
"""
    struct PrimTree{T1, T2} <: AbstractTreeType
        args::T1
        kwargs::T2
    end

Algorithm type for Prim's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`PrimTree` specifies the use of [Prim's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst) for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

    PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

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
struct PrimTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
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
    abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end

Abstract supertype for all network estimator types in PortfolioOptimisers.jl.

All concrete types implementing network-based estimation algorithms should subtype `AbstractNetworkEstimator`.

# Related

  - [`NetworkEstimator`](@ref)
  - [`AbstractCentralityEstimator`](@ref)
"""
abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end
const NwE_Ph_ClE_Cl = Union{<:AbstractNetworkEstimator, <:PhylogenyResult, <:ClE_Cl}
const NwE_ClE_Cl = Union{<:AbstractNetworkEstimator, <:ClE_Cl}
"""
    struct NetworkEstimator{T1, T2, T3, T4} <: AbstractNetworkEstimator
        ce::T1
        de::T2
        alg::T3
        n::T4
    end

Estimator type for network-based phylogeny analysis in PortfolioOptimisers.jl.

`NetworkEstimator` encapsulates the configuration for constructing a network from asset data, including the covariance estimator, distance estimator, tree or similarity algorithm, and the network depth parameter.

# Fields

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Tree or similarity matrix algorithm.
  - `n`: NetworkEstimator depth parameter.

# Constructor

    NetworkEstimator(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                     de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
                     alg::Tree_SimMat = KruskalTree(),
                     n::Integer = 1)

Keyword arguments correspond to the fields above.

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
struct NetworkEstimator{T1, T2, T3, T4} <: AbstractNetworkEstimator
    ce::T1
    de::T2
    alg::T3
    n::T4
    function NetworkEstimator(ce::StatsBase.CovarianceEstimator,
                              de::AbstractDistanceEstimator, alg::Tree_SimMat, n::Integer)
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
    abstract type AbstractCentralityEstimator <: AbstractEstimator end

Abstract supertype for all centrality estimator types in PortfolioOptimisers.jl.

All concrete types implementing centrality-based estimation algorithms should subtype `AbstractCentralityEstimator`.

# Related

  - [`CentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
abstract type AbstractCentralityEstimator <: AbstractEstimator end
"""
    struct CentralityEstimator{T1, T2} <: AbstractCentralityEstimator
        pl::T1
        ct::T2
    end

Estimator type for centrality-based analysis in PortfolioOptimisers.jl.

`CentralityEstimator` encapsulates the configuration for computing centrality measures on a network, including the network estimator and the centrality algorithm.

# Fields

  - `pl`: NetworkEstimator estimator.
  - `ct`: Centrality algorithm.

# Constructor

    CentralityEstimator(;
                        pl::NwE_Ph_ClE_Cl = NetworkEstimator(),
                        ct::AbstractCentralityAlgorithm = DegreeCentrality())

Keyword arguments correspond to the fields above.

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
     │       │   mp ┼ DenoiseDetoneAlgMatrixProcessing
     │       │      │     pdm ┼ Posdef
     │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │       │      │      dn ┼ nothing
     │       │      │      dt ┼ nothing
     │       │      │     alg ┼ nothing
     │       │      │   order ┴ DenoiseDetoneAlg()
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
struct CentralityEstimator{T1, T2} <: AbstractCentralityEstimator
    pl::T1
    ct::T2
    function CentralityEstimator(pl::NwE_Ph_ClE_Cl, ct::AbstractCentralityAlgorithm)
        return new{typeof(pl), typeof(ct)}(pl, ct)
    end
end
function CentralityEstimator(; pl::NwE_Ph_ClE_Cl = NetworkEstimator(),
                             ct::AbstractCentralityAlgorithm = DegreeCentrality())
    return CentralityEstimator(pl, ct)
end
"""
    calc_adjacency(pl::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the adjacency matrix for a network estimator.

# Arguments

  - `pl`: NetworkEstimator estimator.

      + `pl::NetworkEstimator{<:Any, <:Any, <:AbstractTreeType, <:Any}`: Constructs a weighted graph from the distance matrix and computes the minimum spanning tree, returning the adjacency matrix of the resulting graph.
      + `pl::NetworkEstimator{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm, <:Any}`: Computes the similarity and distance matrices, applies the [`PMFG_T2s`](@ref) algorithm, and returns the adjacency matrix of the resulting graph..

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::Matrix{Int}`: Adjacency matrix representing the network.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
"""
function calc_adjacency(pl::NetworkEstimator{<:Any, <:Any, <:AbstractTreeType, <:Any},
                        X::MatNum; dims::Int = 1, kwargs...)
    D = distance(pl.de, pl.ce, X; dims = dims, kwargs...)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(D)
    tree = calc_mst(pl.alg, G)
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(G[tree]))
end
function calc_adjacency(pl::NetworkEstimator{<:Any, <:Any,
                                             <:AbstractSimilarityMatrixAlgorithm, <:Any},
                        X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(pl.de, pl.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(pl.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(Rpm))
end
"""
    phylogeny_matrix(pl::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the phylogeny matrix for a network estimator.

This function constructs the adjacency matrix for the network, then computes the phylogeny matrix by summing powers of the adjacency matrix up to the network depth parameter `n`, clamping values to 0 or 1, and removing self-loops.

# Arguments

  - `pl`: NetworkEstimator estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing asset relationships.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_adjacency`](@ref)
"""
function phylogeny_matrix(pl::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)
    A = calc_adjacency(pl, X; dims = dims, kwargs...)
    P = zeros(Int, size(Matrix(A)))
    for i in 0:(pl.n)
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
  - `dims`: Dimension along which to compute.
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
    clusters = get_clustering_indices(res)
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
  - `dims`: Dimension along which to compute.
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
    P = phylogeny_matrix(pl, X; dims = dims, kwargs...)
    return centrality_vector(P, ct; dims = dims, kwargs...)
end
"""
    centrality_vector(cte::CentralityEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the centrality vector for a centrality estimator.

This function applies the centrality algorithm in the estimator to the network constructed from the data.

# Arguments

  - `cte`: Centrality estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute.
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
    average_centrality(pl::NwE_Ph_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum, X::MatNum;
                       dims::Int = 1, kwargs...)

Compute the weighted average centrality for a network and centrality algorithm.

This function computes the centrality vector and returns the weighted average using the provided weights.

# Arguments

  - `pl`: NetworkEstimator estimator.
  - `ct`: Centrality algorithm.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
"""
function average_centrality(pl::NwE_Ph_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
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
  - `dims`: Dimension along which to compute.
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
  - `dims`: Dimension along which to compute.
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
       CentralityEstimator, centrality_vector
