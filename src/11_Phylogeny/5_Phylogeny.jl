"""
```julia
abstract type AbstractCentralityAlgorithm <: AbstractPhylogenyAlgorithm end
```

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

"""
```julia
struct BetweennessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
```

Centrality algorithm type for betweenness centrality in PortfolioOptimisers.jl.

`BetweennessCentrality` computes the betweenness centrality of nodes in a graph, measuring the extent to which a node lies on shortest paths between other nodes.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

```julia
BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct BetweennessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
"""
```julia
BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`BetweennessCentrality`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Returns

  - `BetweennessCentrality`: Algorithm object for betweenness centrality.

# Examples

```jldoctest
julia> BetweennessCentrality()
BetweennessCentrality
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality(args, kwargs)
end

"""
```julia
struct ClosenessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
```

Centrality algorithm type for closeness centrality in PortfolioOptimisers.jl.

`ClosenessCentrality` computes the closeness centrality of nodes in a graph, measuring how close a node is to all other nodes.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

```julia
ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct ClosenessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
"""
```julia
ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`ClosenessCentrality`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

```julia
ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Returns

  - `ClosenessCentrality`: Algorithm object for closeness centrality.

# Examples

```jldoctest
julia> ClosenessCentrality()
ClosenessCentrality
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality(args, kwargs)
end

"""
```julia
struct DegreeCentrality{T1, T2} <: AbstractCentralityAlgorithm
    kind::T1
    kwargs::T2
end
```

Centrality algorithm type for degree centrality in PortfolioOptimisers.jl.

`DegreeCentrality` computes the degree centrality of nodes in a graph, measuring the number of edges connected to each node. The `kind` parameter specifies the type of degree (0: total, 1: in-degree, 2: out-degree).

# Fields

  - `kind`: Degree type (0: total, 1: in-degree, 2: out-degree).
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

```julia
DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct DegreeCentrality{T1, T2} <: AbstractCentralityAlgorithm
    kind::T1
    kwargs::T2
end
"""
```julia
DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
```

Construct a [`DegreeCentrality`](@ref) algorithm.

# Arguments

  - `kind`: Degree type (0: total, 1: in-degree, 2: out-degree).
  - `kwargs`: Keyword arguments for the centrality computation.

# Returns

  - `DegreeCentrality`: Algorithm object for degree centrality.

# Validation

  - `0 <= kind <= 2`.

# Examples

```jldoctest
julia> DegreeCentrality(; kind = 1)
DegreeCentrality
    kind | Int64: 1
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
    @argcheck(kind in 0:2, DomainError("`kind` must be in (0:2):\nkind => $kind"))
    return DegreeCentrality(kind, kwargs)
end

"""
```julia
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
```

Centrality algorithm type for eigenvector centrality in PortfolioOptimisers.jl.

`EigenvectorCentrality` computes the eigenvector centrality of nodes in a graph, measuring the influence of a node based on the centrality of its neighbors.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end

"""
```julia
struct KatzCentrality{T1} <: AbstractCentralityAlgorithm
    alpha::T1
end
```

Centrality algorithm type for Katz centrality in PortfolioOptimisers.jl.

`KatzCentrality` computes the Katz centrality of nodes in a graph, measuring the influence of a node based on the number and length of walks between nodes, controlled by the attenuation factor `alpha`.

# Fields

  - `alpha`: Attenuation factor for Katz centrality.

# Constructor

```julia
KatzCentrality(; alpha::Real = 0.3)
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct KatzCentrality{T1} <: AbstractCentralityAlgorithm
    alpha::T1
end
"""
```julia
KatzCentrality(; alpha::Real = 0.3)
```

Construct a [`KatzCentrality`](@ref) algorithm.

# Arguments

  - `alpha`: Attenuation factor for Katz centrality.

# Returns

  - `KatzCentrality`: Algorithm object for Katz centrality.

# Examples

```jldoctest
julia> KatzCentrality(; alpha = 0.5)
KatzCentrality
  alpha | Float64: 0.5
```
"""
function KatzCentrality(; alpha::Real = 0.3)
    return KatzCentrality(alpha)
end

"""
```julia
struct Pagerank{T1, T2, T3} <: AbstractCentralityAlgorithm
    n::T1
    alpha::T2
    epsilon::T3
end
```

Centrality algorithm type for PageRank in PortfolioOptimisers.jl.

`Pagerank` computes the PageRank of nodes in a graph, measuring the importance of nodes based on the structure of incoming links. The algorithm is controlled by the damping factor `alpha`, number of iterations `n`, and convergence tolerance `epsilon`.

# Fields

  - `n`: Number of iterations (must be > 0).
  - `alpha`: Damping factor (must be in (0, 1)).
  - `epsilon`: Convergence tolerance (must be > 0).

# Constructor

```julia
Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct Pagerank{T1, T2, T3} <: AbstractCentralityAlgorithm
    n::T1
    alpha::T2
    epsilon::T3
end
"""
```julia
Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
```

Construct a [`Pagerank`](@ref) algorithm.

# Arguments

  - `n`: Number of iterations (must be > 0).
  - `alpha`: Damping factor (must be in (0, 1)).
  - `epsilon`: Convergence tolerance (must be > 0).

# Returns

  - `Pagerank`: Algorithm object for PageRank centrality.

# Validation

  - `n > 0`.
  - `0 < alpha < 1`.
  - `epsilon > 0`.

# Examples

```jldoctest
julia> Pagerank(; alpha = 0.9, n = 200, epsilon = 1e-8)
Pagerank
        n | Int64: 200
    alpha | Float64: 0.9
  epsilon | Float64: 1.0e-8
```
"""
function Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
    @argcheck(n > 0 && zero(alpha) < alpha < one(alpha) && epsilon > zero(epsilon),
              DomainError("The following conditions must hold:\nn > 0 => n = $n\nalpha must be in (0, 1) => alpha = $alpha\nepsilon > 0 => epsilon = $epsilon"))
    return Pagerank(n, alpha, epsilon)
end

"""
```julia
struct RadialityCentrality <: AbstractCentralityAlgorithm end
```

Centrality algorithm type for radiality centrality in PortfolioOptimisers.jl.

`RadialityCentrality` computes the radiality centrality of nodes in a graph, measuring how close a node is to all other nodes, adjusted for the maximum possible distance.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct RadialityCentrality <: AbstractCentralityAlgorithm end

"""
```julia
struct StressCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
```

Centrality algorithm type for stress centrality in PortfolioOptimisers.jl.

`StressCentrality` computes the stress centrality of nodes in a graph, measuring the number of shortest paths passing through each node.

# Fields

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Constructor

```julia
StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct StressCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
"""
```julia
StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`StressCentrality`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the centrality computation.
  - `kwargs`: Keyword arguments for the centrality computation.

# Returns

  - `StressCentrality`: Algorithm object for stress centrality.

# Examples

```jldoctest
julia> StressCentrality()
StressCentrality
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return StressCentrality(args, kwargs)
end

"""
```julia
calc_centrality(cent::BetweennessCentrality, g::AbstractGraph)
calc_centrality(cent::ClosenessCentrality, g::AbstractGraph)
calc_centrality(cent::DegreeCentrality, g::AbstractGraph)
calc_centrality(::EigenvectorCentrality, g::AbstractGraph)
calc_centrality(cent::KatzCentrality, g::AbstractGraph)
calc_centrality(cent::Pagerank, g::AbstractGraph)
calc_centrality(::RadialityCentrality, g::AbstractGraph)
calc_centrality(cent::StressCentrality, g::AbstractGraph)
```

Compute node centrality scores for a graph using the specified centrality algorithm.

This function dispatches to the appropriate centrality computation from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/) based on the type of `cent`. Supported algorithms include betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, and stress centrality.

# Arguments

  - `cent::BetweennessCentrality`: Computes betweenness centrality using `Graphs.betweenness_centrality`.
  - `cent::ClosenessCentrality`: Computes closeness centrality using `Graphs.closeness_centrality`.
  - `cent::DegreeCentrality`: Computes degree centrality using `Graphs._degree_centrality`.
  - `cent::EigenvectorCentrality`: Computes eigenvector centrality using `Graphs.eigenvector_centrality`.
  - `cent::KatzCentrality`: Computes Katz centrality using `Graphs.katz_centrality`.
  - `cent::Pagerank`: Computes PageRank using `Graphs.pagerank`.
  - `cent::RadialityCentrality`: Computes radiality centrality using `Graphs.radiality_centrality`.
  - `cent::StressCentrality`: Computes stress centrality using `Graphs.stress_centrality`.
  - `g`: Graph to compute centrality on.

# Returns

  - `Vector{<:Real}`: Centrality scores for each node in the graph.

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
function calc_centrality(cent::BetweennessCentrality, g::AbstractGraph)
    return Graphs.betweenness_centrality(g, cent.args...; cent.kwargs...)
end
function calc_centrality(cent::ClosenessCentrality, g::AbstractGraph)
    return Graphs.closeness_centrality(g, cent.args...; cent.kwargs...)
end
function calc_centrality(cent::DegreeCentrality, g::AbstractGraph)
    return Graphs._degree_centrality(g, cent.kind; cent.kwargs...)
end
function calc_centrality(::EigenvectorCentrality, g::AbstractGraph)
    return Graphs.eigenvector_centrality(g::AbstractGraph)
end
function calc_centrality(cent::KatzCentrality, g::AbstractGraph)
    return Graphs.katz_centrality(g, cent.alpha)
end
function calc_centrality(cent::Pagerank, g::AbstractGraph)
    return Graphs.pagerank(g, cent.alpha, cent.n, cent.epsilon)
end
function calc_centrality(::RadialityCentrality, g::AbstractGraph)
    return Graphs.radiality_centrality(g::AbstractGraph)
end
function calc_centrality(cent::StressCentrality, g::AbstractGraph)
    return Graphs.stress_centrality(g, cent.args...; cent.kwargs...)
end

"""
```julia
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
```

Abstract supertype for all minimum spanning tree (MST) algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific MST algorithms (e.g., Kruskal, Boruvka, Prim) should subtype `AbstractTreeType`. This enables flexible extension and dispatch of tree-based routines for network and phylogeny analysis.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end

"""
```julia
struct KruskalTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
```

Algorithm type for Kruskal's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`KruskalTree` specifies the use of Kruskal's algorithm for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

```julia
KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractTreeType`](@ref)
"""
struct KruskalTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
"""
```julia
KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`KruskalTree`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Returns

  - `KruskalTree`: Algorithm object for Kruskal's MST.

# Examples

```jldoctest
julia> KruskalTree()
KruskalTree
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree(args, kwargs)
end

"""
```julia
struct BoruvkaTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
```

Algorithm type for Boruvka's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`BoruvkaTree` specifies the use of Boruvka's algorithm for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

```julia
BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractTreeType`](@ref)
"""
struct BoruvkaTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
"""
```julia
BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`BoruvkaTree`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Returns

  - `BoruvkaTree`: Algorithm object for Boruvka's MST.

# Examples

```jldoctest
julia> BoruvkaTree()
BoruvkaTree
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree(args, kwargs)
end

"""
```julia
struct PrimTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
```

Algorithm type for Prim's minimum spanning tree (MST) in PortfolioOptimisers.jl.

`PrimTree` specifies the use of Prim's algorithm for constructing a minimum spanning tree from a graph.

# Fields

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Constructor

```julia
PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

# Related

  - [`AbstractTreeType`](@ref)
"""
struct PrimTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
"""
```julia
PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
```

Construct a [`PrimTree`](@ref) algorithm.

# Arguments

  - `args`: Positional arguments for the MST computation.
  - `kwargs`: Keyword arguments for the MST computation.

# Returns

  - `PrimTree`: Algorithm object for Prim's MST.

# Examples

```jldoctest
julia> PrimTree()
PrimTree
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
```
"""
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree(args, kwargs)
end

"""
```julia
calc_mst(alg::KruskalTree, g::AbstractGraph)
calc_mst(alg::BoruvkaTree, g::AbstractGraph)
calc_mst(alg::PrimTree, g::AbstractGraph)
```

Compute the minimum spanning tree (MST) of a graph using the specified algorithm.

This function dispatches to the appropriate MST computation from `Graphs.jl` based on the type of `alg`. Supported algorithms include Kruskal, Boruvka, and Prim.

# Arguments

  - `alg::KruskalTree`: Computes the MST using Kruskal's algorithm (`Graphs.kruskal_mst`).
  - `alg::BoruvkaTree`: Computes the MST using Boruvka's algorithm (`Graphs.boruvka_mst`).
  - `alg::PrimTree`: Computes the MST using Prim's algorithm (`Graphs.prim_mst`).
  - `g::AbstractGraph`: Graph to compute the MST on.

# Returns

  - `tree::Vector`: Vector of edges representing the MST.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
function calc_mst(cent::KruskalTree, g::AbstractGraph)
    return Graphs.kruskal_mst(g, cent.args...; cent.kwargs...)
end
function calc_mst(cent::BoruvkaTree, g::AbstractGraph)
    return Graphs.boruvka_mst(g, cent.args...; cent.kwargs...)[1]
end
function calc_mst(cent::PrimTree, g::AbstractGraph)
    return Graphs.prim_mst(g, cent.args...; cent.kwargs...)
end

"""
```julia
abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end
```

Abstract supertype for all network estimator types in PortfolioOptimisers.jl.

All concrete types implementing network-based estimation algorithms should subtype `AbstractNetworkEstimator`. This enables a consistent interface for network estimators throughout the package.

# Related

  - [`Network`](@ref)
  - [`AbstractCentralityEstimator`](@ref)
"""
abstract type AbstractNetworkEstimator <: AbstractPhylogenyEstimator end

"""
```julia
struct Network{T1, T2, T3, T4} <: AbstractNetworkEstimator
    ce::T1
    de::T2
    alg::T3
    n::T4
end
```

Estimator type for network-based phylogeny analysis in PortfolioOptimisers.jl.

`Network` encapsulates the configuration for constructing a network from asset data, including the covariance estimator, distance estimator, tree or similarity algorithm, and the network depth parameter.

# Fields

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Tree or similarity matrix algorithm.
  - `n`: Network depth parameter.

# Constructor

```julia
Network(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType} = KruskalTree(),
        n::Integer = 1)
```

# Related

  - [`AbstractNetworkEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
"""
struct Network{T1, T2, T3, T4} <: AbstractNetworkEstimator
    ce::T1
    de::T2
    alg::T3
    n::T4
end
"""
```julia
Network(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType} = KruskalTree(),
        n::Integer = 1)
```

Construct a [`Network`](@ref) estimator for network-based phylogeny analysis.

Creates a network estimator using the specified covariance estimator, distance estimator, tree or similarity algorithm, and network depth.

# Arguments

  - `ce`: Covariance estimator.
  - `de`: Distance estimator.
  - `alg`: Tree or similarity matrix algorithm.
  - `n`: Network depth parameter (integer).

# Returns

  - `Network`: An estimator object for network-based phylogeny analysis.

# Examples

```jldoctest
julia> Network()
Network
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
  alg | KruskalTree
      |     args | Tuple{}: ()
      |   kwargs | @NamedTuple{}: NamedTuple()
    n | Int64: 1
```

# Related

  - [`Network`](@ref)
"""
function Network(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                 de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
                 alg::Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType} = KruskalTree(),
                 n::Integer = 1)
    return Network(ce, de, alg, n)
end

"""
```julia
abstract type AbstractCentralityEstimator <: AbstractPhylogenyEstimator end
```

Abstract supertype for all centrality estimator types in PortfolioOptimisers.jl.

All concrete types implementing centrality-based estimation algorithms should subtype `AbstractCentralityEstimator`. This enables a consistent interface for centrality estimators throughout the package.

# Related

  - [`Centrality`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
abstract type AbstractCentralityEstimator <: AbstractPhylogenyEstimator end

"""
```julia
struct Centrality{T1, T2} <: AbstractCentralityEstimator
    ne::T1
    cent::T2
end
```

Estimator type for centrality-based analysis in PortfolioOptimisers.jl.

`Centrality` encapsulates the configuration for computing centrality measures on a network, including the network estimator and the centrality algorithm.

# Fields

  - `ne`: Network estimator.
  - `cent`: Centrality algorithm.

# Constructor

```julia
Centrality(; ne::AbstractNetworkEstimator = Network(),
           cent::AbstractCentralityAlgorithm = DegreeCentrality())
```

# Related

  - [`AbstractCentralityEstimator`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
"""
struct Centrality{T1, T2} <: AbstractCentralityEstimator
    ne::T1
    cent::T2
end
"""
```julia
Centrality(; ne::AbstractNetworkEstimator = Network(),
           cent::AbstractCentralityAlgorithm = DegreeCentrality())
```

Construct a [`Centrality`](@ref) estimator for centrality-based analysis.

Creates a centrality estimator using the specified network estimator and centrality algorithm.

# Arguments

  - `ne`: Network estimator.
  - `cent`: Centrality algorithm.

# Returns

  - `Centrality`: An estimator object for centrality-based analysis.

# Examples

```jldoctest
julia> Centrality()
Centrality
    ne | Network
       |    ce | PortfolioOptimisersCovariance
       |       |   ce | Covariance
       |       |      |    me | SimpleExpectedReturns
       |       |      |       |   w | nothing
       |       |      |    ce | GeneralWeightedCovariance
       |       |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
       |       |      |       |    w | nothing
       |       |      |   alg | Full()
       |       |   mp | DefaultMatrixProcessing
       |       |      |       pdm | Posdef
       |       |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
       |       |      |   denoise | nothing
       |       |      |    detone | nothing
       |       |      |       alg | nothing
       |    de | Distance
       |       |   alg | CanonicalDistance()
       |   alg | KruskalTree
       |       |     args | Tuple{}: ()
       |       |   kwargs | @NamedTuple{}: NamedTuple()
       |     n | Int64: 1
  cent | DegreeCentrality
       |     kind | Int64: 0
       |   kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`Centrality`](@ref)
"""
function Centrality(; ne::AbstractNetworkEstimator = Network(),
                    cent::AbstractCentralityAlgorithm = DegreeCentrality())
    return Centrality(ne, cent)
end

"""
```julia
calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractTreeType, <:Any}, X::AbstractMatrix;
               dims::Int = 1, kwargs...)
calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm, <:Any},
               X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the adjacency matrix for a network estimator.

# Arguments

  - `ne`: Network estimator.

      + `ne::Network{<:Any, <:Any, <:AbstractTreeType, <:Any}`: Constructs a weighted graph from the distance matrix and computes the minimum spanning tree, returning the adjacency matrix of the resulting graph.
      + `ne::Network{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm, <:Any}`: Computes the similarity and distance matrices, applies the [`PMFG_T2s`](@ref) algorithm, and returns the adjacency matrix of the resulting graph..

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::Matrix{Int}`: Adjacency matrix representing the network.

# Related

  - [`Network`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
"""
function calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractTreeType, <:Any},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    D = distance(ne.de, ne.ce, X; dims = dims, kwargs...)
    G = SimpleWeightedGraph(D)
    tree = calc_mst(ne.alg, G)
    return adjacency_matrix(SimpleGraph(G[tree]))
end
function calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm,
                                    <:Any}, X::AbstractMatrix; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(ne.de, ne.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(ne.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end

"""
```julia
phylogeny_matrix(ne::Network, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the phylogeny matrix for a network estimator.

This function constructs the adjacency matrix for the network, then computes the phylogeny matrix by summing powers of the adjacency matrix up to the network depth parameter `n`, clamping values to 0 or 1, and removing self-loops.

# Arguments

  - `ne`: Network estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing asset relationships.

# Related

  - [`Network`](@ref)
  - [`calc_adjacency`](@ref)
"""
function phylogeny_matrix(ne::Network, X::AbstractMatrix; dims::Int = 1, kwargs...)
    A = calc_adjacency(ne, X; dims = dims, kwargs...)
    P = zeros(Int, size(Matrix(A)))
    for i in 0:(ne.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - I
    return P
end
"""
```julia
phylogeny_matrix(cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                 X::AbstractMatrix; branchorder::Symbol = :optimal, dims::Int = 1,
                 kwargs...)
```

Compute the phylogeny matrix for a clustering estimator or result.

This function applies clustering to the data, assigns assets to clusters, and constructs a binary phylogeny matrix indicating shared cluster membership, with self-loops removed.

# Arguments

  - `cle`: Clustering estimator or result.
  - `X`: Data matrix (observations × assets).
  - `branchorder`: Branch ordering strategy for hierarchical clustering (default: `:optimal`).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing cluster relationships.

# Related

  - [`ClusteringEstimator`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                          X::AbstractMatrix; branchorder::Symbol = :optimal, dims::Int = 1,
                          kwargs...)
    res = clusterise(cle, X; branchorder = branchorder, dims = dims, kwargs...)
    clusters = cutree(res.clustering; k = res.k)
    P = zeros(Int, size(X, 2), res.k)
    for i in axes(P, 2)
        idx = clusters .== i
        P[idx, i] .= one(eltype(P))
    end
    return P * transpose(P) - I
end

"""
```julia
centrality_vector(ne::Network, cent::AbstractCentralityAlgorithm, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
```

Compute the centrality vector for a network and centrality algorithm.

This function constructs the phylogeny matrix for the network, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Arguments

  - `ne`: Network estimator.
  - `cent`: Centrality algorithm.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::Vector{<:Real}`: Centrality scores for each asset.

# Related

  - [`Network`](@ref)
  - [`Centrality`](@ref)
  - [`calc_centrality`](@ref)
"""
function centrality_vector(ne::Network, cent::AbstractCentralityAlgorithm,
                           X::AbstractMatrix; dims::Int = 1, kwargs...)
    P = phylogeny_matrix(ne, X; dims = dims, kwargs...)
    G = SimpleGraph(P)
    return calc_centrality(cent, G)
end

"""
```julia
centrality_vector(cte::Centrality, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the centrality vector for a centrality estimator.

This function applies the centrality algorithm in the estimator to the network constructed from the data.

# Arguments

  - `cte`: Centrality estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::Vector{<:Real}`: Centrality scores for each asset.

# Related

  - [`Centrality`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::Centrality, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return centrality_vector(cte.ne, cte.cent, X; dims = dims, kwargs...)
end

"""
```julia
average_centrality(ne::Network, cent::AbstractCentralityAlgorithm, w::AbstractVector,
                   X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the weighted average centrality for a network and centrality algorithm.

This function computes the centrality vector and returns the weighted average using the provided weights.

# Arguments

  - `ne`: Network estimator.
  - `cent`: Centrality algorithm.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Real`: Average centrality.

# Related

  - [`Network`](@ref)
  - [`Centrality`](@ref)
  - [`centrality_vector`](@ref)
"""
function average_centrality(ne::Network, cent::AbstractCentralityAlgorithm,
                            w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return dot(centrality_vector(ne, cent, X; dims = dims, kwargs...), w)
end
"""
```julia
average_centrality(cte::Centrality, w::AbstractVector, X::AbstractMatrix; dims::Int = 1,
                   kwargs...)
```

Compute the weighted average centrality for a centrality estimator.

This function applies the centrality algorithm in the estimator to the network and returns the weighted average using the provided weights.

# Arguments

  - `cte`: Centrality estimator.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Real`: Average centrality.

# Related

  - [`Centrality`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::Centrality, w::AbstractVector, X::AbstractMatrix;
                            dims::Int = 1, kwargs...)
    return average_centrality(cte.ne, cte.cent, w, X; dims = dims, kwargs...)
end

"""
```julia
asset_phylogeny(w::AbstractVector, X::AbstractMatrix)
```

Compute the asset phylogeny score for a set of weights and a phylogeny matrix.

This function computes the weighted sum of the phylogeny matrix, normalised by the sum of absolute weights.

# Arguments

  - `w`: Weights vector.
  - `X`: Phylogeny matrix.

# Returns

  - `p::Real`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(w::AbstractVector, X::AbstractMatrix)
    aw = abs.(w * transpose(w))
    c = dot(X, aw)
    c /= sum(aw)
    return c
end
"""
```julia
asset_phylogeny(cle::Union{<:Network, <:ClusteringEstimator}, w::AbstractVector,
                X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the asset phylogeny score for a set of weights and a network or clustering estimator.

This function computes the phylogeny matrix using the estimator and data, then computes the asset phylogeny score using the weights.

# Arguments

  - `cle`: Network or clustering estimator.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute (default: `1`).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `p::Real`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(cle::Union{<:Network, <:ClusteringEstimator}, w::AbstractVector,
                         X::AbstractMatrix; dims::Int = 1, kwargs...)
    return asset_phylogeny(w, phylogeny_matrix(cle, X; dims = dims, kwargs...))
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, Network, phylogeny_matrix, average_centrality,
       asset_phylogeny, AbstractCentralityAlgorithm, Centrality, centrality_vector
