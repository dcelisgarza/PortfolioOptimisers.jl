abstract type AbstractCentralityAlgorithm <: AbstractPhilogenyAlgorithm end
struct BetweennessCentrality{T1 <: Tuple, T2 <: NamedTuple} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct ClosenessCentrality{T1 <: Tuple, T2 <: NamedTuple} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct DegreeCentrality{T1 <: Integer, T2 <: NamedTuple} <: AbstractCentralityAlgorithm
    kind::T1
    kwargs::T2
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
    @smart_assert(kind in 0:2)
    return DegreeCentrality{typeof(kind), typeof(kwargs)}(kind, kwargs)
end
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
struct KatzCentrality{T1 <: Real} <: AbstractCentralityAlgorithm
    alpha::T1
end
function KatzCentrality(; alpha::Real = 0.3)
    return KatzCentrality{typeof(alpha)}(alpha)
end
struct Pagerank{T1 <: Integer, T2 <: Real, T3 <: Real} <: AbstractCentralityAlgorithm
    n::T1
    alpha::T2
    epsilon::T3
end
function Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
    @smart_assert(n > 0)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(epsilon > zero(epsilon))
    return Pagerank{typeof(n), typeof(alpha), typeof(epsilon)}(n, alpha, epsilon)
end
struct RadialityCentrality <: AbstractCentralityAlgorithm end
struct StressCentrality{T1 <: Tuple, T2 <: NamedTuple} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return StressCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
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
abstract type AbstractTreeType <: AbstractPhilogenyAlgorithm end
struct KruskalTree{T1 <: Tuple, T2 <: NamedTuple} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct BoruvkaTree{T1 <: Tuple, T2 <: NamedTuple} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct PrimTree{T1 <: Tuple, T2 <: NamedTuple} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
function calc_mst(cent::KruskalTree, g::AbstractGraph)
    return Graphs.kruskal_mst(g, cent.args...; cent.kwargs...)
end
function calc_mst(cent::BoruvkaTree, g::AbstractGraph)
    return Graphs.boruvka_mst(g, cent.args...; cent.kwargs...)[1]
end
function calc_mst(cent::PrimTree, g::AbstractGraph)
    return Graphs.prim_mst(g, cent.args...; cent.kwargs...)
end
abstract type AbstractNetworkEstimator <: AbstractPhilogenyEstimator end
struct NetworkEstimator{T1 <: StatsBase.CovarianceEstimator,
                        T2 <: AbstractDistanceEstimator,
                        T3 <:
                        Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType},
                        T4 <: Integer} <: AbstractNetworkEstimator
    ce::T1
    de::T2
    alg::T3
    n::T4
end
function NetworkEstimator(;
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          de::AbstractDistanceEstimator = Distance(;
                                                                   alg = CanonicalDistance()),
                          alg::Union{<:AbstractSimilarityMatrixAlgorithm,
                                     <:AbstractTreeType} = KruskalTree(), n::Integer = 1)
    return NetworkEstimator{typeof(ce), typeof(de), typeof(alg), typeof(n)}(ce, de, alg, n)
end
struct CentralityEstimator{T1 <: AbstractNetworkEstimator,
                           T2 <: AbstractCentralityAlgorithm}
    ne::T1
    cent::T2
end
function CentralityEstimator(; ne::AbstractNetworkEstimator = NetworkEstimator(),
                             cent::AbstractCentralityAlgorithm = DegreeCentrality())
    return CentralityEstimator{typeof(ne), typeof(cent)}(ne, cent)
end
function calc_adjacency(ne::NetworkEstimator{<:Any, <:Any, <:AbstractTreeType, <:Any},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    # S = cor(ne.ce, X; dims = dims, kwargs...)
    # D = distance(ne.de, S, X; dims = dims, kwargs...)
    D = distance(ne.de, ne.ce, X; dims = dims, kwargs...)
    G = SimpleWeightedGraph(D)
    tree = calc_mst(ne.alg, G)
    return adjacency_matrix(SimpleGraph(G[tree]))
end
function calc_adjacency(ne::NetworkEstimator{<:Any, <:Any,
                                             <:AbstractSimilarityMatrixAlgorithm, <:Any},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    # S = cor(ne.ce, X; dims = dims, kwargs...)
    # D = distance(ne.de, S, X; dims = dims, kwargs...)
    S, D = cor_and_dist(ne.de, ne.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(ne.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end
function philogeny_matrix(ne::NetworkEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
    A = calc_adjacency(ne, X; dims = dims, kwargs...)
    P = zeros(Int, size(Matrix(A)))
    for i in 0:(ne.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - I
    return P
end
function philogeny_matrix(cle::ClusteringEstimator, X::AbstractMatrix;
                          branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    res = clusterise(cle, X; branchorder = branchorder, dims = dims, kwargs...)
    clusters = cutree(res.clustering; k = res.k)
    P = zeros(Int, size(X, 2), res.k)
    for i in axes(P, 2)
        idx = clusters .== i
        P[idx, i] .= one(eltype(P))
    end
    return P * transpose(P) - I
end
function centrality_vector(ne::NetworkEstimator, cent::AbstractCentralityAlgorithm,
                           X::AbstractMatrix; dims::Int = 1, kwargs...)
    P = philogeny_matrix(ne, X; dims = dims, kwargs...)
    G = SimpleGraph(P)
    return calc_centrality(cent, G)
end
function centrality_vector(cte::CentralityEstimator, X::AbstractMatrix; dims::Int = 1,
                           kwargs...)
    return centrality_vector(cte.ne, cte.cent, X; dims = dims, kwargs...)
end
function average_centrality(ne::NetworkEstimator, cent::AbstractCentralityAlgorithm,
                            w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return dot(centrality_vector(ne, cent, X; dims = dims, kwargs...), w)
end
function average_centrality(cte::CentralityEstimator, w::AbstractVector, X::AbstractMatrix;
                            dims::Int = 1, kwargs...)
    return average_centrality(cte.ne, cte.cent, w, X; dims = dims, kwargs...)
end
function asset_philogeny(w::AbstractVector, X::AbstractMatrix)
    aw = abs.(w * transpose(w))
    c = dot(X, aw)
    c /= sum(aw)
    return c
end
function asset_philogeny(cle::Union{<:NetworkEstimator, <:ClusteringEstimator},
                         w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return asset_philogeny(w, philogeny_matrix(cle, X; dims = dims, kwargs...))
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, NetworkEstimator, philogeny_matrix, average_centrality,
       asset_philogeny, AbstractCentralityAlgorithm, CentralityEstimator, centrality_vector
