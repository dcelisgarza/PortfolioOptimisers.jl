abstract type AbstractCentralityAlgorithm <: AbstractPhilogenyAlgorithm end
struct BetweennessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality(args, kwargs)
end
struct ClosenessCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality(args, kwargs)
end
struct DegreeCentrality{T1, T2} <: AbstractCentralityAlgorithm
    kind::T1
    kwargs::T2
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
    @assert(kind in 0:2, DomainError("`kind` must be in (0:2):\nkind => $kind"))
    return DegreeCentrality(kind, kwargs)
end
struct EigenvectorCentrality <: AbstractCentralityAlgorithm end
struct KatzCentrality{T1} <: AbstractCentralityAlgorithm
    alpha::T1
end
function KatzCentrality(; alpha::Real = 0.3)
    return KatzCentrality(alpha)
end
struct Pagerank{T1, T2, T3} <: AbstractCentralityAlgorithm
    n::T1
    alpha::T2
    epsilon::T3
end
function Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
    @assert(n > 0)
    @assert(zero(alpha) < alpha < one(alpha))
    @assert(epsilon > zero(epsilon))
    return Pagerank(n, alpha, epsilon)
end
struct RadialityCentrality <: AbstractCentralityAlgorithm end
struct StressCentrality{T1, T2} <: AbstractCentralityAlgorithm
    args::T1
    kwargs::T2
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return StressCentrality(args, kwargs)
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
struct KruskalTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree(args, kwargs)
end
struct BoruvkaTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree(args, kwargs)
end
struct PrimTree{T1, T2} <: AbstractTreeType
    args::T1
    kwargs::T2
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree(args, kwargs)
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
struct Network{T1, T2, T3, T4} <: AbstractNetworkEstimator
    ce::T1
    de::T2
    alg::T3
    n::T4
end
function Network(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                 de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
                 alg::Union{<:AbstractSimilarityMatrixAlgorithm, <:AbstractTreeType} = KruskalTree(),
                 n::Integer = 1)
    return Network(ce, de, alg, n)
end
struct Centrality{T1, T2}
    ne::T1
    cent::T2
end
function Centrality(; ne::AbstractNetworkEstimator = Network(),
                    cent::AbstractCentralityAlgorithm = DegreeCentrality())
    return Centrality(ne, cent)
end
function calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractTreeType, <:Any},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    # S = cor(ne.ce, X; dims = dims, kwargs...)
    # D = distance(ne.de, S, X; dims = dims, kwargs...)
    D = distance(ne.de, ne.ce, X; dims = dims, kwargs...)
    G = SimpleWeightedGraph(D)
    tree = calc_mst(ne.alg, G)
    return adjacency_matrix(SimpleGraph(G[tree]))
end
function calc_adjacency(ne::Network{<:Any, <:Any, <:AbstractSimilarityMatrixAlgorithm,
                                    <:Any}, X::AbstractMatrix; dims::Int = 1, kwargs...)
    # S = cor(ne.ce, X; dims = dims, kwargs...)
    # D = distance(ne.de, S, X; dims = dims, kwargs...)
    S, D = cor_and_dist(ne.de, ne.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(ne.alg; S = S, D = D)
    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end
function philogeny_matrix(ne::Network, X::AbstractMatrix; dims::Int = 1, kwargs...)
    A = calc_adjacency(ne, X; dims = dims, kwargs...)
    P = zeros(Int, size(Matrix(A)))
    for i in 0:(ne.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - I
    return P
end
function philogeny_matrix(cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult},
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
function centrality_vector(ne::Network, cent::AbstractCentralityAlgorithm,
                           X::AbstractMatrix; dims::Int = 1, kwargs...)
    P = philogeny_matrix(ne, X; dims = dims, kwargs...)
    G = SimpleGraph(P)
    return calc_centrality(cent, G)
end
function centrality_vector(cte::Centrality, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return centrality_vector(cte.ne, cte.cent, X; dims = dims, kwargs...)
end
function average_centrality(ne::Network, cent::AbstractCentralityAlgorithm,
                            w::AbstractVector, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return dot(centrality_vector(ne, cent, X; dims = dims, kwargs...), w)
end
function average_centrality(cte::Centrality, w::AbstractVector, X::AbstractMatrix;
                            dims::Int = 1, kwargs...)
    return average_centrality(cte.ne, cte.cent, w, X; dims = dims, kwargs...)
end
function asset_philogeny(w::AbstractVector, X::AbstractMatrix)
    aw = abs.(w * transpose(w))
    c = dot(X, aw)
    c /= sum(aw)
    return c
end
function asset_philogeny(cle::Union{<:Network, <:ClusteringEstimator}, w::AbstractVector,
                         X::AbstractMatrix; dims::Int = 1, kwargs...)
    return asset_philogeny(w, philogeny_matrix(cle, X; dims = dims, kwargs...))
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, Network, philogeny_matrix, average_centrality,
       asset_philogeny, AbstractCentralityAlgorithm, Centrality, centrality_vector
