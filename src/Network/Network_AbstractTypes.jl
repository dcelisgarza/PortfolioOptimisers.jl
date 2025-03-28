abstract type CentralityType end
struct BetweennessCentrality{T1 <: Tuple, T2 <: NamedTuple} <: CentralityType
    args::T1
    kwargs::T2
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct ClosenessCentrality{T1 <: Tuple, T2 <: NamedTuple} <: CentralityType
    args::T1
    kwargs::T2
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct DegreeCentrality{T1 <: Integer, T2 <: NamedTuple} <: CentralityType
    kind::T1
    kwargs::T2
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))
    @smart_assert(kind ∈ 0:2)
    return DegreeCentrality{typeof(kind), typeof(kwargs)}(kind, kwargs)
end
struct EigenvectorCentrality <: CentralityType end
struct KatzCentrality{T1 <: Real} <: CentralityType
    alpha::T1
end
function KatzCentrality(; alpha::Real = 0.3)
    return KatzCentrality{typeof(alpha)}(alpha)
end
struct Pagerank{T1 <: Real, T2 <: Integer, T3 <: Real} <: CentralityType
    alpha::T1
    n::T2
    epsilon::T3
end
function Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(n > 0)
    @smart_assert(epsilon > zero(epsilon))
    return Pagerank{typeof(alpha), typeof(n), typeof(epsilon)}(alpha, n, epsilon)
end
struct RadialityCentrality <: CentralityType end
struct StressCentrality{T1 <: Tuple, T2 <: NamedTuple} <: CentralityType
    args::T1
    kwargs::T2
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return StressCentrality{typeof(args), typeof(kwargs)}(args, kwargs)
end
function centrality(cent::BetweennessCentrality, g::AbstractGraph)
    return Graphs.betweenness_centrality(g, cent.args...; cent.kwargs...)
end
function centrality(cent::ClosenessCentrality, g::AbstractGraph)
    return Graphs.closeness_centrality(g, cent.args...; cent.kwargs...)
end
function centrality(cent::DegreeCentrality, g::AbstractGraph)
    return Graphs._degree_centrality(g, cent.kind; cent.kwargs...)
end
function centrality(::EigenvectorCentrality, g::AbstractGraph)
    return Graphs.eigenvector_centrality(g::AbstractGraph)
end
function centrality(cent::KatzCentrality, g::AbstractGraph)
    return Graphs.katz_centrality(g, cent.alpha)
end
function centrality(cent::Pagerank, g::AbstractGraph)
    return Graphs.pagerank(g, cent.alpha, cent.n, cent.epsilon)
end
function centrality(::RadialityCentrality, g::AbstractGraph)
    return Graphs.radiality_centrality(g::AbstractGraph)
end
function centrality(cent::StressCentrality, g::AbstractGraph)
    return Graphs.stress_centrality(g, cent.args...; cent.kwargs...)
end
abstract type TreeType end
struct KruskalTree{T1 <: Tuple, T2 <: NamedTuple} <: TreeType
    args::T1
    kwargs::T2
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct BoruvkaTree{T1 <: Tuple, T2 <: NamedTuple} <: TreeType
    args::T1
    kwargs::T2
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
struct PrimTree{T1 <: Tuple, T2 <: NamedTuple} <: TreeType
    args::T1
    kwargs::T2
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree{typeof(args), typeof(kwargs)}(args, kwargs)
end
abstract type NetworkAlgorithm end
struct TriangulatedMaximallyFilteredGraph{T1 <: CentralityType,
                                          T2 <: SimilarityMatrixEstimator, T3 <: Integer} <:
       NetworkAlgorithm
    cent::T1
    sim::T2
    n::T3
end
function TriangulatedMaximallyFilteredGraph(; cent::CentralityType = DegreeCentrality(),
                                            sim::SimilarityMatrixEstimator = DBHT_MaximumDistanceSimilarity(),
                                            n::Integer = 1)
    return TriangulatedMaximallyFilteredGraph{typeof(cent), typeof(sim), typeof(n)}(cent,
                                                                                    sim, n)
end
struct MinimumSpanningTree{T1 <: TreeType, T2 <: CentralityType, T3 <: Integer} <:
       NetworkAlgorithm
    cent::T1
    tree::T2
    n::T3
end
function MinimumSpanningTree(; cent::CentralityType = DegreeCentrality(),
                             tree::TreeType = KruskalTree(), n::Integer = 1,)
    return MinimumSpanningTree{typeof(cent), typeof(tree), typeof(n)}(cent, tree, n)
end
function mst(cent::KruskalTree, g::AbstractGraph)
    return Graphs.kruskal_mst(g, cent.args...; cent.kwargs...)
end
function mst(cent::BoruvkaTree, g::AbstractGraph)
    return Graphs.boruvka_mst(g, cent.args...; cent.kwargs...)[1]
end
function mst(cent::PrimTree, g::AbstractGraph)
    return Graphs.prim_mst(g, cent.args...; cent.kwargs...)
end
abstract type AbstractNetworkEstimator end
struct NetworkEstimator{T1 <: StatsBase.CovarianceEstimator,
                        T2 <: PortfolioOptimisersUnionDistanceMetric,
                        T3 <: NetworkAlgorithm} <: AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
end
function NetworkEstimator(;
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          de::PortfolioOptimisersUnionDistanceMetric = CanonicalDistance(),
                          alg::NetworkAlgorithm = MinimumSpanningTree())
    return NetworkEstimator{typeof(ce), typeof(de), typeof(alg)}(ce, de, alg)
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, TriangulatedMaximallyFilteredGraph, MinimumSpanningTree
