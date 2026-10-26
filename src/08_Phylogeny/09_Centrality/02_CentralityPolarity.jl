"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the polarity of the edge weights a centrality algorithm reads.

A weighted network carries one of two opposite quantities on its edges. A small distance means that two assets are near, and a large similarity means that they are near. The mathematics of an algorithm decides which quantity it needs, and the graph does not. On the same triangulated maximally filtered graph, closeness needs the distances and eigenvector centrality needs the similarities. Each algorithm declares its polarity through [`centrality_polarity`](@ref), and [`centrality_graph`](@ref) gives it the quantity that it needs.

# Polarity never decides whether the call succeeds

It selects which weights an algorithm receives, and nothing else. An algorithm that declares no polarity, and a source that carries no weights, both run on the plain unweighted graph with no error. The warning of [`centrality_vector`](@ref) lists every such case. The source decides whether a graph has weights, and the request does not. No flag asks for weights, so a caller names a configured algorithm and never asks for weights. The one request is [`TopologyOnly`](@ref) in the `ov` field of the algorithm. It asks for no weights, and every source can give that.

# Related

  - [`DistancePolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`centrality_polarity`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_graph`](@ref)
"""
abstract type AbstractCentralityPolarity <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Declares that the edge weights of an algorithm must be distances, where a small value means closely related.

Every algorithm that follows a shortest path needs this polarity, because a shortest path minimises the sum of the weights along it. Over similarities, the same routine takes the route through the weakest links. It returns the reverse answer, and raises no error.

[`calc_distance_weighted_graph`](@ref) gives this polarity, with distances on both branches.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`centrality_polarity`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
"""
struct DistancePolarity <: AbstractCentralityPolarity end
"""
$(DocStringExtensions.TYPEDEF)

Declares that the edge weights of an algorithm must be similarities, where a large value means closely related.

An algorithm that reads the weighted adjacency matrix, and does not follow a path, needs an entry that grows with the relation. A stronger link must give more.

[`calc_weighted_adjacency_graph`](@ref) gives this polarity, on its similarity branch only. On the tree branch, [`calc_mst`](@ref) selects the edges by a minimised distance, and the branch holds no similarity. An algorithm that declares this polarity runs unweighted there.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`centrality_polarity`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
"""
struct SimilarityPolarity <: AbstractCentralityPolarity end
"""
    centrality_polarity(ct::AbstractCentralityAlgorithm)
    centrality_polarity(ct::Union{BetweennessCentrality{<:Any, <:Any, Nothing},
                                  ClosenessCentrality{<:Any, <:Any, Nothing},
                                  StressCentrality{<:Any, <:Any, Nothing},
                                  RadialityCentrality{Nothing}})
    centrality_polarity(ct::Union{BetweennessCentrality{<:Any, <:Any, TopologyOnly},
                                  ClosenessCentrality{<:Any, <:Any, TopologyOnly},
                                  StressCentrality{<:Any, <:Any, TopologyOnly},
                                  RadialityCentrality{TopologyOnly}})
    centrality_polarity(ct::EigenvectorCentrality{Nothing})
    centrality_polarity(ct::EigenvectorCentrality{TopologyOnly})

Return the quantity that the edge weights of a centrality algorithm must be.

It is the interface of [`AbstractCentralityPolarity`](@ref). [`centrality_graph`](@ref) reads it, and selects the weights of the network from it.

# The answer is the effective polarity, not the declared one

A [`TopologyOnly`](@ref) in the `ov` field of the algorithm withdraws the declaration. This function then returns `nothing`, and the caller gets the plain graph. This function resolves the override, and the call site does not. Three algorithms have no `ov` field, so a read of `ct.ov` at the call site fails for them. This function always predicts the graph that [`centrality_graph`](@ref) builds, and for this reason the library exports it.

# The method on the abstract type declares nothing, so a declaration is explicit

The method on [`AbstractCentralityAlgorithm`](@ref) returns `nothing`, which selects the plain unweighted graph. A new algorithm runs unweighted until it declares a polarity. This is the safe default. A wrong polarity raises no error, and it reverses the order that the algorithm reads.

# What the shipped members declare, and why

  - [`DistancePolarity`](@ref): [`BetweennessCentrality`](@ref), [`ClosenessCentrality`](@ref), [`RadialityCentrality`](@ref) and [`StressCentrality`](@ref). All four are defined over shortest paths.
  - [`SimilarityPolarity`](@ref): [`EigenvectorCentrality`](@ref). It is the leading eigenvector of the adjacency matrix, so a larger entry must mean a stronger link.
  - `nothing`: [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref). `Graphs.jl` cannot use weights in any of the three. The first two do not read them. `Graphs.katz_centrality` reads `adjacency_matrix(g, Bool)`, and throws an `InexactError` on a weighted graph.
  - `nothing`: any of the first five with `ov = TopologyOnly()`. Only those five have the field, because the other three already return `nothing` and have nothing to withdraw.

`Graphs.jl` sets the line between the first two groups and the third. A declaration is about correctness, that is, about which weights. The absence of a declaration is about capability.

# Arguments

  - $(arg_dict[:cta])

# Returns

  - `polarity::Option{<:AbstractCentralityPolarity}`: The effective polarity, or `nothing` for an algorithm that cannot read weights or that withdraws its declaration. Each method returns one concrete type, never a `Union`.

# Related

  - [`AbstractCentralityPolarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`centrality_graph`](@ref)
  - [`calc_centrality`](@ref)
"""
function centrality_polarity end
function centrality_polarity(::AbstractCentralityAlgorithm)::Nothing
    return nothing
end
function centrality_polarity(::Union{BetweennessCentrality{<:Any, <:Any, Nothing},
                                     ClosenessCentrality{<:Any, <:Any, Nothing},
                                     StressCentrality{<:Any, <:Any, Nothing},
                                     RadialityCentrality{Nothing}})::DistancePolarity
    return DistancePolarity()
end
function centrality_polarity(::Union{BetweennessCentrality{<:Any, <:Any, TopologyOnly},
                                     ClosenessCentrality{<:Any, <:Any, TopologyOnly},
                                     StressCentrality{<:Any, <:Any, TopologyOnly},
                                     RadialityCentrality{TopologyOnly}})::Nothing
    return nothing
end
function centrality_polarity(::EigenvectorCentrality{Nothing})::SimilarityPolarity
    return SimilarityPolarity()
end
function centrality_polarity(::EigenvectorCentrality{TopologyOnly})::Nothing
    return nothing
end

export DistancePolarity, SimilarityPolarity, centrality_polarity
