"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the polarity of the edge weights a centrality algorithm reads.

A weighted network carries one of two opposite quantities on its edges. A **distance** runs small-is-close; a **similarity** runs large-is-close. Which one an algorithm needs is a fact about its own mathematics and not about the graph it is handed: on one and the same triangulated maximally filtered graph, closeness wants the distances and eigenvector centrality wants the similarities. So the polarity is declared per algorithm, by [`centrality_polarity`](@ref), and the builder supplies the matching quantity.

# Polarity never decides whether the call succeeds

It selects **which** weights an algorithm receives, and nothing else. An algorithm that declares no polarity, and a source that carries no weights, both run on the plain unweighted graph rather than raising — see the warning on [`centrality_vector`](@ref) for the full list. Weightedness is a property of the source, not of the request: there is no flag, so a caller names a configured algorithm and never asks *for* weights. The one request there is, [`TopologyOnly`](@ref) in the algorithm's `ov` field, asks *away* from them, and every source can serve it.

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
    centrality_polarity(ct::Union{<:BetweennessCentrality{<:Any, <:Any, Nothing},
                                  <:ClosenessCentrality{<:Any, <:Any, Nothing},
                                  <:StressCentrality{<:Any, <:Any, Nothing},
                                  <:RadialityCentrality{Nothing}})
    centrality_polarity(ct::Union{<:BetweennessCentrality{<:Any, <:Any, TopologyOnly},
                                  <:ClosenessCentrality{<:Any, <:Any, TopologyOnly},
                                  <:StressCentrality{<:Any, <:Any, TopologyOnly},
                                  <:RadialityCentrality{TopologyOnly}})
    centrality_polarity(ct::EigenvectorCentrality{Nothing})
    centrality_polarity(ct::EigenvectorCentrality{TopologyOnly})

Answer which quantity a centrality algorithm's edge weights must be.

The extension contract of [`AbstractCentralityPolarity`](@ref). [`centrality_graph`](@ref) reads it to decide what to weight the network with.

# The answer is the effective polarity, not the declared one

A [`TopologyOnly`](@ref) in the algorithm's `ov` field withdraws the declaration, so this function answers `nothing` and the caller gets the plain graph. The override is resolved here rather than at the call site, because three algorithms carry no `ov` field at all and an inline read of `ct.ov` cannot be written for them. So this function keeps predicting the graph that [`centrality_graph`](@ref) builds, which is the property that makes it worth exporting.

# The fallback declares nothing, so opting in is explicit

The method on [`AbstractCentralityAlgorithm`](@ref) returns `nothing`, which routes to the plain unweighted graph. A new algorithm therefore runs unweighted until it says otherwise, which is the safe default: a wrong polarity does not raise, it silently reverses the ordering the algorithm is reading.

# What the shipped members declare, and why

  - [`DistancePolarity`](@ref) — [`BetweennessCentrality`](@ref), [`ClosenessCentrality`](@ref), [`RadialityCentrality`](@ref), [`StressCentrality`](@ref). All four are defined over shortest paths.
  - [`SimilarityPolarity`](@ref) — [`EigenvectorCentrality`](@ref). It is the leading eigenvector of the adjacency matrix itself, so a larger entry must mean a stronger link.
  - `nothing` — [`DegreeCentrality`](@ref), [`Pagerank`](@ref), [`KatzCentrality`](@ref). `Graphs.jl` cannot use weights in any of the three: the first two ignore them, and `Graphs.katz_centrality` binarises through `adjacency_matrix(g, Bool)` and throws an `InexactError` when handed a weighted graph.
  - `nothing` — any of the first five carrying `ov = TopologyOnly()`. Those five are the only ones that carry the field, because the other three already answer `nothing` and have nothing to withdraw.

The line between the first two groups and the third is `Graphs.jl`'s own. The declaration is about correctness — which weights — and the absence of one is about capability.

# Arguments

  - $(arg_dict[:cta])

# Returns

  - `polarity::Option{<:AbstractCentralityPolarity}`: The effective polarity, or `nothing` for an algorithm that cannot read weights or has withdrawn its declaration. Each method returns one concrete type, never a `Union`.

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
function centrality_polarity(::Union{<:BetweennessCentrality{<:Any, <:Any, Nothing},
                                     <:ClosenessCentrality{<:Any, <:Any, Nothing},
                                     <:StressCentrality{<:Any, <:Any, Nothing},
                                     <:RadialityCentrality{Nothing}})::DistancePolarity
    return DistancePolarity()
end
function centrality_polarity(::Union{<:BetweennessCentrality{<:Any, <:Any, TopologyOnly},
                                     <:ClosenessCentrality{<:Any, <:Any, TopologyOnly},
                                     <:StressCentrality{<:Any, <:Any, TopologyOnly},
                                     <:RadialityCentrality{TopologyOnly}})::Nothing
    return nothing
end
function centrality_polarity(::EigenvectorCentrality{Nothing})::SimilarityPolarity
    return SimilarityPolarity()
end
function centrality_polarity(::EigenvectorCentrality{TopologyOnly})::Nothing
    return nothing
end

export DistancePolarity, SimilarityPolarity, centrality_polarity
