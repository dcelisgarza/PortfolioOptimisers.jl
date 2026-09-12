"""
    centrality_graph(pl::ClE_Cl, ct::AbstractCentralityAlgorithm, X::MatNum;
                     dims::Int = 1, kwargs...)
    centrality_graph(nte::AbstractNetworkEstimator, ct::AbstractCentralityAlgorithm,
                     X::MatNum; dims::Int = 1, kwargs...)
    centrality_graph(polarity::Option{<:AbstractCentralityPolarity},
                     nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Build the graph [`calc_centrality`](@ref) runs on, weighted in the polarity [`centrality_polarity`](@ref) answers for `ct`.

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

# Algorithm

 1. Answer the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref) at once when the source is a clustering estimator or a [`Clusters`](@ref). A partition carries no edge weights, so no polarity is read.

 2. Resolve the effective polarity of `ct` with [`centrality_polarity`](@ref). The methods that take the polarity itself are handed it and start at step 3.

 3. Build the graph, through the branch the polarity and the source name together.

      + `nothing`: the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref). It is the one route that reads the estimator's `sep`.
      + [`DistancePolarity`](@ref): [`calc_distance_weighted_graph`](@ref)'s structure, which carries distances on either branch.
      + [`SimilarityPolarity`](@ref) on a similarity branch: [`calc_weighted_adjacency_graph`](@ref)'s structure, carrying the similarities that selected its edges.
      + [`SimilarityPolarity`](@ref) on a tree branch: the plain graph again, because a tree is selected by minimising a distance and holds no similarity to read.

# Arguments

  - $(arg_dict[:pler])
  - $(arg_dict[:cta])
  - `polarity`: Effective polarity of `ct`, from [`centrality_polarity`](@ref).
  - $(arg_dict[:nte])
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

This function builds the graph with [`centrality_graph`](@ref) — weighted in the polarity [`centrality_polarity`](@ref) answers for `ct`, where the source can supply it — and computes node centrality scores with [`calc_centrality`](@ref).

!!! warning

    Five cases run on the **unweighted** graph, and none of them raises. A caller names a configured algorithm and never asks *for* weights, so an unweightable pairing has not been handed a request it cannot serve. [`TopologyOnly`](@ref) asks *away* from them, which every source can serve, so it adds no case to this list and is not one of the five.

     1. A clustering estimator, a precomputed [`Clusters`](@ref), or a precomputed [`PhylogenyResult`](@ref) as the source. A partition has no edge weights, and does not borrow any.
     2. [`DegreeCentrality`](@ref). `Graphs.jl` ignores weights.
     3. [`Pagerank`](@ref). `Graphs.jl` ignores weights.
     4. [`KatzCentrality`](@ref). `Graphs.katz_centrality` binarises through `adjacency_matrix(g, Bool)`.
     5. [`EigenvectorCentrality`](@ref) on a tree branch. The branch carries no similarity for it to read.

    On the weighted routes the estimator's `sep` field is **inert**: they read the structure itself rather than the separation closure [`phylogeny_matrix`](@ref) builds. At the default `HopCount(; n = 1)` the two agree, because the closure of a graph at one hop is the graph.

# Algorithm

 1. Build the graph with [`centrality_graph`](@ref), weighted in the polarity `ct` declares wherever the source can supply it.
 2. Score the vertices of that graph with [`calc_centrality`](@ref).
 3. Wrap the scores in a [`PhylogenyResult`](@ref).

# Arguments

  - `pl`: Phylogeny estimator.
  - $(arg_dict[:cta])
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

  - $(arg_dict[:cte])
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

This function computes the centrality vector and returns the weighted average using the provided weights. It is the average centrality measure of [`CentralityEstimator`](@ref)'s Equation 13.6,

```math
\\begin{align}
    \\mathrm{CM}(\\boldsymbol{x}) &= \\boldsymbol{C}_n^{\\intercal} \\boldsymbol{x}\\,,
\\end{align}
```

Where:

  - ``\\boldsymbol{C}_n``: Centrality score vector from [`centrality_vector`](@ref).
  - ``\\boldsymbol{x}``: Portfolio weight vector.

There is no normalisation and no absolute value, so the average carries the units of the score. A [`DegreeCentrality`](@ref) score is divided by ``n - 1`` before it arrives here. Measured over the minimum spanning tree of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz` at equal weights, the code and the formula agree exactly.

# Algorithm

 1. Score the assets with [`centrality_vector`](@ref), giving the score vector.
 2. Take the dot product of that vector and `w`.

# Arguments

  - `pl`: NetworkEstimator estimator.
  - $(arg_dict[:cta])
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

  - $(arg_dict[:cte])
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

export centrality_vector, average_centrality
