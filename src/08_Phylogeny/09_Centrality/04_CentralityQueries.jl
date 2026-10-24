"""
    centrality_graph(pl::ClE_Cl, ct::AbstractCentralityAlgorithm, X::MatNum;
                     dims::Int = 1, kwargs...)
    centrality_graph(nte::AbstractNetworkEstimator, ct::AbstractCentralityAlgorithm,
                     X::MatNum; dims::Int = 1, kwargs...)
    centrality_graph(polarity::Option{<:AbstractCentralityPolarity},
                     nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Build the graph that [`calc_centrality`](@ref) scores, with the weights of the polarity that [`centrality_polarity`](@ref) returns for `ct`.

This function is the one place that holds both the source and the algorithm, so it is the one place that can match them. [`centrality_polarity`](@ref) states the quantity that `ct` needs, and the source states the quantities that it has.

# The routing

| source                                              | polarity                     | graph                                                                                  |
|:--------------------------------------------------- |:---------------------------- |:-------------------------------------------------------------------------------------- |
| [`AbstractNetworkEstimator`](@ref)                  | [`DistancePolarity`](@ref)   | [`calc_distance_weighted_graph`](@ref), with distances on either branch                |
| [`NetworkEstimator`](@ref) on the similarity branch | [`SimilarityPolarity`](@ref) | [`calc_weighted_adjacency_graph`](@ref), with the similarities that selected the edges |
| any source                                          | `nothing`                    | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                               |
| a clustering estimator or [`Clusters`](@ref)        | any                          | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                               |
| [`AbstractNetworkEstimator`](@ref) on a tree branch | [`SimilarityPolarity`](@ref) | plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref)                               |

The similarity route is narrower than the distance route on purpose. [`calc_distance_weighted_graph`](@ref) carries distances on both branches, but a similarity selects the edges of the similarity branch only. On a tree, [`calc_mst`](@ref) selects the edges by a minimised distance. A similarity made from that distance puts on the structure a quantity that did not select it.

# A partition carries no weights, and does not borrow any

A clustering source can reach a distance estimator through its own `de`, and it does not use it. The triangulated maximally filtered graph selects each edge by a pairwise quantity, so a distance orders that selection. A partition selects by a dendrogram and a cut, and two assets in the same cluster can be far apart in the distance. The distance does not order the membership of a cluster, so the partition has no quantity to borrow.

# Only the unweighted route reads the separation

The unweighted route goes through [`phylogeny_matrix`](@ref), so it reads the [`AbstractSeparationAlgorithm`](@ref) of the estimator. A [`HopCount`](@ref) of `n = 2` gives the centrality on the closure at two hops. The weighted routes do not go through [`phylogeny_matrix`](@ref), and read the structure. A closure is a sum of matrix powers, and a power of a weighted matrix sums products of distances, which is not a separation. The weighted routes do not read the `sep` field. At the default `HopCount(; n = 1)` the two routes agree, because the closure of a graph at one hop is the graph.

# Two entry points, because the function resolves the polarity one time

The methods that take `ct` resolve [`centrality_polarity`](@ref), and pass the polarity to the methods that take it. [`separation_matrix`](@ref) has the same shape. The algorithm that decides comes first, and the estimator only gives the graph.

# Algorithm

 1. Return the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref) when the source is a clustering estimator or a [`Clusters`](@ref). A partition carries no edge weights, so the function reads no polarity.

 2. Resolve the effective polarity of `ct` with [`centrality_polarity`](@ref). The methods that take the polarity receive it and start at step 3.

 3. Build the graph, through the branch that the polarity and the source select together.

      + `nothing`: the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref). It is the one route that reads the `sep` of the estimator.
      + [`DistancePolarity`](@ref): the structure of [`calc_distance_weighted_graph`](@ref), with distances on either branch.
      + [`SimilarityPolarity`](@ref) on a similarity branch: the structure of [`calc_weighted_adjacency_graph`](@ref), with the similarities that selected its edges.
      + [`SimilarityPolarity`](@ref) on a tree branch: the plain graph, because a minimised distance selects a tree, and a tree holds no similarity to read.

# Arguments

  - `pl`: Clustering estimator or clustering result.
  - $(arg_dict[:cta])
  - `polarity`: Effective polarity of `ct`, from [`centrality_polarity`](@ref).
  - $(arg_dict[:nte])
  - $(arg_dict[:X])
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

Compute the centrality vector of the structure that `pl` builds, with the algorithm `ct`.

[`centrality_graph`](@ref) builds the graph, with the weights of the polarity that [`centrality_polarity`](@ref) returns for `ct` where the source can give them. [`calc_centrality`](@ref) then scores each asset.

!!! warning

    Five cases run on the unweighted graph, and none of them raises an error. A caller names a configured algorithm and never asks for weights. A source and an algorithm that cannot use weights have no request to refuse. [`TopologyOnly`](@ref) asks for no weights, and every source can give that, so it is not one of the five cases.

     1. A clustering estimator, a precomputed [`Clusters`](@ref), or a precomputed [`PhylogenyResult`](@ref) as the source. Neither a partition nor a precomputed phylogeny matrix has edge weights, and neither borrows weights.
     2. [`DegreeCentrality`](@ref). `Graphs.jl` does not read the weights.
     3. [`Pagerank`](@ref). `Graphs.jl` does not read the weights.
     4. [`KatzCentrality`](@ref). `Graphs.katz_centrality` reads `adjacency_matrix(g, Bool)`, and raises an `InexactError` on a weighted graph.
     5. [`EigenvectorCentrality`](@ref) on a tree branch. The branch carries no similarity for it to read.

    The weighted routes do not read the `sep` field of the estimator. They read the structure, and not the separation closure that [`phylogeny_matrix`](@ref) builds. At the default `HopCount(; n = 1)` the two agree, because the closure of a graph at one hop is the graph.

# Algorithm

 1. Build the graph with [`centrality_graph`](@ref), with the weights of the polarity that `ct` declares wherever the source can give them.
 2. Score the vertices of that graph with [`calc_centrality`](@ref).
 3. Return the scores in a [`PhylogenyResult`](@ref).

# Arguments

  - `pl`: Network estimator, clustering estimator, or clustering result.
  - $(arg_dict[:cta])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::PhylogenyResult{<:VecNum}`: One centrality score for each asset.

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

Compute the centrality vector that a centrality estimator states.

The method calls [`centrality_vector`](@ref) with the `pl` and the `ct` of `cte`.

# Arguments

  - $(arg_dict[:cte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `cv::PhylogenyResult{<:VecNum}`: One centrality score for each asset.

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

Compute the average centrality of a portfolio, weighted by `w`.

The function computes the centrality vector, and returns its dot product with `w`. It is the average centrality measure of Equation 13.6 of [cajas2025](@cite),

```math
\\begin{align}
    \\mathrm{CM}(\\boldsymbol{x}) &= \\boldsymbol{C}_n^{\\intercal} \\boldsymbol{x}\\,,
\\end{align}
```

Where:

  - ``\\boldsymbol{C}_n``: Centrality score vector from [`centrality_vector`](@ref).
  - ``\\boldsymbol{x}``: Portfolio weight vector.

The function does not normalise and takes no absolute value, so the average has the units of the score. The score of [`DegreeCentrality`](@ref) is already divided by ``n - 1``.

# Algorithm

 1. Score the assets with [`centrality_vector`](@ref).
 2. Return the dot product of the scores and `w`.

# Arguments

  - $(arg_dict[:pler])
  - $(arg_dict[:cta])
  - `w`: Portfolio weight vector.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.5.1, Equation 13.6.
"""
function average_centrality(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            X::MatNum; dims::Int = 1, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, X; dims = dims, kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, X::MatNum;
                       dims::Int = 1, kwargs...)

Compute the average centrality of a portfolio with the configuration of a centrality estimator.

The method calls [`average_centrality`](@ref) with the `pl` and the `ct` of `cte`.

# Arguments

  - $(arg_dict[:cte])
  - `w`: Portfolio weight vector.
  - $(arg_dict[:X])
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
