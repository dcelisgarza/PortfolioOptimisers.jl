"""
$(DocStringExtensions.TYPEDSIGNATURES)

Cluster a matrix of pseudo-distances with `alg`, and return a [`Clusters`](@ref) result.

The type of `alg` selects the clustering routine, and `onc` selects the number of clusters. Both [`clusterise`](@ref) methods of a [`NetworkClustersEstimator`](@ref) call this function.

# Algorithm

 1. Cluster `P` into `res`, through the branch that `alg` selects.

      + [`HClustAlgorithm`](@ref): [`branch_ordered_hclust`](@ref) under `alg.linkage` on `Val(branchorder)`.
      + [`DBHT`](@ref): [`DBHTs`](@ref) over `P` and `S`. The last value that it returns is the clustering.
      + [`AbstractNonHierarchicalClusteringAlgorithm`](@ref): [`optimal_number_clusters`](@ref), which returns the clustering and the number of clusters together.

 2. Select the number of clusters `k` with [`optimal_number_clusters`](@ref) over `onc`, `res` and `P`. The third branch already has `k` from step 1.

 3. Return the [`Clusters`](@ref) result of `res`, `S`, `D`, `P` and `k`.

# Arguments

  - `alg`: Clustering algorithm.

      + `alg::HClustAlgorithm`: Hierarchical clustering of `P` with [`branch_ordered_hclust`](@ref).
      + `alg::DBHT`: Direct Bubble Hierarchical Tree clustering of `P` and `S` with [`DBHTs`](@ref).
      + `alg::AbstractNonHierarchicalClusteringAlgorithm`: Non-hierarchical clustering of `P` with [`optimal_number_clusters`](@ref).

  - $(arg_dict[:onc])

  - $(arg_dict[:S])

  - $(arg_dict[:D])

  - `P::MatNum`: Symmetric matrix of pseudo-distances from the network.

  - `branchorder`: Order method for the dendrogram branches, `:optimal` by default. [`branch_ordered_hclust`](@ref) and [`DBHTs`](@ref) state what each value does.

# Returns

  - `clr::Clusters`: Clustering result. It holds the clustering, the similarity matrix, the distance matrix, the matrix of pseudo-distances and the number of clusters.

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
    res = branch_ordered_hclust(P, alg.linkage, Val(branchorder))
    k = optimal_number_clusters(onc, res, P)
    return Clusters(; res = res, S = S, D = D, P = P, k = k)
end
function _clusterise(alg::DBHT, onc::AbstractOptimalNumberClustersEstimator, S::MatNum,
                     D::MatNum, P::MatNum; branchorder::Symbol = :optimal)
    res = DBHTs(P, S; branchorder = branchorder, root = alg.root, sim = alg.sim)[end]
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
                                                                <:AbstractTreeType,
                                                                <:HopCount}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster the assets over a minimum spanning tree, and return a [`Clusters`](@ref) result.

The method builds the tree from the distance matrix. It adds up the symmetric matrix of pseudo-distances `P` as ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)`` over the hop count `n`. [`_clusterise`](@ref) then clusters `P` and selects the number of clusters.

``\\mathbf{A}`` is the matrix of [`calc_weighted_adjacency`](@ref), read from the graph of [`calc_weighted_adjacency_graph`](@ref) through the method that takes `G`. This method reads the same structure as every other consumer of a network. ``\\mathbf{A}`` carries weights, and not `0` and `1`. The polarity of the tree branch is the distance, so ``\\mathbf{D}^i - \\mathbf{A}^i`` subtracts a quantity of the same kind. The method uses the entry point that takes `alg`, because it already holds `D`. It keeps the graph, so that a budget rule resolves over it and the distance is not computed again.

# The method admits only a hop count

The fourth type parameter is bound to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at dispatch. `nte.nte.sep.n` indexes the sum of powers, and a matrix power counts edges. A radius has no equivalent of ``\\mathbf{D}^i - \\mathbf{A}^i``, so the refusal is correct, and it is not a missing method. [`phylogeny_matrix`](@ref) has a method for a radius, so the two consumers of a network are different here on purpose.

# Algorithm

 1. Compute the correlation matrix `S` and the distance matrix `D` from `X` with `nte.nte.de` and `nte.nte.ce`.
 2. Build the tree `G` over `D` with the entry point of [`calc_weighted_adjacency_graph`](@ref) that takes `alg`. Read its weighted adjacency matrix `A` with [`calc_weighted_adjacency`](@ref).
 3. Resolve `nte.nte.sep` with [`resolve_separation`](@ref), against the structure that [`separation_graph`](@ref) makes from `G`. Read the hop count `n` from the resolved separation.
 4. Add up the matrix of pseudo-distances `P` as the sum of `D^i - A^i` over `i in 0:n`.
 5. Set the diagonal of `P` to zero. Give the symmetric matrix, `S` and `D` to [`_clusterise`](@ref).

# Arguments

  - `nte`: Network clustering estimator, whose [`NetworkEstimator`](@ref) builds a minimum spanning tree.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Order method for the dendrogram branches, `:optimal` by default. [`branch_ordered_hclust`](@ref) and [`DBHTs`](@ref) state what each value does.
  - `kwargs...`: Keyword arguments for the estimators of `nte`.

# Validation

  - Throws a `DomainError` if an off-diagonal entry of `D` is negative or `NaN`, through [`graph_weight_matrix`](@ref).

# Returns

  - `clr::Clusters`: Clustering result. It holds the clustering, the similarity matrix, the distance matrix, the matrix of pseudo-distances and the number of clusters.

# Related

  - [`NetworkClustersEstimator`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`Clusters`](@ref)
  - [`_clusterise`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_mst`](@ref)
  - [`HopCount`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractTreeType,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    P = zeros(eltype(D), size(D))
    # The distance is the tree branch's selecting quantity, and it is in hand already for
    # the power sum below, so the shared routine is entered at its two-argument form.
    G = calc_weighted_adjacency_graph(nte.nte.alg, D)
    A = calc_weighted_adjacency(G)
    # `n` is read as a matrix-power count rather than as a budget: a separation member that
    # measures something other than hops has no `n`, and fails here rather than silently
    # truncating a power sum it cannot index. `resolve_separation` is what makes a rule in
    # that field safe to index by -- it checks that the rule answered with an `Integer`.
    #
    # It is handed the structure above rather than `X` alone. A rule asked to derive its own
    # would repeat this method's `cor_and_dist`, which is 98% of its runtime under
    # `VariationInfoDistance`; the binarisation `separation_graph` does instead is one pass
    # over the edges. That is the whole reason the two-argument entry point exists.
    n = resolve_separation(nte.nte.sep, nte.nte, X, separation_graph(nte.nte.sep, G);
                           dims = dims, kwargs...).n
    for i in 0:n
        P .+= D^i - A^i
    end
    P .-= LinearAlgebra.Diagonal(P)
    return _clusterise(nte.alg, nte.onc, S, D, LinearAlgebra.Symmetric(P);
                       branchorder = branchorder)
end
"""
    clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                <:HopCount}},
               X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)

Cluster the assets over a planar maximally filtered graph (PMFG), and return a [`Clusters`](@ref) result.

The method builds the PMFG from the similarity matrix with [`PMFG_T2s`](@ref). It adds up the symmetric matrix of pseudo-distances `P` as ``\\sum_{i=0}^{n}(\\mathbf{S}^i - \\mathbf{A}^i)`` over the hop count `n`. [`_clusterise`](@ref) then clusters `P` and selects the number of clusters.

``\\mathbf{A}`` is the matrix of [`calc_weighted_adjacency`](@ref), read from the graph as in the tree method. The polarity of this branch is the similarity, so ``\\mathbf{S}^i - \\mathbf{A}^i`` also subtracts a quantity of the same kind. The method uses the entry point that takes `alg`, because it already holds `S`, and it keeps the graph for the same reason as the tree method.

# The method admits only a hop count

The fourth type parameter is bound to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at dispatch. A matrix power counts edges, and a radius has no equivalent of the sum of powers, as in the tree method.

# Algorithm

 1. Compute the correlation matrix and the distance matrix `D` from `X` with `nte.nte.de` and `nte.nte.ce`. Check `D` against the domain of `nte.nte.alg` with [`assert_similarity_domain`](@ref).
 2. Convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref).
 3. Build the triangulated maximally filtered graph `G` over `S` with the entry point of [`calc_weighted_adjacency_graph`](@ref) that takes `alg`. Read its weighted adjacency matrix `Rpm` with [`calc_weighted_adjacency`](@ref).
 4. Resolve `nte.nte.sep` with [`resolve_separation`](@ref), against the structure that [`separation_graph`](@ref) makes from `G`. Read the hop count `n` from the resolved separation.
 5. Add up the matrix of pseudo-distances `P` as the sum of `S^i - Rpm^i` over `i in 0:n`.
 6. Set the diagonal of `P` to zero. Give the symmetric matrix, `S` and `D` to [`_clusterise`](@ref).

# Arguments

  - `nte`: Network clustering estimator, whose [`NetworkEstimator`](@ref) builds a PMFG from a similarity matrix.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Order method for the dendrogram branches, `:optimal` by default. [`branch_ordered_hclust`](@ref) and [`DBHTs`](@ref) state what each value does.
  - `kwargs...`: Keyword arguments for the estimators of `nte`.

# Validation

  - Throws a `DomainError` if `D` leaves the domain of `nte.nte.alg`, through [`assert_similarity_domain`](@ref).
  - Throws a `DomainError` if a zero weight removes an edge of the triangulated maximally filtered graph, through [`assert_pmfg_weights`](@ref).

# Returns

  - `clr::Clusters`: Clustering result. It holds the clustering, the similarity matrix, the distance matrix, the matrix of pseudo-distances and the number of clusters.

# Related

  - [`NetworkClustersEstimator`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`Clusters`](@ref)
  - [`_clusterise`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`HopCount`](@ref)
"""
function clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
    S, D = cor_and_dist(nte.nte.de, nte.nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.nte.alg, nte.nte.de, D)
    P = zeros(eltype(D), size(D))
    S = distance_to_similarity(nte.nte.alg; S = S, D = D)
    # The similarity is the PMFG branch's selecting quantity. See the tree method.
    G = calc_weighted_adjacency_graph(nte.nte.alg, S)
    Rpm = calc_weighted_adjacency(G)
    # See the tree method: a matrix-power count, not a budget, resolved before it is indexed
    # by, and resolved against the structure this method already built.
    n = resolve_separation(nte.nte.sep, nte.nte, X, separation_graph(nte.nte.sep, G);
                           dims = dims, kwargs...).n
    for i in 0:n
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

Union of the hierarchical clustering estimators and results.

It matches three types:

  - a [`ClustersEstimator`](@ref) with a hierarchical clustering algorithm,
  - a [`NetworkClustersEstimator`](@ref) with a hierarchical clustering algorithm,
  - a [`Clusters`](@ref) result that holds a `Clustering.Hclust`.

It is the bound of the `cle` argument of [`plot_dendrogram`](@ref) and [`plot_clusters`](@ref), which need a dendrogram.

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
