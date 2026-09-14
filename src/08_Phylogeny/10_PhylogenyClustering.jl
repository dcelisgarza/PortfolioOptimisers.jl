"""
$(DocStringExtensions.TYPEDSIGNATURES)

Internal dispatch helper for constructing a [`Clusters`](@ref) result within a network-based clustering workflow.

Selects the appropriate clustering routine based on `alg`, determines the optimal number of clusters, and returns a [`Clusters`](@ref) result encapsulating all relevant outputs.

# Algorithm

 1. Cluster `P`, through the branch that `alg` selects, giving `res`.

      + [`HClustAlgorithm`](@ref): `Clustering.hclust` under `alg.linkage` and `branchorder`.
      + [`DBHT`](@ref): [`DBHTs`](@ref) over `P` and `S`, whose last returned value is the clustering.
      + [`AbstractNonHierarchicalClusteringAlgorithm`](@ref): [`optimal_number_clusters`](@ref), which answers with the clustering and the count together.

 2. Select the number of clusters `k` with [`optimal_number_clusters`](@ref) over `onc`, `res` and `P`. The third branch holds `k` from step 1 already.

 3. Assemble the [`Clusters`](@ref) result from `res`, `S`, `D`, `P` and `k`.

# Arguments

  - `alg`: Clustering algorithm.

      + `alg::HClustAlgorithm`: Applies hierarchical clustering via `Clustering.hclust` on the pseudo-distance matrix `P`.
      + `alg::DBHT`: Applies Direct Bubble Hierarchical Tree clustering via [`DBHTs`](@ref) on `P` and `S`.
      + `alg::AbstractNonHierarchicalClusteringAlgorithm`: Applies non-hierarchical clustering via [`optimal_number_clusters`](@ref) on `P`.

  - $(arg_dict[:onc])

  - $(arg_dict[:S])

  - $(arg_dict[:D])

  - `P::MatNum`: Symmetric pseudo-distance matrix derived from the network or similarity structure.

  - `branchorder`: Branch ordering strategy for hierarchical clustering.

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

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
    res = Clustering.hclust(P; linkage = alg.linkage, branchorder = branchorder)
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

Cluster assets using a minimum spanning tree (MST) network structure and return a [`Clusters`](@ref) result.

Builds the MST from the distance matrix, accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{D}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

``\\mathbf{A}`` is [`calc_weighted_adjacency`](@ref)'s matrix, read off [`calc_weighted_adjacency_graph`](@ref)'s graph through its one-argument form, so this method reads the same structure as every other consumer of a network and carries **weights**, not `0`/`1` — the tree branch's polarity is the distance, which is what ``\\mathbf{D}^i - \\mathbf{A}^i`` subtracts a like quantity from. The two-argument entry point is the one used, because `D` is already in hand, and the graph is kept rather than discarded so that a budget **rule** is answered over it instead of re-deriving the distance.

# Only a hop count is admitted

The fourth type parameter is narrowed to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at **dispatch**. The power sum is indexed by `nte.nte.sep.n`, and a matrix power counts edges: there is no radius analogue of ``\\mathbf{D}^i - \\mathbf{A}^i``, so the refusal is the honest answer rather than a gap. [`phylogeny_matrix`](@ref) does have a radius method, so the two consumers of a network differ here on purpose.

# Algorithm

 1. Derive the correlation matrix `S` and the distance matrix `D` from `X` with `nte.nte.de` and `nte.nte.ce`.
 2. Build the tree over `D` with [`calc_weighted_adjacency_graph`](@ref)'s two-argument entry point, giving the structure `G`, and read its weighted adjacency matrix `A` with [`calc_weighted_adjacency`](@ref).
 3. Resolve `nte.nte.sep` against `G` with [`resolve_separation`](@ref), and read the hop count `n` off the resolved separation.
 4. Accumulate the pseudo-distance matrix `P` as the sum of `D^i - A^i` over `i in 0:n`.
 5. Clear the diagonal of `P`, and hand the symmetric matrix to [`_clusterise`](@ref) together with `S` and `D`.

# Arguments

  - `nte`: Network clustering estimator configured with an MST-based [`NetworkEstimator`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

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

Cluster assets using a Planar Maximally Filtered Graph (PMFG) network structure and return a [`Clusters`](@ref) result.

Builds the PMFG from the similarity matrix via [`PMFG_T2s`](@ref), accumulates a symmetric pseudo-distance matrix `P` over the configured network depth `n` as ``\\sum_{i=0}^{n}(\\mathbf{S}^i - \\mathbf{A}^i)``, and dispatches to `_clusterise` to perform the actual clustering and select the optimal number of clusters.

``\\mathbf{A}`` is [`calc_weighted_adjacency`](@ref)'s matrix, read off the graph as on the tree method, and this branch's polarity is the **similarity** — so ``\\mathbf{S}^i - \\mathbf{A}^i`` again subtracts a like quantity. The two-argument entry point is the one used, because `S` is already in hand, and the graph is kept for the same reason.

# Only a hop count is admitted

The fourth type parameter is narrowed to [`HopCount`](@ref), so a [`PathLength`](@ref) separation fails at **dispatch**. See the tree method: a matrix power counts edges, and there is no radius analogue of the power sum.

# Algorithm

 1. Derive the correlation matrix and the distance matrix `D` from `X` with `nte.nte.de` and `nte.nte.ce`, and check `D` against `nte.nte.alg`'s domain with [`assert_similarity_domain`](@ref).
 2. Convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref).
 3. Build the triangulated maximally filtered graph over `S` with [`calc_weighted_adjacency_graph`](@ref)'s two-argument entry point, giving the structure `G`, and read its weighted adjacency matrix `Rpm` with [`calc_weighted_adjacency`](@ref).
 4. Resolve `nte.nte.sep` against `G` with [`resolve_separation`](@ref), and read the hop count `n` off the resolved separation.
 5. Accumulate the pseudo-distance matrix `P` as the sum of `S^i - Rpm^i` over `i in 0:n`.
 6. Clear the diagonal of `P`, and hand the symmetric matrix to [`_clusterise`](@ref) together with `S` and `D`.

# Arguments

  - `nte`: Network clustering estimator configured with a similarity-matrix-based [`NetworkEstimator`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Validation

  - Throws a `DomainError` if `D` leaves the domain of `nte.nte.alg`, through [`assert_similarity_domain`](@ref).

# Returns

  - `clr::Clusters`: Clustering result containing the clustering object, similarity matrix, distance matrix, pseudo-distance matrix, and optimal number of clusters.

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

Alias for a hierarchical clustering estimator or result.

Matches either a [`ClustersEstimator`](@ref) parameterised with a hierarchical clustering algorithm, or a [`Clusters`](@ref) result wrapping a `Clustering.Hclust`. Used internally for dispatch in hierarchical clustering workflows.

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
