"""
    graph_weight_matrix(D::MatNum)

Return `D` as a matrix whose off-diagonal entries are representable as `SimpleWeightedGraphs` edge weights.

A distance matrix and a weighted graph disagree about what `0` means. In the distance codomain `0` is the *floor* — two assets as close as they can be. In the graph representation `0` is the reserved value meaning *absent*: `SimpleWeightedGraph` sparsifies its input, and `add_edge!` with a zero weight refuses outright. Handing a zero distance straight to the constructor therefore deletes exactly the edge the minimum spanning tree most wants, and the two assets come out non-adjacent — the most related pair in the universe reported as unrelated, with no error raised.

A zero is not a symptom of bad data. `SimpleAbsoluteDistance` and `LogDistance` are defined on `abs(rho)`, so a perfectly *anti*-correlated pair — a long/short leg, an inverse ETF, a pairs trade — is at distance zero and is genuinely maximally related. The square-root algorithms reach zero from the other side, since their `clamp!` maps any `rho >= 1` to exactly zero.

So the zero is *repaired*, not rejected: each off-diagonal zero moves to `nextfloat(zero(eltype(D)))`, the smallest representable positive value. That is the nearest value the representation can carry, it is orders of magnitude below any distance a caller could mean, and it is absorbed exactly by any sum it enters. `D` itself is returned untouched when no entry needs moving, so the copy is only paid for when it buys something.

Negative and `NaN` entries have no such nearest representable value and are rejected. A negative distance inverts the ordering it expresses and is *unsound* rather than merely wrong under the shortest-path routines that consume these weights — they return an answer instead of raising. A `NaN` — which a zero-variance asset produces, via a `NaN` correlation — silently fails every comparison the tree algorithms make.

`Inf` is left alone: it is the honest distance between uncorrelated assets under [`LogDistance`](@ref), the graph accepts it, and a spanning tree simply takes those edges last.

# Algorithm

 1. Walk every off-diagonal entry of `D`. Throw a `DomainError` on a negative entry and on a `NaN`, and record whether any entry is zero, giving `repair`.
 2. Return `D` itself when `repair` is `false`. An input that needs no move is never copied.
 3. Copy `D` into `W`, and take `tiny`, the smallest representable positive value of the element type.
 4. Move every off-diagonal zero of `W` to `tiny`, giving the repaired matrix.

# Arguments

  - `D`: Symmetric distance matrix.

# Validation

  - Throws a `DomainError` if any off-diagonal entry is negative or `NaN`.

# Returns

  - `W::MatNum`: `D` itself, or a repaired copy of it.

# Related

  - [`calc_weighted_adjacency_graph`](@ref)
  - [`clusterise`](@ref)
"""
function graph_weight_matrix(D::MatNum)
    z = zero(eltype(D))
    repair = false
    for j in axes(D, 2), i in axes(D, 1)
        if i == j
            continue
        end
        d = D[i, j]
        @argcheck(!isnan(d) && d >= z,
                  DomainError(d,
                              "off-diagonal distances must be non-negative and not NaN, because a graph edge weight is. Got\nD[$i, $j] => $d"))
        repair |= iszero(d)
    end
    if !repair
        return D
    end
    W = copy(D)
    tiny = nextfloat(z)
    for j in axes(W, 2), i in axes(W, 1)
        if i != j && iszero(W[i, j])
            W[i, j] = tiny
        end
    end
    return W
end
"""
    calc_weighted_adjacency_graph(alg::AbstractTreeType, D::MatNum)
    calc_weighted_adjacency_graph(alg::AbstractNonNegativeSimilarityMatrixAlgorithm,
                                  S::MatNum)
    calc_weighted_adjacency_graph(nte::NetworkEstimator, X::MatNum; dims::Int = 1,
                                  kwargs...)

Build the weighted graph whose edges are the network structure.

This is the **one construction site** of that structure. [`calc_weighted_adjacency`](@ref) and [`calc_adjacency`](@ref) are each a single operation on the graph returned here, so how the structure is selected is decided in this function and nowhere else.

# Polarity is per branch, and the two branches are not interchangeable

Each branch keeps **the quantity that selected its own edges**, and the two quantities run in opposite directions.

  - `alg::AbstractTreeType`: [`calc_mst`](@ref) minimises the distance, so the weights are **distances**. Small means closely related.
  - `alg::AbstractNonNegativeSimilarityMatrixAlgorithm`: [`PMFG_T2s`](@ref) maximises the gain over the similarity, so the weights are **similarities**. Large means closely related.

Re-weighting either branch with the other quantity would weight a structure by the quantity that did not select it, so neither is converted. The result carries no polarity tag, because the polarity is recoverable by dispatch on the algorithm.

A consumer that walks a path must therefore branch on `nte.alg` first. A shortest path over similarities inverts the ordering it is meant to express and **returns an answer instead of raising**, so the two graphs are interchangeable in shape but not in meaning.

# Two entry points, because the selecting quantity is not always cheap

The two-argument methods take **the selecting quantity itself** — the distance on the tree branch, the similarity on the PMFG branch — and the three-argument method derives it from `X` and forwards. Which branch is which is decided by the same dispatch either way, so the polarity above is a property of the algorithm and not of the entry point.

The two-argument form exists for a caller that already holds that matrix. [`clusterise`](@ref) is one: it needs `D` and `S` for its own power sum and for the [`Clusters`](@ref) it returns, so re-deriving them here would compute the same correlation twice. That is not a rounding error — under [`VariationInfoDistance`](@ref) the derivation is `98%` of `clusterise`'s runtime, so the second one would almost double it.

# The weights

  - Tree branch: strictly positive, and finite or `Inf`. [`graph_weight_matrix`](@ref) moves every zero distance off the value the representation reserves for *absent*, and rejects a negative or a `NaN`. `Inf` is legal — it is the honest [`LogDistance`](@ref) between two uncorrelated assets.
  - PMFG branch: strictly positive and finite. [`PMFG_T2s`](@ref) checks its input for non-negativity, and it inserts every remaining vertex whatever the gain, so it declines no edge. A zero weight would therefore be stored as an *absent* edge and silently shrink the structure, which is why [`assert_pmfg_weights`](@ref) refuses one here.

# Algorithm

**The tree branch**, under an [`AbstractTreeType`](@ref):

 1. Derive the distance matrix `D` from `X` with `nte.de` and `nte.ce`. The two-argument entry point is handed `D` and starts at step 2.
 2. Repair `D` with [`graph_weight_matrix`](@ref), and build the complete `SimpleWeightedGraphs.SimpleWeightedGraph` `G` over the repaired matrix.
 3. Minimise the distance over `G` with [`calc_mst`](@ref), giving the edge vector of the tree.
 4. Take the subgraph of `G` on those edges. It is the tree, and it carries `D`'s distances.

**The similarity branch**, under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref):

 1. Derive the correlation and the distance matrix `D` from `X` with `nte.de` and `nte.ce`, check `D` against `nte.alg`'s domain with [`assert_similarity_domain`](@ref), and convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref). The two-argument entry point is handed `S` and starts at step 2.
 2. Maximise the planar gain over `S` with [`PMFG_T2s`](@ref), giving `A`, the weighted adjacency matrix of the triangulated maximally filtered graph.
 3. Refuse a zero weight in `A` with [`assert_pmfg_weights`](@ref).
 4. Build the `SimpleWeightedGraphs.SimpleWeightedGraph` over `A`. It carries `S`'s similarities.

# Arguments

  - `alg`: Tree or similarity matrix algorithm.
  - $(arg_dict[:D])
  - $(arg_dict[:S])
  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Validation

  - Tree branch: throws a `DomainError` if an off-diagonal entry of `D` is negative or `NaN`, through [`graph_weight_matrix`](@ref).
  - Similarity branch: throws a `DomainError` if a zero weight cost the triangulated maximally filtered graph an edge, through [`assert_pmfg_weights`](@ref).
  - Similarity branch, on the estimator entry point alone: throws a `DomainError` if `D` leaves the domain of `alg`, through [`assert_similarity_domain`](@ref). The two-argument entry point is handed `S` and never sees `D`.

# Returns

  - `G::SimpleWeightedGraphs.SimpleWeightedGraph`: The network structure, carrying its branch's own weights.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_adjacency`](@ref)
  - [`graph_weight_matrix`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`clusterise`](@ref)
"""
function calc_weighted_adjacency_graph(alg::AbstractTreeType, D::MatNum)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(graph_weight_matrix(D))
    return G[calc_mst(alg, G)]
end
function calc_weighted_adjacency_graph(alg::AbstractNonNegativeSimilarityMatrixAlgorithm,
                                       S::MatNum)
    A = PMFG_T2s(S)[1]
    assert_pmfg_weights(A, alg)
    return SimpleWeightedGraphs.SimpleWeightedGraph(A)
end
function calc_weighted_adjacency_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                             <:AbstractTreeType}, X::MatNum;
                                       dims::Int = 1, kwargs...)
    return calc_weighted_adjacency_graph(nte.alg,
                                         distance(nte.de, nte.ce, X; dims = dims,
                                                  kwargs...))
end
function calc_weighted_adjacency_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                             <:AbstractNonNegativeSimilarityMatrixAlgorithm},
                                       X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(nte.de, nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.alg, nte.de, D)
    return calc_weighted_adjacency_graph(nte.alg,
                                         distance_to_similarity(nte.alg; S = S, D = D))
end
"""
    calc_weighted_adjacency(G::Graphs.AbstractGraph)
    calc_weighted_adjacency(alg::Tree_SimMat, W::MatNum)
    calc_weighted_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the weighted adjacency matrix of the network structure.

`Graphs.adjacency_matrix` of a *weighted* graph returns the **weights**, not `0`/`1`, so this is the matrix form of [`calc_weighted_adjacency_graph`](@ref) and inherits that function's per-branch polarity unchanged: distances on the tree branch, similarities on the PMFG branch. Read the polarity section of [`calc_weighted_adjacency_graph`](@ref) before consuming the values.

The sparsity pattern is the structure itself, so it is identical to [`calc_adjacency`](@ref)'s on the same input. Only the stored values differ.

The entry points are [`calc_weighted_adjacency_graph`](@ref)'s, one `Graphs.adjacency_matrix` call further on, plus one for a caller that already holds the graph itself. `W` is the selecting quantity — the distance on the tree branch, the similarity on the PMFG branch — and [`clusterise`](@ref) supplies it directly, having already paid for it; it then reads the matrix off the graph it keeps, through the one-argument form, because it needs that graph again to answer a budget **rule**.

# Algorithm

 1. Build the network structure with [`calc_weighted_adjacency_graph`](@ref), through the entry point the arguments name. The one-argument method is handed the graph and starts at step 2.
 2. Read `Graphs.adjacency_matrix` off that graph. The graph is weighted, so the entries are its edge weights and not `0` and `1`.

# Arguments

  - `G`: Network structure a caller already holds, from [`calc_weighted_adjacency_graph`](@ref).
  - `alg`: Tree or similarity matrix algorithm.
  - `W`: Selecting quantity of `alg`'s branch: a distance matrix under an [`AbstractTreeType`](@ref), a similarity matrix under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref).
  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC`: Weighted adjacency matrix of the network, in its branch's own polarity.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`calc_adjacency`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`clusterise`](@ref)
"""
function calc_weighted_adjacency(G::Graphs.AbstractGraph)
    return Graphs.adjacency_matrix(G)
end
function calc_weighted_adjacency(alg::Tree_SimMat, W::MatNum)
    return Graphs.adjacency_matrix(calc_weighted_adjacency_graph(alg, W))
end
function calc_weighted_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return Graphs.adjacency_matrix(calc_weighted_adjacency_graph(nte, X; dims = dims,
                                                                 kwargs...))
end
"""
    calc_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the binary adjacency matrix for a network estimator.

The structure comes from [`calc_weighted_adjacency_graph`](@ref); this function is the round trip through `Graphs.SimpleGraph` that discards the weights. Both branches share the one body, because the branch is decided in the tier below.

Consumers that need the weights call [`calc_weighted_adjacency`](@ref) instead. They must then observe the per-branch polarity documented on [`calc_weighted_adjacency_graph`](@ref). The binarisation here is what exempts this function from it.

# Algorithm

 1. Build the weighted network structure with [`calc_weighted_adjacency_graph`](@ref).
 2. Rebuild it as a `Graphs.SimpleGraph`, which keeps the edge set and discards the weights.
 3. Read `Graphs.adjacency_matrix` off that graph, giving the binary matrix.

# Arguments

  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC{Int, Int}`: Binary adjacency matrix representing the network.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`calc_weighted_adjacency`](@ref)
  - [`calc_mst`](@ref)
  - [`PMFG_T2s`](@ref)
"""
function calc_adjacency(nte::NetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return Graphs.adjacency_matrix(Graphs.SimpleGraph(calc_weighted_adjacency_graph(nte, X;
                                                                                    dims = dims,
                                                                                    kwargs...)))
end
"""
    calc_distance_weighted_graph(nte::NetworkEstimator, X::MatNum; dims::Int = 1,
                                 kwargs...)

Build the network structure carrying **distances** on its edges, on either branch.

[`calc_weighted_adjacency_graph`](@ref) gives each branch the quantity that *selected* its edges, so its two branches hold opposite polarities. This function gives both branches the same one. The structure is unchanged — it is the same edge set, vertex for vertex — and only the weights differ, on the PMFG branch alone.

# Why the PMFG branch may be re-weighted here, and may not be there

Re-weighting a structure with a quantity that did not select it is what [`calc_weighted_adjacency_graph`](@ref) refuses. This is not that. Every [`AbstractSimilarityMatrixAlgorithm`](@ref) is a strictly decreasing function of the distance, so the similarity that selected the PMFG's edges is a **monotone image of `D`**, not a foreign quantity: `D` is the selecting quantity's preimage, and the same PMFG comes out of it.

What must not happen is a path taken over the similarities themselves. A shortest path *minimises* the sum of its edge weights, so over similarities it seeks the route through the **weakest** links — the ordering it produces is backwards. It is also quiet about it: measured over the four similarity algorithms, the backwards answer correlates `0.95` to `0.97` with the right one, which is close enough to pass a glance and not close enough to be usable.

# Algorithm

**The tree branch**, under an [`AbstractTreeType`](@ref):

 1. Return [`calc_weighted_adjacency_graph`](@ref)'s graph unchanged. That branch weights its edges with `D` already, so the two structures are one graph.

**The similarity branch**, under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref):

 1. Derive the correlation and the distance matrix `D` from `X`, check `D` against `nte.alg`'s domain with [`assert_similarity_domain`](@ref), and convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref).
 2. Repair `D` with [`graph_weight_matrix`](@ref), giving `W`. The repair is the tree branch's, needed here for the same reason: a zero distance is the value the representation reserves for *absent*.
 3. Select the edges by maximising the planar gain over `S` with [`PMFG_T2s`](@ref), giving `A`, and refuse a zero weight in it with [`assert_pmfg_weights`](@ref).
 4. Read the row and column index of every stored entry of `A`, and take the entry of `W` at each one, giving the length of every selected edge.
 5. Build the `SimpleWeightedGraphs.SimpleWeightedGraph` over those indices and lengths. It is `A`'s edge set carrying `D`'s distances.

# Arguments

  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Validation

  - Throws a `DomainError` if an off-diagonal entry of `D` is negative or `NaN`, through [`graph_weight_matrix`](@ref).
  - Similarity branch: throws a `DomainError` if `D` leaves the domain of `nte.alg`, through [`assert_similarity_domain`](@ref), and a `DomainError` if a zero weight cost the triangulated maximally filtered graph an edge, through [`assert_pmfg_weights`](@ref).

# Returns

  - `G::SimpleWeightedGraphs.SimpleWeightedGraph`: The network structure, weighted by distance on both branches.

# Related

  - [`NetworkEstimator`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`graph_weight_matrix`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
"""
function calc_distance_weighted_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                            <:AbstractTreeType}, X::MatNum;
                                      dims::Int = 1, kwargs...)
    # The tree branch already weights its edges with `D`, so the distance-weighted structure
    # and the selecting-quantity one are the same graph.
    return calc_weighted_adjacency_graph(nte, X; dims = dims, kwargs...)
end
function calc_distance_weighted_graph(nte::NetworkEstimator{<:Any, <:Any,
                                                            <:AbstractNonNegativeSimilarityMatrixAlgorithm},
                                      X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(nte.de, nte.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(nte.alg, nte.de, D)
    S = distance_to_similarity(nte.alg; S = S, D = D)
    # `PMFG_T2s` selects the edges from the similarity; `D` then supplies their lengths. The
    # repair is `calc_weighted_adjacency_graph`'s tree-branch one, needed here for the same
    # reason -- a zero distance is the value the representation reserves for *absent*.
    W = graph_weight_matrix(D)
    A = PMFG_T2s(S)[1]
    assert_pmfg_weights(A, nte.alg, nte.de)
    r, c, _ = SparseArrays.findnz(A)
    v = [W[i, j] for (i, j) in zip(r, c)]
    return SimpleWeightedGraphs.SimpleWeightedGraph(SparseArrays.sparse(r, c, v,
                                                                        size(W)...))
end
