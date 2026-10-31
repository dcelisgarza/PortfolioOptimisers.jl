"""
    graph_weight_matrix(D::MatNum)

Return `D` as a matrix whose off-diagonal entries a `SimpleWeightedGraphs` edge weight can hold.

A distance matrix and a weighted graph give `0` different meanings. A distance of `0` is the smallest distance, so the two assets are as close as possible. In a weighted graph, a weight of `0` means that the edge is absent. `SimpleWeightedGraph` stores its input as a sparse matrix, and `add_edge!` refuses a zero weight. A zero distance in the constructor removes the edge that the minimum spanning tree wants most. The two assets are then not adjacent, and no error occurs.

A zero distance does not show bad data. `SimpleAbsoluteDistance` and `LogDistance` read `abs(rho)`, so a pair with a correlation of `-1` is at distance zero. An inverse ETF and its index are such a pair, and the two are as related as two assets can be. The square-root algorithms reach zero at the other end, because their `clamp!` sets the radicand to zero for every `rho >= 1`.

The function repairs a zero distance, and does not refuse it. Each off-diagonal zero moves to `nextfloat(zero(T))`, the smallest positive value of `T = float_if_integer(eltype(D))`. It is the value nearest to zero that an edge weight can hold. It is far below any distance that a caller can mean, and a sum with a distance of normal size does not change when it enters. When no entry moves, the function returns `D` and makes no copy.

An `Integer` has no smallest positive value, so the repaired copy of an integer matrix has a floating point element type. A `Rational` also has no smallest positive value, and `nextfloat` raises a `MethodError` for it.

The function refuses a negative entry and a `NaN`, because neither has a nearest value that an edge weight can hold. A negative distance inverts the order that it states, and the shortest path routines that read these weights return an answer for it with no error. An asset with zero variance gives a `NaN` correlation, and so a `NaN` distance. A `NaN` is false in every comparison that the tree algorithms make.

The function keeps `Inf`. [`LogDistance`](@ref) gives `Inf` for two uncorrelated assets, the graph accepts it, and a spanning tree takes those edges last.

# Algorithm

 1. Read every off-diagonal entry of `D`. Throw a `DomainError` on a negative entry and on a `NaN`. Set `repair` to `true` if an entry is zero.
 2. Return `D` when `repair` is `false`. The function makes no copy of an input that needs no repair.
 3. Copy `D` into `W`, whose element type is `T = float_if_integer(eltype(D))`. Set `tiny` to the smallest positive value of `T`.
 4. Set every off-diagonal zero of `W` to `tiny`, and return `W`.

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
    # An `Integer` has no smallest positive value, so the repaired copy floats it.
    T = float_if_integer(eltype(D))
    W = copyto!(similar(D, T), D)
    tiny = nextfloat(zero(T))
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

[`calc_weighted_adjacency`](@ref) and [`calc_adjacency`](@ref) each do one operation on the graph that this function returns, so they get the structure from it. [`calc_distance_weighted_graph`](@ref) builds the same edge set, and puts distances on it.

# Polarity is per branch, and the two branches are not interchangeable

Each branch keeps the quantity that selected its edges, and the two quantities go in opposite directions.

  - `alg::AbstractTreeType`: [`calc_mst`](@ref) minimises the distance, so the weights are distances. A small weight means that the two assets are closely related.
  - `alg::AbstractNonNegativeSimilarityMatrixAlgorithm`: [`PMFG_T2s`](@ref) maximises the gain over the similarity, so the weights are similarities. A large weight means that the two assets are closely related.

With the other quantity, a branch gets its weights from a quantity that did not select its edges, so the function converts neither branch. The result carries no polarity tag, because dispatch on the algorithm gives the polarity.

A consumer that follows a path must first dispatch on `nte.alg`. A shortest path over similarities inverts the order that it must state, and it returns an answer with no error. The two graphs have the same shape and different meanings.

# Two entry points, because the quantity that selects the edges is not always cheap

The methods that take `alg` take the quantity that selects the edges. It is the distance on the tree branch and the similarity on the PMFG branch. The methods that take `nte` compute that quantity from `X`, and pass it on. The same dispatch selects the branch in both cases, so the polarity above belongs to the algorithm and not to the entry point.

The methods that take `alg` are for a caller that already holds that matrix. [`clusterise`](@ref) is such a caller. It needs `D` and `S` for its sum of powers and for the [`Clusters`](@ref) that it returns. A second derivation here computes the same correlation two times. Under [`VariationInfoDistance`](@ref), the derivation is `98%` of the run time of `clusterise`, so a second one almost doubles it.

# The weights

  - Tree branch: each weight is positive, and finite or `Inf`. [`graph_weight_matrix`](@ref) moves every zero distance away from `0`, the value that means an absent edge. It refuses a negative distance and a `NaN`. The graph accepts `Inf`, which [`LogDistance`](@ref) gives for two uncorrelated assets.
  - PMFG branch: each weight is positive and finite. [`PMFG_T2s`](@ref) checks that its input is non-negative. It inserts every vertex whatever the gain, so it declines no edge. The graph stores a zero weight as an absent edge, so the structure then has fewer edges. For this reason, [`assert_pmfg_weights`](@ref) refuses a zero weight.

# Algorithm

**The tree branch**, under an [`AbstractTreeType`](@ref):

 1. Compute the distance matrix `D` from `X` with `nte.de` and `nte.ce`. The entry point that takes `alg` receives `D` and starts at step 2.
 2. Repair `D` with [`graph_weight_matrix`](@ref). Build the complete `SimpleWeightedGraphs.SimpleWeightedGraph` `G` over the repaired matrix.
 3. Find the edges of the minimum spanning tree of `G` with [`calc_mst`](@ref).
 4. Return the subgraph of `G` on those edges. It is the tree, and its weights are the distances of `D`.

**The similarity branch**, under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref):

 1. Compute the correlation and the distance matrix `D` from `X` with `nte.de` and `nte.ce`. Check `D` against the domain of `nte.alg` with [`assert_similarity_domain`](@ref). Convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref). The entry point that takes `alg` receives `S` and starts at step 2.
 2. Build the triangulated maximally filtered graph over `S` with [`PMFG_T2s`](@ref). Its weighted adjacency matrix is `A`.
 3. Refuse a zero weight in `A` with [`assert_pmfg_weights`](@ref).
 4. Return the `SimpleWeightedGraphs.SimpleWeightedGraph` over `A`. Its weights are the similarities of `S`.

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
  - Similarity branch: throws a `DomainError` if a zero weight removes an edge of the triangulated maximally filtered graph, through [`assert_pmfg_weights`](@ref).
  - Similarity branch, on the entry point that takes `nte` only: throws a `DomainError` if `D` leaves the domain of `alg`, through [`assert_similarity_domain`](@ref). The entry point that takes `alg` receives `S` and does not read `D`.

# Returns

  - `G::SimpleWeightedGraphs.SimpleWeightedGraph`: The network structure, with the weights of its branch.

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

`Graphs.adjacency_matrix` of a weighted graph returns the weights, and not `0` and `1`. This function gives the matrix form of [`calc_weighted_adjacency_graph`](@ref), with the same polarity on each branch. The entries are distances on the tree branch and similarities on the PMFG branch. Read the section on polarity of [`calc_weighted_adjacency_graph`](@ref) before you use the values.

The pattern of nonzero entries is the structure, so it is the same as that of [`calc_adjacency`](@ref) on the same input. Only the stored values are different.

The entry points are those of [`calc_weighted_adjacency_graph`](@ref), with one more call to `Graphs.adjacency_matrix`. One more method takes a graph that the caller already holds. `W` is the quantity that selects the edges, the distance on the tree branch and the similarity on the PMFG branch. [`clusterise`](@ref) builds the graph from that quantity, and reads the matrix from the graph through the method that takes `G`. It keeps the graph, because it needs the graph again to resolve a budget rule.

# Algorithm

 1. Build the network structure with [`calc_weighted_adjacency_graph`](@ref), through the entry point that the arguments name. The method that takes `G` receives the graph and starts at step 2.
 2. Return `Graphs.adjacency_matrix` of that graph. The graph is weighted, so the entries are its edge weights and not `0` and `1`.

# Arguments

  - `G`: Network structure that the caller holds, from [`calc_weighted_adjacency_graph`](@ref).
  - `alg`: Tree or similarity matrix algorithm.
  - `W`: Quantity that selects the edges on the branch of `alg`. It is a distance matrix under an [`AbstractTreeType`](@ref), and a similarity matrix under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref).
  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC`: Weighted adjacency matrix of the network, in the polarity of its branch.

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

Compute the binary adjacency matrix of the network that `nte` builds.

The structure comes from [`calc_weighted_adjacency_graph`](@ref). This function converts it to a `Graphs.SimpleGraph`, which removes the weights. Both branches use the same method, because [`calc_weighted_adjacency_graph`](@ref) selects the branch.

A consumer that needs the weights calls [`calc_weighted_adjacency`](@ref). It must then obey the polarity of each branch, which [`calc_weighted_adjacency_graph`](@ref) states. This function removes the weights, so the polarity does not apply to it.

# Algorithm

 1. Build the weighted network structure with [`calc_weighted_adjacency_graph`](@ref).
 2. Convert it to a `Graphs.SimpleGraph`, which keeps the edge set and removes the weights.
 3. Return `Graphs.adjacency_matrix` of that graph, which is binary.

# Arguments

  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `adj::SparseArrays.SparseMatrixCSC{Int, Int}`: Binary adjacency matrix of the network.

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

Build the network structure with distances on its edges, on either branch.

[`calc_weighted_adjacency_graph`](@ref) gives each branch the quantity that selected its edges, so its two branches have opposite polarities. This function gives both branches the distance. The edge set is the same, vertex for vertex. Only the weights are different, and only on the PMFG branch.

# Why this function can put distances on the PMFG branch

[`calc_weighted_adjacency_graph`](@ref) does not put on a structure a quantity that did not select it. The distance is a different case. Every [`AbstractSimilarityMatrixAlgorithm`](@ref) is a strictly decreasing function of the distance. The order of the similarities that selected the edges is then the reverse of the order of the distances, pair for pair. [`PMFG_T2s`](@ref) selects the edges from `S`, and `D` gives each selected edge its length.

A shortest path over the similarities is wrong. A shortest path minimises the sum of its edge weights, so over similarities it takes the route through the weakest links. Its order is then the reverse of the correct order. Nothing shows the error. Over the four similarity algorithms, the measured correlation of the reversed answer with the correct one is `0.95` to `0.97`.

# Algorithm

**The tree branch**, under an [`AbstractTreeType`](@ref):

 1. Return the graph of [`calc_weighted_adjacency_graph`](@ref). That branch already puts the distances of `D` on its edges, so the two structures are one graph.

**The similarity branch**, under an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref):

 1. Compute the correlation and the distance matrix `D` from `X`. Check `D` against the domain of `nte.alg` with [`assert_similarity_domain`](@ref). Convert the pair to the similarity matrix `S` with [`distance_to_similarity`](@ref).
 2. Repair `D` into `W` with [`graph_weight_matrix`](@ref). The tree branch does the same repair for the same reason, because a zero weight means an absent edge.
 3. Select the edges of the triangulated maximally filtered graph over `S` with [`PMFG_T2s`](@ref), into `A`. Refuse a zero weight in `A` with [`assert_pmfg_weights`](@ref).
 4. Read the row and the column of every stored entry of `A`. The entry of `W` at each of them is the length of that edge.
 5. Return the `SimpleWeightedGraphs.SimpleWeightedGraph` over those indices and lengths. It has the edge set of `A` and the distances of `D`.

# Arguments

  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Validation

  - Throws a `DomainError` if an off-diagonal entry of `D` is negative or `NaN`, through [`graph_weight_matrix`](@ref).
  - Similarity branch: throws a `DomainError` if `D` leaves the domain of `nte.alg`, through [`assert_similarity_domain`](@ref).
  - Similarity branch: throws a `DomainError` if a zero weight removes an edge of the triangulated maximally filtered graph, through [`assert_pmfg_weights`](@ref).

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
