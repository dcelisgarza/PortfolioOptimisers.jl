"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Direct Bubble Hierarchy Tree (DBHT) root selection methods.

The root is chosen inside [`CliqHierarchyTree2s`](@ref), which builds the clique hierarchy of the planar graph. A hierarchy needs one node with no parent, and the planar clique tree can present several candidates, so the choice of which of them becomes the root is a member of this family.

# Related

  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)

# References

  - $(ref_dict[:NHPG])
  - $(ref_dict[:DBHTs])
"""
abstract type DBHTRootMethod <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes one clique of the planar hierarchy as its single root.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG])
"""
struct UniqueRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Builds one root from the adjacency tree of every root candidate.

This keeps several equally plausible roots of the DBHT hierarchy rather than choosing between them.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG])
"""
struct EqualRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Clusters assets by the bubble hierarchy of a triangulated maximally filtered graph.

`DBHT` is a composable clustering algorithm type for constructing hierarchical clusterings using the Direct Bubble Hierarchical Tree (DBHT) method, as described in [DBHTs](@cite).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DBHT(;
        sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
        root::DBHTRootMethod = UniqueRoot()
    ) -> DBHT

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> DBHT()
DBHT
   sim ┼ MaximumDistanceSimilarity()
  root ┴ UniqueRoot()
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`DBHTRootMethod`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)

# References

  - $(ref_dict[:DBHTs])
"""
@concrete struct DBHT <: AbstractHierarchicalClusteringAlgorithm
    """
    $(field_dict[:sim])
    """
    sim <: AbstractNonNegativeSimilarityMatrixAlgorithm
    """
    $(field_dict[:root])
    """
    root
    function DBHT(sim::AbstractNonNegativeSimilarityMatrixAlgorithm, root::DBHTRootMethod)
        return new{typeof(sim), typeof(root)}(sim, root)
    end
end
function DBHT(;
              sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              root::DBHTRootMethod = UniqueRoot())::DBHT
    return DBHT(sim, root)
end
"""
    PMFG_T2s(W::MatNum, nargout::Integer = 3)

Constructs a Triangulated Maximally Filtered Graph (TMFG) starting from a tetrahedron and recursively inserting vertices inside existing triangles (T2 move) in order to approximate a Maximal Planar Graph with the largest total weight, also known as the Planar Maximally Filtered Graph (PMFG). All weights must be non-negative.

This function is a core step in the DBHT (Direct Bubble Hierarchical Tree) and LoGo algorithms, providing the planar graph structure and clique information required for hierarchical clustering and sparse inverse covariance estimation.

`nargout` is a **positional** argument, and every caller passes it positionally.

# The TMFG approximates the PMFG, and is not it

The planar maximally filtered graph is the exact solution of the weighted maximal planar graph problem, which is costly. The triangulation this function builds is the cheap greedy approximation to it, so the name of the function is the problem and the algorithm is the approximation. Both are maximal planar graphs, so both carry exactly ``3N - 6`` edges against the ``N - 1`` of a minimum spanning tree: measured over a 20-asset sample, the graph holds `54` edges.

# Mathematical definition

The T2 move inserts vertex ``v`` into face ``f`` and gains the weight of the three edges it adds. The greedy step takes the pair that gains most.

```math
\\begin{align}
g(v,\\, f) &= \\sum_{u \\in f} W_{u,\\,v}\\,, \\\\
(v^{\\star},\\, f^{\\star}) &= \\underset{v \\notin V,\\, f \\in F}{\\arg\\max}\\; g(v,\\, f)\\,.
\\end{align}
```

Where:

  - ``W_{u,\\,v}``: Weight of the pair ``(u,\\, v)``, the entry of the input matrix.
  - ``f``: Triangular face, a set of three vertices.
  - ``F``: Set of the faces built so far.
  - ``V``: Set of the vertices inserted so far.
  - ``g(v,\\, f)``: Gain of inserting vertex ``v`` into face ``f``.
  - $(math_dict[:N])

# Algorithm

 1. Score every vertex by `s`, the row sum of `W` over the entries above the mean of `W`.
 2. Take the four vertices of largest `s` as `in_v[1:4]`, and the rest as `ou_v`.
 3. Build the tetrahedron on those four vertices: its four faces into `tri[1:4, :]`, and its six edges into `A`.
 4. Build the gain table `gain[v, f]`, one entry per vertex of `ou_v` and per face of `tri`.
 5. Take the pair of largest gain, giving the vertex `ve` and the face `agm`. Remove `ve` from `ou_v` and record it in `in_v`.
 6. Join `ve` to the three vertices of face `agm` in `A`, and record that face in `clique3`, so it becomes a 3-clique that is no longer a face.
 7. Replace face `agm` and append two more, so the three faces of the split each carry `ve`.
 8. Rebuild the three changed columns of `gain`, and zero the row of `ve`. Repeat from step 5 until every vertex is inserted.
 9. Weight the structure: `A = W ⊙ ((A + A') .== 1)`, so a stored entry is an edge and its value is its weight.
10. When `nargout > 3`, build `cliques`: the initial tetrahedron, then one 4-clique per inserted vertex, holding the face it entered and itself.
11. When `nargout > 4`, build `cliqueTree`: for each 4-clique, count in `ss` how many of its first three vertices every 4-clique holds, and mark the ones whose count is `2`.

# Arguments

  - `W`: `N × N` matrix of non-negative, finite weights (e.g. a similarity matrix from an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), or an absolute correlation matrix).
  - `nargout`: Number of outputs to build. `cliques` is built when `nargout > 3` and `cliqueTree` when `nargout > 4`; each is `nothing` otherwise. The first three outputs are always built.

# Validation

  - `N >= 9` is required for a meaningful PMFG.
  - No entry in `W` is `NaN`.
  - All entries in `W` are non-negative.

An entry that is exactly zero passes all three and still costs the graph an edge, because `A` carries
the structure in its sparsity pattern and this function declines no edge on the way in. That is
[`assert_pmfg_weights`](@ref)'s check, and it runs in the callers that consume the weighted structure
rather than here, because [`logo!`](@ref) reads only the cliques and is unaffected by a zero.

# The checks are a backstop, not the enforcement

Every estimator that reaches this function — [`NetworkEstimator`](@ref), [`DBHT`](@ref) and [`LoGo`](@ref) — bounds its similarity field by [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) and calls [`assert_similarity_domain`](@ref) before it transforms, so a shipped configuration that would fail here fails earlier, at construction or at the seam, with a message that names the configuration rather than `W`.

These two checks are kept for the case those cannot cover: that family is open **by declaration**, so an extension can subtype it and return a negative anyway. The failure downstream is silent — `DirectHb` sums signed mass and a cancelling row manufactures a separating bubble — so a wrong clustering would come back with no error at all.

# Returns

  - `A::SparseMatrixCSC{<:Number, Int}`: Adjacency matrix of the PMFG with weights.
  - `tri::Matrix{Int}`: List of triangles (triangular faces) in the PMFG.
  - `clique3::Matrix{Int}`: List of 3-cliques that are not triangular faces; all 3-cliques are given by `[tri; clique3]`.
  - `cliques::Option{Matrix{Int}}`: List of all 4-cliques (tetrahedra), or `nothing` if `nargout <= 3`.
  - `cliqueTree::Option{SparseMatrixCSC{Int, Int}}`: 4-cliques tree structure (adjacency matrix), or `nothing` if `nargout <= 4`.

# Related

  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
  - [`LoGo`](@ref)

# References

  - $(ref_dict[:PMFG])
  - $(ref_dict[:tumminello2005])
"""
function PMFG_T2s(W::MatNum, nargout::Integer = 3)
    N = size(W, 1)
    @argcheck(9 <= N, DimensionMismatch("9 <= size(W, 1) must hold. Got\nsize(W, 1) => $N"))
    # Split in two, because `0 <= NaN` is `false`: one check would report a `NaN` as a
    # negative weight and send the caller looking for the wrong thing.
    @argcheck(!any(isnan, W),
              DomainError("!any(isnan, W) must hold. Got\ncount(isnan, W) => $(count(isnan, W))."))
    @argcheck(all(x -> zero(x) <= x, W),
              DomainError("all(x -> x >= 0, W) must hold. Got\nminimum(W) => $(minimum(W))."))
    A = SparseArrays.spzeros(Int, N, N)  # Initialize adjacency matrix
    in_v = zeros(Int, N)    # Initialize list of inserted vertices
    tri = zeros(Int, 2 * N - 4, 3)  # Initialize list of triangles
    clique3 = zeros(Int, N - 4, 3)   # Initialize list of 3-cliques (non-face triangles)

    # Find 3 vertices with largest strength
    s = sum(W ⊙ (W .> Statistics.mean(W)); dims = 2)
    j = sortperm(vec(s); rev = true)

    in_v[1:4] = j[1:4]
    ou_v = sort!(setdiff(1:N, in_v))  # List of vertices not inserted yet

    # Build the tetrahedron with largest strength
    tri[1, :] = in_v[[1, 2, 3]]
    tri[2, :] = in_v[[2, 3, 4]]
    tri[3, :] = in_v[[1, 2, 4]]
    tri[4, :] = in_v[[1, 3, 4]]
    A[in_v[1], in_v[2]] = 1
    A[in_v[1], in_v[3]] = 1
    A[in_v[1], in_v[4]] = 1
    A[in_v[2], in_v[3]] = 1
    A[in_v[2], in_v[4]] = 1
    A[in_v[3], in_v[4]] = 1

    # Build initial gain table
    gain = zeros(eltype(W), N, 2 * N - 4)
    gain[ou_v, 1] = sum(W[ou_v, tri[1, :]]; dims = 2)
    gain[ou_v, 2] = sum(W[ou_v, tri[2, :]]; dims = 2)
    gain[ou_v, 3] = sum(W[ou_v, tri[3, :]]; dims = 2)
    gain[ou_v, 4] = sum(W[ou_v, tri[4, :]]; dims = 2)

    kk = 4  # Number of triangles
    for k in 5:N
        # Find best vertex to add in a triangle
        if length(ou_v) == 1  # Special case for the last vertex
            ve = ou_v[1]
            v = 1
            agm = argmax(vec(gain[ou_v, :]))
        else
            gij, v = findmax(gain[ou_v, :]; dims = 1)
            v = vec(getindex.(v, 1))
            agm = argmax(vec(gij))
            ve = ou_v[v[agm]]
            v = v[agm]
        end

        # Update vertex lists
        ou_v = ou_v[deleteat!(collect(1:length(ou_v)), v)]
        # vcat(ou_v[1:(v - 1)], ou_v[(v + 1):end])
        in_v[k] = ve

        # Update adjacency matrix
        A[ve, tri[agm, :]] .= 1

        # Update 3-clique list
        clique3[k-4, :] = tri[agm, :]

        # Update triangle list replacing 1 and adding 2 triangles
        tri[kk+1, :] = vcat(tri[agm, [1, 3]], ve) # add
        tri[kk+2, :] = vcat(tri[agm, [2, 3]], ve) # add
        tri[agm, :] = vcat(tri[agm, [1, 2]], ve)     # replace

        # # Update gain table
        gain[ve, :] .= 0
        gain[ou_v, agm] = sum(W[ou_v, tri[agm, :]]; dims = 2)
        gain[ou_v, kk+1] = sum(W[ou_v, tri[kk+1, :]]; dims = 2)
        gain[ou_v, kk+2] = sum(W[ou_v, tri[kk+2, :]]; dims = 2)

        # # Update number of triangles
        kk += 2
    end

    A = SparseArrays.sparse(W ⊙ ((A + A') .== 1))

    cliques = nothing
    cliqueTree = nothing

    if nargout > 3
        cliques = vcat(transpose(in_v[1:4]), hcat(clique3, in_v[5:end]))
    end

    if nargout > 4
        M = size(cliques, 1)
        cliqueTree = SparseArrays.spzeros(Int, M, M)
        ss = zeros(Int, M)
        for i in axes(cliques, 1)
            ss .= 0
            for j in 1:3
                ss .+= vec(sum(cliques .== cliques[i, j]; dims = 2))
            end
            cliqueTree[i, ss .== 2] .= 1
        end
    end

    return A, tri, clique3, cliques, cliqueTree
end
"""
    assert_pmfg_weights(A::MatNum,
                        sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing,
                        de::Option{<:AbstractDistanceEstimator} = nothing)

Check that the weights did not delete an edge from the graph [`PMFG_T2s`](@ref) built.

A maximal planar graph on `N >= 3` vertices has exactly `3N - 6` edges, and [`PMFG_T2s`](@ref) returns the structure and the weights in one matrix, `A = W ⊙ ((A + A') .== 1)`. An **exactly zero** weight is therefore an *absent* edge rather than a weak one, and what reaches the consumer is no longer a PMFG. This function counts the stored edges and refuses the difference.

# The zero is admissible input and an unusable structure

[`PMFG_T2s`](@ref)'s own check is `>= 0` and stays that way, because a zero is an honest similarity. [`ExponentialSimilarity`](@ref) maps the infinite distance [`LogDistance`](@ref) returns at an exactly zero correlation to `exp(-Inf)`, which is `0` exactly, and [`ComplementSimilarity`](@ref) maps `D = 1` to `0`. The value is right. What it cannot do is carry an edge.

Without this check the failure is a `BoundsError` about a matrix index, raised much later inside [`turn_into_Hclust_merges`](@ref), because [`HierarchyConstruct4s`](@ref) then builds fewer merges than the dendrogram needs.

# Where it runs, and where it deliberately does not

At the three sites that consume the **weighted** structure: [`DBHTs`](@ref), [`calc_weighted_adjacency_graph`](@ref) and [`calc_distance_weighted_graph`](@ref).

[`logo!`](@ref) is the fourth [`PMFG_T2s`](@ref) caller and is **not** guarded. It reads separators and cliques, which [`PMFG_T2s`](@ref) derives from the insertion order rather than from `A`, so a zero weight does not change its answer and refusing it would refuse a configuration that works.

# Algorithm

 1. Count the stored non-zero entries of `A` and halve them, giving `edges`. `A` is symmetric, so each edge is stored twice.
 2. Build `source`, the part of the message that names the configuration, from as much of `sim` and `de` as the caller passed.
 3. Raise a `DomainError` when `edges` is not `expected`, which is `3N - 6`.

# Arguments

  - `A`: `N × N` weighted adjacency matrix, the first output of [`PMFG_T2s`](@ref).
  - `sim`: Similarity matrix algorithm that produced the weights, named in the message. Read for nothing else, as [`assert_similarity_domain`](@ref) reads its `de`.
  - `de`: Distance estimator the similarity was derived from, named in the message beside `sim`.

Each caller passes what it holds, so the message names as much of the configuration as the site knows. [`calc_distance_weighted_graph`](@ref) holds both halves, [`calc_weighted_adjacency_graph`](@ref) and [`DBHTs`](@ref) hold the similarity, and a caller that holds only the matrices names neither.

# Validation

  - The number of stored edges is `3N - 6`.

# Returns

  - `nothing`.

# Related

  - [`PMFG_T2s`](@ref)
  - [`DBHTs`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
"""
function assert_pmfg_weights(A::MatNum,
                             sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing,
                             de::Option{<:AbstractDistanceEstimator} = nothing)::Nothing
    N = size(A, 1)
    edges = count(!iszero, A) ÷ 2
    expected = 3 * N - 6
    source = if isnothing(sim)
        ""
    elseif isnothing(de)
        " for $(nameof(typeof(sim)))"
    else
        " for $(nameof(typeof(sim))), from $(typeof(de))"
    end
    @argcheck(edges == expected,
              DomainError(edges,
                          "count(!iszero, A) / 2 == 3 * size(A, 1) - 6 must hold$source. Got\nedges => $edges\n3 * N - 6 => $expected\nAn exactly zero weight is an absent edge rather than a weak one, so the PMFG is missing $(expected - edges) of its edges and the structure is not a PMFG. Use a similarity that is strictly positive over this data."))
    return nothing
end
"""
    distance_wei(L::MatNum)

Compute the shortest weighted path lengths between all node pairs in a network.

This function computes the distance matrix containing the lengths of the shortest paths between all node pairs in a (possibly weighted) network, using Dijkstra's algorithm. An entry `[u, v]` represents the length of the shortest path from node `u` to node `v`. The average shortest path length is the characteristic path length of the network.

!!! note

    Based on a Matlab implementation by Mika Rubinov, Rick Betzel, and Andrea Avena.

# Mathematical definition

```math
\\begin{align}
D_{u,\\,v} &= \\underset{p \\in P(u,\\, v)}{\\min} \\sum_{(i,\\,j) \\in p} L_{i,\\,j}\\,, \\\\
B_{u,\\,v} &= \\left| p^{\\star}(u,\\, v) \\right|\\,.
\\end{align}
```

Where:

  - ``L_{i,\\,j}``: Connection length of the pair ``(i,\\, j)``, the entry of the input matrix.
  - ``P(u,\\, v)``: Set of the paths from vertex ``u`` to vertex ``v``.
  - ``p^{\\star}(u,\\, v)``: Shortest of those paths, and ``\\left| p^{\\star}(u,\\, v) \\right|`` its edge count.
  - ``D_{u,\\,v}``: Shortest path length from vertex ``u`` to vertex ``v``.
  - ``B_{u,\\,v}``: Edge count of that shortest path.

A pair that no path joins keeps ``D_{u,\\,v} = \\infty``, because the minimum over an empty set is an infinity.

# Algorithm

 1. Set every entry of `D` to `typemax`, its diagonal to zero, and every entry of `B` to zero.
 2. For each source `u`, mark every vertex temporary in `S`, copy `L` into `L1`, and set the frontier `V` to `[u]`.
 3. Make the frontier permanent in `S`, and zero the columns of `L1` at `V`, so no edge re-enters a settled vertex.
 4. For each vertex `v` of the frontier, read its remaining neighbours `T` and take, entry by entry, the smaller of `D[u, T]` and `D[u, v] + L1[v, T]`.
 5. Where the second of the two won, set `B[u, T]` to `B[u, v] + 1`, so the edge count follows the path that won.
 6. Take `minD`, the smallest entry of `D[u, :]` over the temporary vertices. Stop when no temporary vertex remains, and stop when `minD` is infinite, which leaves every unreachable vertex at `typemax`.
 7. Set the new frontier `V` to every vertex at distance `minD`, and repeat from step 3.

# Arguments

  - `L`: `N × N` directed or undirected connection-length matrix.

      + Lengths between disconnected nodes should be set to `Inf`.
      + Lengths on the main diagonal should be set to `0`.

!!! note

    The input matrix must be a connection-length matrix, typically obtained by mapping weights to lengths (e.g., inverse of a similarity or correlation matrix). In weighted networks, shortest weighted paths may traverse more edges than shortest binary paths.

# Returns

  - `D::Matrix{<:Number}`: `N × N` shortest weighted path length matrix. `D[u, v]` is the length of the shortest path from vertex `u` to vertex `v`.
  - `B::Matrix{Int}`: `N × N` matrix of the edge count of each shortest weighted path.

# Related

  - [`PMFG_T2s`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
"""
function distance_wei(L::MatNum)
    N = size(L, 1)
    D = fill(typemax(eltype(L)), N, N)
    D[LinearAlgebra.diagind(D)] .= 0  # Distance matrix
    B = zeros(Int, N, N)     # Number of edges matrix

    for u in axes(L, 1)
        S = fill(true, N)   # Distance permanence (true is temporary)
        L1 = copy(L)
        V = [u]
        while true
            S[V] .= false   # Distance u -> V is now permanent
            L1[:, V] .= 0   # No inside edges as already shortest
            SparseArrays.dropzeros!(L1)
            for v in V
                T = SparseArrays.findnz(L1[v, :])[1] # neighbours of shortest nodes
                d, wi = findmin(vcat(vcat(transpose(D[u, T]),
                                          transpose(D[u, v] .+ L1[v, T]))); dims = 1)
                wi = vec(getindex.(wi, 2))
                D[u, T] = vec(d)   # Smallest of old/new path lengths
                ind = T[wi .== 3]   # Indices of lengthened paths
                B[u, ind] .= B[u, v] + 1    # Increment number of edges in lengthened paths
            end

            dus = D[u, S]
            minD = !isempty(dus) ? minimum(dus) : eltype(D)[]

            # isempty: all nodes reached
            # isinf: some nodes cannot be reached
            if isempty(minD) || isinf(minD)
                break
            end

            V = findall(D[u, :] .== minD)
        end
    end

    return D, B
end
"""
    clique3(A::MatNum)

Computes the list of 3-cliques in a Maximal Planar Graph (MPG).

This function identifies all 3-cliques (triangles) in the adjacency matrix `A` of a MPG. It returns the candidate cliques, their edge indices, and a matrix listing all unique 3-cliques. Used internally in DBHT and related phylogenetic clustering algorithms.

# Algorithm

 1. Remove the diagonal of `A` and reduce it to a binary matrix.
 2. Form `A2 = A * A`, whose entry counts the paths of length two between a pair.
 3. Keep the upper triangle of the entries where `A2` and `A` are both non-zero, giving `P`. A stored entry of `P` is an edge whose two ends share at least one neighbour.
 4. Read the row and the column index of every stored entry of `P` into the two columns of `E`, one row per candidate edge.
 5. For each candidate edge, intersect the neighbourhoods of its two ends, giving `K3[n]`, the third vertices that close a triangle on it.
 6. Sort each triple `(E[n, 1], E[n, 2], K3[n][m])` and append it to `clique` when `clique` does not already hold it, so a triangle found from each of its three edges is stored once.
 7. Sort the rows of `clique` on its three columns, and drop the placeholder first row.

# Arguments

  - `A`: `N × N` adjacency matrix of a Maximal Planar Graph (MPG). A non-zero entry is an edge.

# Returns

  - `K3::Vector{Vector{Int}}`: Vector of vectors, each containing the indices of nodes forming a candidate 3-clique.
  - `E::Matrix{Int}`: Matrix with nonzero indices and entries of candidate cliques (edge pairs).
  - `clique::Matrix{Int}`: `Nc×3` matrix. Each row lists the three vertices of a unique 3-clique in the MPG.

# Related

  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleHierarchy`](@ref)
  - [`DBHT`](@ref)
"""
function clique3(A::MatNum)
    A = A - LinearAlgebra.Diagonal(A)
    A = A .!= 0
    A2 = A * A
    P = (A2 .!= 0) ⊙ (A .!= 0)
    P = SparseArrays.sparse(LinearAlgebra.UpperTriangular(P))
    r, c = SparseArrays.findnz(P .!= 0)[1:2]
    E = hcat(r, c)

    lr = length(r)
    N3 = Vector{Int}(undef, lr)
    K3 = Vector{Vector{Int}}(undef, lr)
    for n in eachindex(r)
        i = r[n]
        j = c[n]
        a = A[i, :] ⊙ A[j, :]
        idx = SparseArrays.findnz(a .!= 0)[1]
        K3[n] = idx
        N3[n] = length(idx)
    end

    clique = zeros(Int, 1, 3)
    for n in eachindex(r)
        temp = K3[n]
        for m in eachindex(temp)
            candidate = transpose(E[n, :])
            candidate = hcat(candidate, temp[m])
            sort!(candidate; dims = 2)
            a = clique[:, 1] .== candidate[1]
            b = clique[:, 2] .== candidate[2]
            c = clique[:, 3] .== candidate[3]
            check = a ⊙ b ⊙ c
            check = sum(check)

            if iszero(check)
                clique = vcat(clique, candidate)
            end
        end
    end

    isort = sortperm(collect(zip(clique[:, 1], clique[:, 2], clique[:, 3])))
    clique = clique[isort, :]
    clique = clique[2:size(clique, 1), :]

    return K3, E, clique
end
"""
    breadth(CIJ::MatNum, source::Integer)

Breadth-first search.

This function performs a breadth-first search (BFS) on a binary (directed or undirected) connection matrix, starting from a specified source vertex. It computes the shortest path distances from the source to all other vertices and records the predecessor (branch) for each node in the BFS tree. The tree holds one shortest path per reachable vertex, and not every shortest path, so `branch` reconstructs one route of minimum length rather than all of them.

!!! note

    Original implementation by Olaf Sporns, Indiana University, 2002/2007/2008.

# Algorithm

 1. Colour every vertex white, set every entry of `distance` to `Inf`, and set every entry of `branch` to zero.
 2. Colour `source` grey, set its distance to zero and its branch to `-1`, and put it in the queue `Q`.
 3. Take the head `u` of `Q`, and read its out-neighbours `ns` from the stored entries of row `u`.
 4. For each neighbour `v` whose distance is still zero, set it to `distance[u] + 1`. This is what records the distance of `source` to itself when the graph carries a self-loop.
 5. For each white neighbour `v`, colour it grey, set `distance[v]` to `distance[u] + 1`, set `branch[v]` to `u`, and append `v` to `Q`.
 6. Drop `u` from `Q` and colour it black. Repeat from step 3 until `Q` is empty.

# Arguments

  - `CIJ`: `N × N` binary (0/1) connection matrix representing the graph. Row `u` holds the out-neighbours of vertex `u`.
  - `source`: Index of the source vertex from which to start the search.

# Returns

  - `distance::VecNum`: `N × 1` vector of shortest path distances from the source to each vertex (`0` for the source itself, `Inf` for unreachable nodes).
  - `branch::Vector{Int}`: `N × 1` vector of predecessor indices for each vertex in the BFS tree (`-1` for the source, `0` for an unreachable vertex).

# Related

  - [`FindDisjoint`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function breadth(CIJ::MatNum, source::Integer)
    N = size(CIJ, 1)
    # Colours
    white = 0
    gray = 1
    black = 2
    # Initialise colours
    color = zeros(Int, N)
    # Initialise distances
    distance = fill(Inf, N)
    # Initialise branches
    branch = zeros(Int, N)
    # Start on vertex `source`
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = [source]
    # Keep going until the entire graph is explored
    while !isempty(Q)
        u = Q[1]
        ns = SparseArrays.findnz(CIJ[u, :])[1]
        for v in ns
            # This allows the `source` distance to itself to be recorded
            if all(x -> x == zero(x), distance[v])
                distance[v] = distance[u] + 1
            end
            if all(x -> x == white, color[v])
                color[v] = gray
                distance[v] = distance[u] + 1
                branch[v] = u
                Q = vcat(Q, v)
            end
        end
        Q = Q[2:length(Q)]
        color[u] = black
    end

    return distance, branch
end
"""
    FindDisjoint(Adj::MatNum, Cliq::VecNum)

Finds disjointed cliques in an adjacency matrix.

This function identifies nodes that are not adjacent to a given 3-clique in the adjacency matrix, and classifies all nodes into three groups: members of the clique, nodes in the same connected component as the clique, and nodes in a disjoint component.

# Algorithm

 1. Copy `Adj` into `Temp`, and collect in `IndxNot` every vertex that is not one of the three of `Cliq`.
 2. Zero the rows and the columns of `Temp` at `Cliq`, which cuts the clique out of the graph and separates the two sides it was joining.
 3. Run [`breadth`](@ref) from `IndxNot[1]`, giving `d`, and mark every vertex `d` left at an infinity with `-1`.
 4. Write `1` into `T` at every vertex marked `-1`, and `2` at every other vertex, so `2` is the side that holds `IndxNot[1]`.
 5. Write `0` into `T` at the three vertices of `Cliq`.

# Arguments

  - `Adj`: `N × N` adjacency matrix of the MPG. A non-zero entry is an edge.
  - `Cliq`: `3×1` vector of node indices forming a 3-clique.

# Returns

  - `T::Vector{Int}`: `N × 1` vector containing the adjacency number of each node:

      + `0` for nodes in the clique,
      + `1` for nodes in a disjoint component,
      + `2` for nodes in the same component as the clique.

  - `IndxNot::Vector{Int}`: `N × 1` vector of nodes with no adjacencies to the clique.

# Related

  - [`breadth`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleHierarchy`](@ref)
"""
function FindDisjoint(Adj::MatNum, Cliq::VecNum)
    N = size(Adj, 1)
    Temp = copy(Adj)
    T = zeros(Int, N)
    IndxTotal = 1:N
    IndxNot = findall(IndxTotal .!= Cliq[1] .&&
                          IndxTotal .!= Cliq[2] .&&
                          IndxTotal .!= Cliq[3])
    Temp[Cliq, :] .= 0
    Temp[:, Cliq] .= 0
    SparseArrays.dropzeros!(Temp)
    d = breadth(Temp, IndxNot[1])[1]
    d[isinf.(d)] .= -1
    d[IndxNot[1]] = 0
    Indx1 = d .== -1
    Indx2 = d .!= -1
    T[Indx1] .= 1
    T[Indx2] .= 2
    T[Cliq] .= 0
    return T, IndxNot
end
"""
    BuildHierarchy(M::MatNum)

Builds the predicted parent hierarchy for 3-cliques in a Maximal Planar Graph (MPG).

This function constructs the parent index vector (`Pred`) for each 3-clique, given the node-to-clique membership matrix `M`. It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline, enabling the construction of the clique hierarchy tree.

# Algorithm

 1. For each 3-clique `n`, read `Children`, the vertices that column `n` of `M` marks.
 2. Sum the rows of `M` over `Children`, and take as `Parents` every clique whose sum equals `length(Children)`. Such a clique holds every vertex of clique `n`, so it is a superset of it. Drop `n` itself from that list.
 3. Set `Pred[n] = 0` when `Parents` is empty, which makes clique `n` a root.
 4. Otherwise take the parent of the smallest vertex count, which is the smallest superset.
 5. Return an empty vector when two parents tie on the smallest count, which reports that no hierarchy was built. The loop stops at that point, so no later clique writes to the empty vector.

# Arguments

  - `M`: `N × Nc` binary matrix of node-to-3-clique memberships, where `M[i, n] = 1` if node `i` belongs to 3-clique `n`.

# Returns

  - `Pred::Vector{Int}`: `Nc×1` vector of predicted parent indices for each 3-clique. `Pred[n] = 0` indicates a root clique. It is empty when step 5 of the algorithm fired.

# Related

  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleHierarchy`](@ref)
  - [`DBHT`](@ref)
"""
function BuildHierarchy(M::MatNum)
    N = size(M, 2)
    Pred = zeros(Int, N)
    SparseArrays.dropzeros!(M)
    for n in axes(M, 2)
        Children = SparseArrays.findnz(M[:, n] .== 1)[1]
        ChildrenSum = vec(sum(M[Children, :]; dims = 1))
        Parents = findall(ChildrenSum .== length(Children))
        Parents = Parents[Parents .!= n]
        if !isempty(Parents)
            ParentSum = vec(sum(M[:, Parents]; dims = 1))
            a = findall(ParentSum .== minimum(ParentSum))
            if length(a) != 1
                return Int[]
            end
            Pred[n] = Parents[a[1]]
        else
            Pred[n] = 0
        end
    end
    return Pred
end
"""
    AdjCliq(A::MatNum, CliqList::MatNum,
            CliqRoot::VecNum)

Find adjacent cliques to the root candidates in a Maximal Planar Graph (MPG).

This function computes the adjacency matrix among root candidate 3-cliques. Two root candidates are adjacent when they share exactly two vertices. Used internally by [`CliqueRoot`](@ref) with [`EqualRoot`](@ref) to construct a root from the adjacency tree of all root candidates.

`A` is read for its size and for nothing else, and no edge of the MPG reaches the answer. Nothing is lost by that. Every row of `CliqList` is a 3-clique of the MPG, so two rows that share two vertices both hold the edge between those two vertices. The count of the shared vertices is therefore the test for adjacency in the graph.

# Algorithm

 1. Clear `Indicator`, then mark in it the three vertices of root candidate `n`.
 2. Read `Indicator` back at the three vertex columns of every root candidate, giving `Indi`.
 3. Take the root candidates whose row of `Indi` sums to `2`, and set their entries of column `CliqRoot[n]` of `Adj` to one.
 4. Repeat from step 1 for the next candidate. The test of step 3 is symmetric, so `Adj` needs no symmetrisation.

# Arguments

  - `A`: `N × N` adjacency matrix of the MPG. Only `size(A, 1)` is read, which sets the length of `Indicator`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.
  - `CliqRoot`: Vector of indices of root candidate cliques, indexing the rows of `CliqList`.

# Returns

  - `Adj::SparseMatrixCSC{Int, Int}`: `Nc×Nc` symmetric adjacency matrix of the cliques. `Adj[i, j]` is one when cliques `i` and `j` are both root candidates and share exactly two vertices. Every other entry is zero, so a clique that is not a root candidate carries an empty row and an empty column.

# Related

  - [`CliqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function AdjCliq(A::MatNum, CliqList::MatNum, CliqRoot::VecNum)
    Nc = size(CliqList, 1)
    N = size(A, 1)
    Adj = SparseArrays.spzeros(Int, Nc, Nc)
    Indicator = zeros(Int, N)
    for n in eachindex(CliqRoot)
        Indicator .= 0
        Indicator[CliqList[CliqRoot[n], :]] .= 1
        Indi = hcat(Indicator[CliqList[CliqRoot, 1]], Indicator[CliqList[CliqRoot, 2]],
                    Indicator[CliqList[CliqRoot, 3]])

        adjacent = CliqRoot[vec(sum(Indi; dims = 2)) .== 2]
        Adj[adjacent, CliqRoot[n]] .= 1
    end

    return Adj
end
"""
    BubbleHierarchy(Pred::VecNum, Sb::VecNum)

Build the bubble hierarchy from the clique hierarchy and separating set information.

This function constructs the bubble hierarchy tree and the bubble membership matrix for 3-cliques, given the predicted parent indices (`Pred`) and separating set vector (`Sb`). It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline, grouping 3-cliques into bubbles and building the adjacency structure among bubbles.

# Algorithm

 1. Take `Root`, the cliques whose entry of `Pred` is zero, and mark them in `CliqCount`.
 2. When more than one root exists, open one bubble that holds all of them, as the first column of `Mb`.
 3. For each root `n`, open a bubble that holds `n` and its direct children — the cliques whose parent is `n` — append it as a column of `Mb`, and mark those children in `CliqCount`.
 4. Collect as the next roots the direct children whose separating set is non-empty, `Sb[.] != 0`.
 5. Repeat from step 3 until `CliqCount` marks every clique.
 6. Build `H`: two bubbles are neighbours when at least one 3-clique belongs to both.
 7. Symmetrise `H` and clear its diagonal, so a bubble is not its own neighbour.

# Arguments

  - `Pred`: `Nc×1` vector of predicted parent indices for each 3-clique, as returned by [`BuildHierarchy`](@ref).
  - `Sb`: `Nc×1` vector indicating the size of the separating set for each 3-clique (`Sb[n] ≠ 0` means clique `n` is separating).

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `Nb×Nb` symmetric adjacency matrix representing the bubble hierarchy tree, where `Nb` is the number of bubbles.
  - `Mb::Matrix{Int}`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicates that 3-clique `n` belongs to bubble `bi`.

# Related

  - [`BuildHierarchy`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
"""
function BubbleHierarchy(Pred::VecNum, Sb::VecNum)
    Nc = size(Pred, 1)
    Root = findall(Pred .== 0)
    CliqCount = zeros(Int, Nc)
    CliqCount[Root] .= 1
    Mb = Matrix{Int}(undef, Nc, 0)

    if length(Root) > 1
        TempVec = zeros(Int, Nc)
        TempVec[Root] .= 1
        Mb = hcat(Mb, TempVec)
    end

    while sum(CliqCount) < Nc
        NxtRoot = Int[]
        for n in eachindex(Root)
            DirectChild = findall(Pred .== Root[n])
            TempVec = zeros(Int, Nc)
            TempVec[[Root[n]; DirectChild]] .= 1
            Mb = hcat(Mb, TempVec)
            CliqCount[DirectChild] .= 1

            for m in eachindex(DirectChild)
                if Sb[DirectChild[m]] != 0
                    NxtRoot = [NxtRoot; DirectChild[m]]
                end
            end
        end
        Root = sort!(unique(NxtRoot))
    end
    Nb = size(Mb, 2)
    H = SparseArrays.spzeros(Int, Nb, Nb)

    for n in axes(Mb, 2)
        Indx = Mb[:, n] .== 1
        JointSum = vec(sum(Mb[Indx, :]; dims = 1))
        Neigh = JointSum .>= 1
        H[n, Neigh] .= 1
    end

    H = H + transpose(H)
    H = H - LinearAlgebra.Diagonal(H)
    return H, Mb
end
"""
    CliqueRoot(::UniqueRoot, Root::VecNum, Pred::VecNum, Nc::Integer, args...)

Construct the hierarchical adjacency matrix for 3-cliques in a Maximal Planar Graph (MPG) using the unique root selection method.

This method enforces a unique root in the clique hierarchy. If multiple root candidates are present, a synthetic root is created and all root candidates are attached to it. Used internally by [`CliqHierarchyTree2s`](@ref) when the root selection method is [`UniqueRoot`](@ref).

# Algorithm

 1. When more than one root candidate exists, append a synthetic clique to `Pred` and set the parent of every root candidate to it. `Pred` is mutated in place, so the caller's vector gains that entry.
 2. Allocate `H` over `Nc + 1` rows and columns, which is the room the synthetic clique of step 1 needs.
 3. Write `H[n, Pred[n]] = 1` for every clique that has a parent.
 4. Symmetrise `H`.

# Arguments

  - `::UniqueRoot`: Root selection method enforcing a unique root.
  - `Root`: Vector of indices of root candidate cliques, indexing the entries of `Pred`.
  - `Pred`: `Nc×1` vector of predicted parent indices for each clique.
  - `Nc`: Number of 3-cliques.
  - `args...`: Additional arguments (ignored for this method). [`CliqHierarchyTree2s`](@ref) passes the adjacency matrix and the clique list here, which the [`EqualRoot`](@ref) method reads and this one does not.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `(Nc + 1)×(Nc + 1)` symmetric adjacency matrix representing the hierarchical tree of 3-cliques. Row and column `Nc + 1` hold the synthetic root, and they are empty when step 1 of the algorithm did not fire.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`CliqueRoot`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function CliqueRoot(::UniqueRoot, Root::VecNum, Pred::VecNum, Nc::Integer, args...)
    if length(Root) > 1
        push!(Pred, 0)
        Pred[Root] .= length(Pred)
    end

    H = SparseArrays.spzeros(Int, Nc + 1, Nc + 1)
    for n in eachindex(Pred)
        if Pred[n] != 0
            H[n, Pred[n]] = 1
        end
    end
    return H = H + transpose(H)
end
"""
    CliqueRoot(::EqualRoot, Root::VecNum, Pred::VecNum, Nc::Integer,
               A::MatNum, CliqList::MatNum)

Construct the hierarchical adjacency matrix for 3-cliques in a Maximal Planar Graph (MPG) using the equal root selection method.

This method creates a root from the adjacency tree of all root candidate cliques, allowing for multiple equally plausible roots in the DBHT hierarchy. It is used internally by [`CliqHierarchyTree2s`](@ref) when the root selection method is [`EqualRoot`](@ref).

# Algorithm

 1. When more than one root candidate exists, build the adjacency `Adj` among the candidates with [`AdjCliq`](@ref). One candidate alone has nothing to be joined to, so `Adj` is a zero matrix in that case, which makes step 4 add nothing.
 2. Allocate `H` over `Nc` rows and columns. This method adds no synthetic clique, so it needs no extra row.
 3. Write `H[n, Pred[n]] = 1` for every clique that has a parent.
 4. Return a `0 × 0` matrix when `Pred` is empty. Otherwise symmetrise `H` and add `Adj` to it, which joins the root candidates to each other.

# Arguments

  - `::EqualRoot`: Root selection method that creates a root from the adjacency tree of all root candidates.
  - `Root`: Vector of indices of root candidate cliques, indexing the entries of `Pred`.
  - `Pred`: `Nc×1` vector of predicted parent indices for each clique.
  - `Nc`: Number of 3-cliques.
  - `A`: `N × N` adjacency matrix of the MPG. It is forwarded to [`AdjCliq`](@ref), which reads its size alone.
  - `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `Nc×Nc` symmetric adjacency matrix representing the hierarchical tree of 3-cliques, or a `0 × 0` matrix when `Pred` is empty.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`CliqueRoot`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function CliqueRoot(::EqualRoot, Root::VecNum, Pred::VecNum, Nc::Integer, A::MatNum,
                    CliqList::MatNum)
    Adj = if length(Root) > 1
        AdjCliq(A, CliqList, Root)
    else
        SparseArrays.spzeros(Int, Nc, Nc)
    end

    H = SparseArrays.spzeros(Int, Nc, Nc)
    for n in eachindex(Pred)
        if Pred[n] != 0
            H[n, Pred[n]] = 1
        end
    end

    return if !isempty(Pred)
        H .+= transpose(H)
        H .+= Adj
    else
        H = SparseArrays.spzeros(Int, 0, 0)
    end
end
"""
    CliqHierarchyTree2s(Apm::MatNum, root::DBHTRootMethod = UniqueRoot())

Construct the clique and bubble hierarchy trees for a Maximal Planar Graph (MPG) using the DBHT (Direct Bubble Hierarchical Tree) approach.

This function builds the hierarchical structure of 3-cliques (triangles) and bubbles from the adjacency matrix of a planar graph, supporting different root selection strategies via the `root` argument. It is a core routine for DBHT clustering and related phylogenetic analyses.

`root` is a **positional** argument, and every caller passes it positionally.

# Algorithm

 1. Reduce `Apm` to the binary adjacency `A`, and list every 3-clique of it with [`clique3`](@ref), giving `CliqList`.
 2. For each 3-clique, cut it out of the graph with [`FindDisjoint`](@ref), which splits the vertices into the clique `indx0`, the side `indx1` that the cut separated, and the side `indx2` that stayed connected.
 3. Take the smaller of the two sides, together with the clique, as the separated set `indx_s`. A tie takes `indx1`, the side the cut separated. Record `length(indx_s) - 3` in `Sb[n]`, and mark the vertices of `indx_s` in column `n` of `M`.
 4. Build the parent vector `Pred` from `M` with [`BuildHierarchy`](@ref), and read the root candidates `Root` off it.
 5. Build the clique hierarchy `H` with [`CliqueRoot`](@ref), through the branch `root` selects.
 6. When `H` is non-empty, build the bubble hierarchy `H2` and the bubble membership `Mb` with [`BubbleHierarchy`](@ref), reduce `H2` to binary, and trim `Mb` to the rows of `CliqList`.
 7. Return `0 × 0` matrices for `H2` and `Mb` when `H` is empty.

# Arguments

  - `Apm`: `N × N` adjacency matrix of the MPG, where nonzero entries indicate edges. Only the sparsity pattern is read, so a weighted matrix and its binary form give the same answer.
  - `root`: Root selection method for the clique hierarchy.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix representing the hierarchical tree of 3-cliques. Its size is set by the [`CliqueRoot`](@ref) method that `root` selects.
  - `H2::SparseMatrixCSC{Int, Int}`: `Nb×Nb` symmetric adjacency matrix representing the bubble hierarchy tree, where `Nb` is the number of bubbles.
  - `Mb::Matrix{Int}`: Bubble membership matrix for 3-cliques (`Nc×Nb`), where `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `CliqList::Matrix{Int}`: List of 3-cliques (`Nc×3`), each row contains the vertex indices of a 3-clique.
  - `Sb::Vector{Int}`: `Nc×1` vector indicating the size of the separating set for each 3-clique.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`PMFG_T2s`](@ref)

# References

  - $(ref_dict[:NHPG])
"""
function CliqHierarchyTree2s(Apm::MatNum, root::DBHTRootMethod = UniqueRoot())
    N = size(Apm, 1)
    A = Apm .!= 0
    K3, E, clique = clique3(A)

    Nc = size(clique, 1)
    M = SparseArrays.spzeros(Int, N, Nc)
    CliqList = copy(clique)
    Sb = zeros(Int, Nc)

    for n in axes(clique, 1)
        cliq_vec = CliqList[n, :]
        T = FindDisjoint(A, cliq_vec)[1]
        indx0 = findall(T .== 0)
        indx1 = findall(T .== 1)
        indx2 = findall(T .== 2)

        indx_s = length(indx1) > length(indx2) ? vcat(indx2, indx0) : vcat(indx1, indx0)

        Sb[n] = !isempty(indx_s) ? length(indx_s) - 3 : 0

        M[indx_s, n] .= 1
    end

    Pred = BuildHierarchy(M)
    Root = findall(Pred .== 0)

    H = CliqueRoot(root, Root, Pred, Nc, A, CliqList)

    if !isempty(H)
        H2, Mb = BubbleHierarchy(Pred, Sb)
        H2 = H2 .!= 0
        Mb = Mb[1:size(CliqList, 1), :]
    else
        H2 = SparseArrays.spzeros(Int, 0, 0)
        Mb = SparseArrays.spzeros(Int, 0, 0)
    end

    return H, H2, Mb, CliqList, Sb
end
"""
    DirectHb(Rpm::MatNum, Hb::MatNum,
             Mb::MatNum, Mv::MatNum,
             CliqList::MatNum)

Compute the directed bubble hierarchy tree (DBHT) for a Maximal Planar Graph (MPG).

This function assigns directions to each separating 3-clique in the undirected bubble tree of a Planar Maximally Filtered Graph (PMFG), producing the directed bubble hierarchy tree (DBHT). The direction is determined by comparing the sum of edge weights on either side of each separating clique, enabling the identification of converging and diverging bubbles.

# Mathematical definition

Each edge of the bubble tree carries one separating 3-clique. Cutting the edge splits the bubbles into two sides, and the mass each side draws through the clique decides the direction.

```math
m(\\mathcal{V}) = \\sum_{u \\in \\mathcal{V}_{0}} \\sum_{v \\in \\mathcal{V}} R_{u,\\,v}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``\\mathcal{V}_{0}``: The three vertices of the separating clique.
  - ``\\mathcal{V}``: Vertices of one side of the cut, with ``\\mathcal{V}_{0}`` removed.
  - ``m(\\mathcal{V})``: Mass the clique draws from that side.

The edge is directed towards the heavier side, so a bubble that draws mass from both of its neighbours has no outgoing edge and is a converging bubble.

# Algorithm

 1. Reduce `Hb` to binary, and read the row and column index of each edge of its upper triangle.
 2. For each such edge, find the 3-cliques that both of its bubbles hold, and record `(row, column, clique)` as a row of `CliqEdge`.
 3. For each row of `CliqEdge`, remove that edge from a copy of `Hb`, run [`breadth`](@ref) from bubble `1`, and mark every bubble it did not reach with `-1`.
 4. Split the two bubbles of the edge into `bleft`, the one on the reached side, and `bright`, the one on the cut side.
 5. Collect `vleft` and `vright`, the vertices of the bubbles of each side, and remove from both the three vertices `vo` of the separating clique.
 6. Sum the PMFG weights from `vo` into each side, giving `left` and `right`, and write the heavier of the two into `Hc` as an edge directed towards the heavier side.
 7. Set `Sep[b] = 1` for a bubble with no outgoing edge in `Hc`, then set `Sep[b] = 2` for a bubble with no incoming edge that has more than one neighbour in `Hb`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Hb`: `Nb×Nb` undirected bubble tree of the PMFG (as from [`BubbleHierarchy`](@ref)). A non-zero entry joins two bubbles.
  - `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N × Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.

# Returns

  - `Hc::SparseMatrixCSC{Number, Int}`: `Nb×Nb` directed adjacency matrix of the DBHT. A non-zero `Hc[i, j]` is a directed edge from bubble `i` to bubble `j`, and its value is the mass of the heavier side.
  - `Sep::Vector{Int}`: `Nb×1` vector of the type of each bubble. `1` is a converging bubble, which has no outgoing edge in `Hc`. `2` is a diverging bubble, which has no incoming edge and more than one neighbour in `Hb`. `0` is every other bubble.

# Related

  - [`BubbleHierarchy`](@ref)
  - [`BubbleCluster8s`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
"""
function DirectHb(Rpm::MatNum, Hb::MatNum, Mb::MatNum, Mv::MatNum, CliqList::MatNum)
    Hb = Hb .!= 0
    r, c, _ = SparseArrays.findnz(SparseArrays.sparse(LinearAlgebra.UpperTriangular(Hb) .!=
                                                      0))
    CliqEdge = Matrix{Int}(undef, 0, 3)
    for n in eachindex(r)
        data = findall(Mb[:, r[n]] .!= 0 .&& Mb[:, c[n]] .!= 0)
        data = hcat(r[n], c[n], data)
        CliqEdge = vcat(CliqEdge, data)
    end

    kb = vec(sum(Hb; dims = 1))
    sMv = size(Mv, 2)
    Hc = SparseArrays.spzeros(sMv, sMv)

    sCE = size(CliqEdge, 1)
    for n in axes(CliqEdge, 1)
        Temp = copy(Hb)
        Temp[CliqEdge[n, 1], CliqEdge[n, 2]] = 0
        Temp[CliqEdge[n, 2], CliqEdge[n, 1]] = 0
        SparseArrays.dropzeros!(Temp)
        d, _ = breadth(Temp, 1)
        d[isinf.(d)] .= -1
        d[1] = 0

        vo = CliqList[CliqEdge[n, 3], :]
        b = CliqEdge[n, 1:2]
        bleft = b[d[b] .!= -1]
        bright = b[d[b] .== -1]

        vleft = getindex.(findall(Mv[:, d .!= -1] .!= 0), 1)
        vleft = setdiff(vleft, vo)

        vright = getindex.(findall(Mv[:, d .== -1] .!= 0), 1)
        vright = setdiff(vright, vo)

        left = sum(Rpm[vo, vleft])
        right = sum(Rpm[vo, vright])

        left > right ? Hc[bright, bleft] .= left : Hc[bleft, bright] .= right
    end

    Sep = vec(Int.(iszero.(sum(Hc; dims = 2))))
    Sep[vec(iszero.(sum(Hc; dims = 1))).&&kb .> 1] .= 2

    return Hc, Sep
end
"""
    BubbleCluster8s(Rpm::MatNum, Dpm::MatNum,
                    Hb::MatNum, Mb::MatNum,
                    Mv::MatNum, CliqList::MatNum)

Obtain non-discrete and discrete clusterings from the bubble topology of the Planar Maximally Filtered Graph (PMFG).

This function assigns each vertex to a cluster based on the directed bubble hierarchy tree (DBHT) structure. It computes both a non-discrete cluster membership matrix and a discrete cluster assignment vector, using the converging bubbles identified in the directed bubble tree.

# Mathematical definition

A vertex that more than one converging bubble holds is given to the bubble whose edges bind it most tightly, per edge of that bubble.

```math
\\chi(v,\\, b) = \\frac{\\displaystyle\\sum_{u \\in b} R_{u,\\,v}}{3\\left(\\left|b\\right| - 2\\right)}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``b``: Vertex set of a converging bubble, and ``\\left|b\\right|`` its vertex count.
  - ``\\chi(v,\\, b)``: Association of vertex ``v`` with bubble ``b``.

The denominator is the edge count of a maximal planar graph on ``\\left|b\\right|`` vertices, which is the same ``3n - 6`` the PMFG itself carries. It divides out the size of the bubble, so a large bubble does not win on its size alone.

# Algorithm

 1. Direct the bubble tree with [`DirectHb`](@ref), giving `Hc` and `Sep`.
 2. Take `indx`, the converging bubbles, `Sep .== 1`. When one or none exists, put every vertex in cluster `1`, leave `Adjv` at `0 × 0`, and stop.
 3. For each converging bubble, run [`breadth`](@ref) on the transpose of `Hc`, and mark in column `n` of `Adjv` every vertex of every bubble it reaches. A vertex can be marked in more than one column, which is what makes `Adjv` non-discrete.
 4. Gather `Bubv`, the vertex membership of the converging bubbles alone. Copy into `Mdjv` the rows of `Bubv` for the vertices `cv` that exactly one converging bubble holds.
 5. For each vertex of `uv`, which more than one holds, take the converging bubble of largest ``\\chi`` and mark it in `Mdjv`.
 6. Read the discrete assignment `Tc` off the stored entries of `Mdjv`.
 7. For a vertex that no converging bubble holds, take the mean shortest path length `Udjv` to each converging bubble, block the bubbles that `Adjv` does not reach with `typemax`, and assign the closest of the rest.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Dpm`: `N × N` shortest path lengths matrix of the PMFG.
  - `Hb`: `Nb×Nb` undirected bubble tree of the PMFG (from [`BubbleHierarchy`](@ref)).
  - `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N × Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.

# Returns

  - `Adjv::SparseMatrixCSC{Int, Int}`: `N × Nk` cluster membership matrix for vertices for non-discrete clustering via the bubble topology, `Nk` being the number of converging bubbles. `Adjv[n, k] = 1` indicates cluster membership of vertex `n` to the `k`-th non-discrete cluster, and a vertex can belong to more than one. It is `0 × 0` when step 2 of the algorithm stopped.
  - `Tc::Vector{Int}`: `N × 1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`-th discrete cluster. Every vertex carries exactly one.

# Related

  - [`DirectHb`](@ref)
  - [`BubbleHierarchy`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
"""
function BubbleCluster8s(Rpm::MatNum, Dpm::MatNum, Hb::MatNum, Mb::MatNum, Mv::MatNum,
                         CliqList::MatNum)
    Hc, Sep = DirectHb(Rpm, Hb, Mb, Mv, CliqList)   # Assign directions on the bubble tree

    N = size(Rpm, 1)    # Number of vertices in the PMFG
    indx = findall(Sep .== 1)   # Look for the converging bubbles
    Adjv = SparseArrays.spzeros(Int, 0, 0)

    SparseArrays.dropzeros!(Hc)
    lidx = length(indx)
    if lidx > 1
        Adjv = SparseArrays.spzeros(Int, size(Mv, 1), lidx)   # Set the non-discrete cluster membership matrix 'Adjv' at default

        # Identify the non-discrete cluster membership of vertices by each converging bubble
        for n in eachindex(indx)
            d, _ = breadth(transpose(Hc), indx[n])
            d[isinf.(d)] .= -1
            d[indx[n]] = 0
            r = getindex.(findall(Mv[:, d .!= -1] .!= 0), 1)
            Adjv[unique(r), n] .= 1
        end

        Tc = zeros(Int, N)  # Set the discrete cluster membership vector at default
        Bubv = Mv[:, indx]  # Gather the list of vertices in the converging bubbles
        cv = findall(vec(sum(Bubv; dims = 2) .== 1))    # Identify vertices which belong to single converging bubbles
        uv = findall(vec(sum(Bubv; dims = 2) .> 1)) # Identify vertices which belong to more than one converging bubbles
        Mdjv = SparseArrays.spzeros(N, lidx) # Set the cluster membership matrix for vertices in the converging bubbles at default
        Mdjv[cv, :] = Bubv[cv, :]   # Assign vertices which belong to single converging bubbles to the rightful clusters

        # Assign converging bubble membership of vertices in `uv'
        for v in eachindex(uv)
            v_cont = vec(sum(Rpm[:, uv[v]] ⊙ Bubv; dims = 1))  # sum of edge weights linked to uv(v) in each converging bubble
            all_cont = vec(3 * (sum(Bubv; dims = 1) .- 2))  # number of edges in converging bubble
            imx = argmax(v_cont ⊘ all_cont)    # computing chi(v,b_{alpha})
            Mdjv[uv[v], imx] = 1    # Pick the most strongly associated converging bubble
        end

        # Assign discrete cluster membership of vertices in the converging bubbles
        v, ci, _ = SparseArrays.findnz(Mdjv)
        Tc[v] .= ci

        # Compute the distance between a vertex and the converging bubbles
        Udjv = Dpm * Mdjv * LinearAlgebra.diagm(1 ⊘ vec(sum(Mdjv .!= 0; dims = 1)))
        Udjv[Adjv .== 0] .= typemax(eltype(Dpm))

        imn = vec(getindex.(argmin(Udjv[vec(sum(Mdjv; dims = 2)) .== 0, :]; dims = 2), 2))  # Look for the closest converging bubble
        Tc[Tc .== 0] .= imn # Assign discrete cluster membership according to the distances to the converging bubbles
    else
        Tc = ones(Int, N)   # If there is one converging bubble, all vertices belong to a single cluster
    end

    return Adjv, Tc
end
"""
    BubbleMember(Rpm::MatNum, Mv::MatNum,
                 Mc::MatNum)

Assign each vertex to a specific bubble in the bubble hierarchy.

This function determines the bubble membership of each vertex, resolving ambiguities when a vertex may belong to multiple bubbles. Assignment is based on the strength of connections (edge weights) between the vertex and each candidate bubble.

# Mathematical definition

A vertex that more than one bubble of the cluster holds is given to the bubble whose internal weight it carries the largest fraction of.

```math
\\phi(v,\\, b) = \\frac{\\displaystyle\\sum_{u \\in b} R_{u,\\,v}}{\\displaystyle\\frac{1}{2}\\sum_{u \\in b} \\sum_{u' \\in b} R_{u,\\,u'}}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``b``: Vertex set of a bubble.
  - ``\\phi(v,\\, b)``: Fraction of the internal weight of bubble ``b`` that vertex ``v`` draws.

The denominator is halved because the PMFG weights are symmetric and the double sum counts each edge twice. This differs from the ``\\chi`` of [`BubbleCluster8s`](@ref): that one divides by the edge **count** of a maximal planar bubble, and this one by the edge **weight** the bubble actually holds.

# Algorithm

 1. Split the vertices that `Mc` marks into `v`, held by exactly one bubble, and `vu`, held by more than one.
 2. Copy the rows of `Mc` at `v` into `Mvv`, which assigns them directly.
 3. For each vertex of `vu`, read its candidate bubbles `bub` off its row of `Mc`, score each with ``\\phi``, and mark the largest in `Mvv`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Mv`: `N × Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `Mc`: `N × Nb` bubble membership matrix restricted to one cluster. `Mc[n, bi] = 1` means vertex `n` is a vertex of bubble `bi` **and** belongs to that cluster. Every other entry is zero, so a bubble that the cluster does not reach carries an empty column.

# Returns

  - `Mvv::Matrix{Int}`: `N × Nb` matrix where `Mvv[n, bi] = 1` if vertex `n` is assigned to bubble `bi`. Each row of it carries at most one non-zero, which is what makes the assignment discrete.

# Related

  - [`HierarchyConstruct4s`](@ref)
  - [`BubbleHierarchy`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function BubbleMember(Rpm::MatNum, Mv::MatNum, Mc::MatNum)
    Mvv = zeros(Int, size(Mv, 1), size(Mv, 2))

    vu = findall(vec(sum(Mc; dims = 2) .> 1))
    v = findall(vec(sum(Mc; dims = 2) .== 1))

    Mvv[v, :] = Mc[v, :]

    for n in eachindex(vu)
        bub = findall(Mc[vu[n], :] .!= 0)
        vu_bub = vec(sum(Rpm[:, vu[n]] ⊙ Mv[:, bub]; dims = 1))
        all_bub = LinearAlgebra.diag(transpose(Mv[:, bub]) * Rpm * Mv[:, bub]) / 2
        frac = vu_bub ⊘ all_bub
        imx = vec(argmax(frac; dims = 1))
        Mvv[vu[n], bub[imx]] .= 1
    end

    return Mvv
end
"""
    DendroConstruct(Zi::MatNum, LabelVec1::VecNum,
                    LabelVec2::VecNum,
                    LinkageDist::Num_VecNum)

Construct the linkage matrix by continually adding rows to the matrix.

This function appends a new row to the linkage matrix at each iteration, recording the merge of clusters as indicated by changes in the label vectors. It is used internally for building dendrograms in DBHT and related hierarchical clustering routines.

# Algorithm

 1. Take `indx`, the vertices whose label differs between `LabelVec1` and `LabelVec2`. Those are the vertices the merge moved.
 2. Read the labels `LabelVec1` gave them, drop the repeats and sort them. A merge joins two labels, so exactly two survive.
 3. Append one row to `Zi`: those two labels, followed by `LinkageDist`.

# Arguments

  - `Zi`: `i × 3` linkage matrix at iteration `i` in the same format as the output from Matlab. Each row holds the two merged labels and the height of the merge.
  - `LabelVec1`: `N × 1` label vector for the vertices in the bubble for the previous valid iteration.
  - `LabelVec2`: `N × 1` label vector for the vertices in the bubble for the trial iteration.
  - `LinkageDist`: Height of the current merge, written into the third column.

# Returns

  - `Z::MatNum`: `(i + 1)×3` linkage matrix at iteration `i + 1` in the same format as the output from Matlab. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

# Related

  - [`HierarchyConstruct4s`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
"""
function DendroConstruct(Zi::MatNum, LabelVec1::VecNum, LabelVec2::VecNum,
                         LinkageDist::Num_VecNum)
    indx = LabelVec1 .!= LabelVec2
    Z = vcat(Zi, hcat(transpose(sort!(unique(LabelVec1[indx]))), LinkageDist))
    return Z
end
"""
    LinkageFunction(d::MatNum, labelvec::VecNum)

Find the pair of clusters of smallest union diameter in a bubble.

This function scores every pair of clusters that `labelvec` names by the diameter of their union under the distance matrix `d`, and returns the pair of smallest score. The diameter is the largest non-zero distance inside the union, so the score is a complete linkage. Used internally for hierarchical linkage construction in DBHT dendrogram routines.

# Mathematical definition

```math
\\begin{align}
\\delta(I,\\, J) &= \\underset{u,\\,v \\in \\mathcal{C}_{I} \\cup \\mathcal{C}_{J},\\; d_{u,\\,v} \\neq 0}{\\max}\\; d_{u,\\,v}\\,, \\\\
(I^{\\star},\\, J^{\\star}) &= \\underset{I < J}{\\arg\\min}\\; \\delta(I,\\, J)\\,.
\\end{align}
```

Where:

  - ``d_{u,\\,v}``: Distance between vertices ``u`` and ``v``, the entry of the input matrix.
  - ``\\mathcal{C}_{I}``: Vertices that carry label ``I``.
  - ``\\delta(I,\\, J)``: Diameter of the union of the two clusters.

The union is scored, not the cut between the two clusters, so a distance between two members of the **same** cluster can set ``\\delta``. A pair whose union carries no non-zero distance scores ``0``, which is the smallest score there is.

# Algorithm

 1. Take `lvec`, the sorted distinct labels of `labelvec`.
 2. For each pair `(r, c)` of labels with `r < c`, select the vertices that carry either label.
 3. Take the largest non-zero entry of the distance submatrix on those vertices, and record `(lvec[r], lvec[c], value)` as a row of `Links`. Record `0` as the value when the submatrix carries no non-zero entry.
 4. Take the row of smallest value, giving the pair `PairLink` and the score `dvu`.

# Arguments

  - `d`: `Nv×Nv` distance matrix for the vertices assigned to a bubble. Row and column `i` are the same vertex, and entry `i` of `labelvec` names its cluster.
  - `labelvec`: `Nv×1` label vector for the vertices in the bubble.

# Returns

  - `PairLink::Vector{Int}`: `2 × 1` vector of the two cluster labels of the selected pair.
  - `dvu::Number`: Diameter of the union of that pair, the smallest such value over every pair.

# Related

  - [`build_link_and_dendro`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function LinkageFunction(d::MatNum, labelvec::VecNum)
    lvec = sort!(unique(labelvec))
    Links = Matrix{Int}(undef, 0, 3)
    for r in 1:(length(lvec) - 1)
        vecr = labelvec .== lvec[r]
        for c in (r + 1):length(lvec)
            vecc = labelvec .== lvec[c]
            x1 = vecr .|| vecc
            dd = d[x1, x1]
            de = dd[dd .!= 0]
            Link1 = if !isempty(de)
                hcat(lvec[r], lvec[c], vec(maximum(de; dims = 1)))
            else
                hcat(lvec[r], lvec[c], 0)
            end
            Links = vcat(Links, Link1)
        end
    end
    dvu, imn = findmin(Links[:, 3])
    PairLink = Links[imn, 1:2]
    return PairLink, dvu
end
"""
    build_link_and_dendro(
        rg::AbstractRange,
        dpm::MatNum,
        LabelVec::VecNum,
        LabelVec1::VecNum,
        LabelVec2::VecNum,
        V::VecNum,
        nc::Number,
        Z::MatNum
    )

Iteratively construct the linkage matrix for a bubble or cluster.

This function iterates over the vertices in a bubble or cluster, merging the pair of clusters with the best linkage at each step (as determined by [`LinkageFunction`](@ref)), and appending the corresponding row to the linkage matrix using [`DendroConstruct`](@ref). Used internally for building dendrograms in DBHT and related hierarchical clustering routines.

# Algorithm

 1. Take the pair of smallest union diameter with [`LinkageFunction`](@ref) over `dpm` and `LabelVec`.
 2. Give both members of the pair the label `maximum(LabelVec1) + 1`, so the merged cluster takes a label no vertex carries yet.
 3. Write the merged labels back into `LabelVec2` at the vertices `V`.
 4. Append one row to `Z` with [`DendroConstruct`](@ref), at the height `1 / nc`.
 5. Subtract one from `nc`, and copy `LabelVec2` into `LabelVec1`.
 6. Repeat from step 1 once for each element of `rg`.

# Arguments

  - `rg`: Range whose **length** sets the number of merges. Its values are not read.
  - `dpm`: `Nv×Nv` distance matrix for the vertices assigned to the bubble or cluster, in the order of `V`.
  - `LabelVec`: `Nv×1` label vector of those vertices. It is mutated by step 2.
  - `LabelVec1`: `N × 1` label vector over every vertex, for the previous valid iteration.
  - `LabelVec2`: `N × 1` label vector over every vertex, for the trial iteration. It is mutated by step 3.
  - `V`: `Nv×1` vector of the indices of the vertices in the bubble or cluster, indexing the rows of `LabelVec1`.
  - `nc::Number`: Merge counter. Step 4 writes the height `1 / nc` and step 5 lowers it by one, so the heights of a run rise towards `1`.
  - `Z`: Current linkage matrix, with three columns.

# Returns

  - `Z::MatNum`: Linkage matrix after every merge of the range, one row longer per merge.
  - `nc::Number`: Merge counter, lowered by the number of merges.
  - `LabelVec1::VecNum`: `N × 1` label vector for the next iteration, carrying the merged labels.

# Related

  - [`LinkageFunction`](@ref)
  - [`DendroConstruct`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function build_link_and_dendro(rg::AbstractRange, dpm::MatNum, LabelVec::VecNum,
                               LabelVec1::VecNum, LabelVec2::VecNum, V::VecNum, nc::Number,
                               Z::MatNum)
    for _ in rg
        PairLink, dvu = LinkageFunction(dpm, LabelVec)  # Look for the pair of clusters which produces the best linkage
        LabelVec[LabelVec .== PairLink[1].||LabelVec .== PairLink[2]] .= maximum(LabelVec1) +
                                                                         1  # Merge the cluster pair by updating the label vector with a same label.
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copy(LabelVec2)
    end
    return Z, nc, LabelVec1
end
"""
    HierarchyConstruct4s(
        Rpm::MatNum,
        Dpm::MatNum,
        Tc::VecNum,
        Mv::MatNum
    )

Constructs the intra- and inter-cluster hierarchy by utilizing the Bubble Hierarchy structure of a Maximal Planar Graph, specifically a Planar Maximally Filtered Graph (PMFG).

This function builds a hierarchical clustering (dendrogram) by first constructing intra-cluster linkages within each cluster (using the bubble structure), and then merging clusters to form the global hierarchy. It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline.

# Algorithm

 1. Give every vertex its own label in `LabelVec1`, and build `E`, the `N × maximum(Tc)` indicator of the discrete clustering `Tc`.
 2. For each cluster `k`, restrict `Mv` to the vertices of that cluster, giving `Mc`, and assign each of them to exactly one bubble with [`BubbleMember`](@ref), giving `Mvv`. Set the merge counter `nc` to the vertex count of the cluster less one.
 3. For each bubble of the cluster that holds more than one vertex, merge its vertices with [`build_link_and_dendro`](@ref) over `length(V) - 1` steps, on the distance submatrix of that bubble.
 4. Merge the bubbles of the cluster with [`build_link_and_dendro`](@ref) over `length(Bub) - 1` steps, on the distance submatrix of the whole cluster. Steps 3 and 4 share one `nc`, so the heights of a cluster rise across both.
 5. Repeat steps 2 to 4 for each cluster, which leaves one label per cluster.
 6. Merge the clusters over `length(kvec) - 1` steps: take the pair of smallest union diameter with [`LinkageFunction`](@ref) over the whole of `Dpm`, and give both sides a fresh label.
 7. Write the height of that merge from `dcl` and not from the score of step 6. `dcl` starts at `1` for every vertex and each merge sets both sides to the sum of the two, so the height counts the clusters the merge joins. This is what puts every inter-cluster merge above every intra-cluster one, whose heights never exceed `1`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG. It is read by [`BubbleMember`](@ref) alone.
  - `Dpm`: `N × N` shortest path lengths matrix of the PMFG. Every linkage score is read from it.
  - `Tc`: `N × 1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`-th discrete cluster.
  - `Mv`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.

# Returns

  - `Z::MatNum`: `(N-1)×3` linkage matrix in the same format as the output from Matlab. Each row holds the two merged labels and the height of the merge. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

# Related

  - [`BubbleMember`](@ref)
  - [`build_link_and_dendro`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
  - [`DBHT`](@ref)
"""
function HierarchyConstruct4s(Rpm::MatNum, Dpm::MatNum, Tc::VecNum, Mv::MatNum)
    N = size(Dpm, 1)
    kvec = sort!(unique(Tc))
    LabelVec1 = collect(1:N)
    E = SparseArrays.sparse(LabelVec1, Tc, ones(Int, N), N, maximum(Tc))
    Z = Matrix{Float64}(undef, 0, 3)

    # Intra-cluster hierarchy construction
    for n in eachindex(kvec)
        Mc = vec(E[:, kvec[n]]) ⊙ Mv   # Get the list of bubbles which coincide with nth cluster
        Mvv = BubbleMember(Rpm, Mv, Mc) # Assign each vertex in the nth cluster to a specific bubble
        Bub = findall(vec(sum(Mvv; dims = 1) .> 0)) # Get the list of bubbles which contain the vertices of nth cluster
        nc = sum(Tc .== kvec[n]) - 1

        # Apply the linkage within the bubbles.
        for m in eachindex(Bub)
            V = vec(findall(Mvv[:, Bub[m]] .!= 0)) # Retrieve the list of vertices assigned to mth bubble
            if length(V) > 1
                dpm = Dpm[V, V] # Retrieve the distance matrix for the vertices in V
                LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters
                LabelVec2 = copy(LabelVec1)
                Z, nc, LabelVec1 = build_link_and_dendro(1:(length(V) - 1), dpm, LabelVec,
                                                         LabelVec1, LabelVec2, V, nc, Z)
            end
        end

        V = findall(E[:, kvec[n]] .!= 0)
        dpm = Dpm[V, V]

        # Perform linkage merging between the bubbles
        LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters.
        LabelVec2 = copy(LabelVec1)
        Z, nc, LabelVec1 = build_link_and_dendro(1:(length(Bub) - 1), dpm, LabelVec,
                                                 LabelVec1, LabelVec2, V, nc, Z)
    end

    # Inter-cluster hierarchy construction
    LabelVec2 = copy(LabelVec1)
    dcl = ones(Int, length(LabelVec1))
    for _ in 1:(length(kvec) - 1)
        PairLink, dvu = LinkageFunction(Dpm, LabelVec1)
        LabelVec2[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= maximum(LabelVec1) +
                                                                            1
        dvu = unique(dcl[LabelVec1 .== PairLink[1]]) +
              unique(dcl[LabelVec1 .== PairLink[2]])
        dcl[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= dvu
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        LabelVec1 = copy(LabelVec2)
    end

    return Z
end
"""
    turn_into_Hclust_merges(Z::MatNum)

Convert a Matlab-style linkage matrix to a format compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

This function transforms a linkage matrix produced by DBHT or similar hierarchical clustering routines into the format required by [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), including proper indexing and cluster size tracking.

**This is the seam to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), so the convention below is the one every consumer downstream reads.** [`DBHTs`](@ref) loads the first two columns into `mleft` and `mright` and the third into `heights`, and [`Clusters`](@ref) and everything that cuts a dendrogram reads them back on that convention.

Both conventions number one merge per row, and they differ in how a row names its two sides.

| The side is                  | Matlab writes             | `Clustering.Hclust` writes |
|:---------------------------- |:------------------------- |:-------------------------- |
| a leaf, vertex `a`           | `a`, which is at most `N` | `-a`                       |
| the cluster built by row `j` | `j + N`                   | `j`                        |

A row therefore names only rows above it, and the size of the cluster it builds is the sum of the sizes of its two sides.

# Algorithm

 1. Set `N` to `size(Z, 1) + 1`, which is the leaf count, and append a fourth column of zeros to hold the cluster sizes.
 2. For each row `i` and for each of its first two entries `a`: when `a` is at most `N` it names a leaf, so write `-a` in its place and add `1` to the size of row `i`.
 3. Otherwise `a` names the cluster that row `j = a - N` built, so write `j` in its place and add the size of row `j` to the size of row `i`. Row `j` is above row `i`, so its size is already final.

# Arguments

  - `Z`: `(N-1)×3` Matlab-style linkage matrix, where each row represents a merge step with cluster indices and linkage heights.

# Returns

  - `Z::MatNum`: `(N-1)×4` linkage matrix in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format. Columns one and two hold the two sides on the convention of the table above, column three keeps the heights unchanged, and column four holds the leaf count of the cluster each row builds.

# Related

  - [`HierarchyConstruct4s`](@ref)
  - [`DendroConstruct`](@ref)
  - [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust)
"""
function turn_into_Hclust_merges(Z::MatNum)
    N = size(Z, 1) + 1
    Z = hcat(Z, zeros(eltype(Z), N - 1))

    for i in axes(Z, 1)

        # Cluster indices.
        a = Int(Z[i, 1])
        b = Int(Z[i, 2])

        # If the cluster index is less than N, it represents a leaf,
        # so only one add one to the count.
        if a <= N
            Z[i, 1] = -a
            Z[i, 4] += 1
        else
            # Clusters in index Z[i, 1:2] are combined to form cluster i + N.
            # If a cluster has index a > N, it's a combined cluster.
            # The index of the child is j = a - N, so we need to go to index j
            # which is being combined by cluster a, get the count at index j
            # and add it to the count at index i, which contains cluster a.
            j = a - N
            Z[i, 1] = j
            Z[i, 4] += Z[j, 4]
        end

        if b <= N
            Z[i, 2] = -b
            Z[i, 4] += 1
        else
            # Do the same with the other side of the cluster, to wherever that side leads.
            j = b - N
            Z[i, 2] = j
            Z[i, 4] += Z[j, 4]
        end
    end
    return Z
end
"""
    DBHTs(D::MatNum, S::MatNum; branchorder::Symbol = :optimal,
          root::DBHTRootMethod = UniqueRoot(),
          sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing)

Perform Direct Bubble Hierarchical Tree clustering, a deterministic clustering algorithm [DBHTs](@cite). This version uses a graph-theoretic filtering technique called Triangulated Maximally Filtered Graph (TMFG).

This function implements the full DBHT clustering pipeline: it constructs a Planar Maximally Filtered Graph (PMFG) from the similarity matrix, extracts the clique and bubble hierarchies, assigns clusters, and builds a hierarchical clustering (dendrogram) compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Algorithm

 1. Check that `D` and `S` are non-empty and of equal size.
 2. Build the PMFG from `S` with [`PMFG_T2s`](@ref), giving the weighted adjacency `Rpm`, and check its edge count with [`assert_pmfg_weights`](@ref).
 3. Copy the sparsity pattern of `Rpm` into `Apm` and fill it with the dissimilarities of `D`, so the structure comes from the similarities and the lengths from the distances.
 4. Take the shortest path lengths `Dpm` on `Apm` with [`distance_wei`](@ref).
 5. Build the clique and bubble hierarchies from `Rpm` with [`CliqHierarchyTree2s`](@ref), giving `Hb`, `Mb`, `CliqList` and `Sb`.
 6. Lift the clique membership `Mb` to the vertex membership `Mv`: column `n` marks every vertex of every 3-clique that bubble `n` holds.
 7. Assign the clusters with [`BubbleCluster8s`](@ref), giving `Adjv` and the discrete membership `T8`.
 8. Build the linkage matrix `Z` with [`HierarchyConstruct4s`](@ref), and convert it with [`turn_into_Hclust_merges`](@ref).
 9. Load the two merge columns and the heights into a `Clustering.HclustMerges`, and order its branches through the branch `branchorder` selects.
10. Wrap the merges in a `Clustering.Hclust` tagged `:DBHT`.

# Arguments

  - `D`: `N × N` dissimilarity matrix (e.g., a distance matrix). It must be symmetric, and the symmetry is a caller contract that this function does not check.
  - `S`: `N × N` non-negative similarity matrix. It must be symmetric, on the same unchecked contract.
  - `branchorder`: Ordering method for the dendrogram branches. `:optimal` and `:barjoseph` both call `Clustering.orderbranches_barjoseph!`, and `:r` calls `Clustering.orderbranches_r!`. Any other value is **not** refused: it leaves the branches in the order [`HierarchyConstruct4s`](@ref) built them.
  - `root`: Root selection method for the clique hierarchy.
  - `sim`: Similarity matrix algorithm that produced `S`. It is forwarded to [`assert_pmfg_weights`](@ref) and read for nothing else, so that a refusal names the configuration rather than the matrix. A caller that holds only the matrices leaves it `nothing`.

# Validation

  - `!isempty(S)`, raising `IsEmptyError`.
  - `!isempty(D)`, raising `IsEmptyError`.
  - `size(S) == size(D)`, raising `DimensionMismatch`.
  - The PMFG built from `S` keeps its `3N - 6` edges, by [`assert_pmfg_weights`](@ref). An exactly zero similarity is an absent edge.

Symmetry is **not** among them. A caller that derives both matrices from a correlation matrix gets
it by construction, and a caller that assembles either by hand carries the contract itself.

# Returns

  - `T8::Vector{Int}`: `N × 1` cluster membership vector. `T8[n] = k` puts vertex `n` in the `k`-th discrete cluster.
  - `Rpm::SparseMatrixCSC{<:Number, Int}`: `N × N` adjacency matrix of the Planar Maximally Filtered Graph (PMFG).
  - `Adjv::SparseMatrixCSC{Int, Int}`: Bubble cluster membership matrix from [`BubbleCluster8s`](@ref).
  - `Dpm::Matrix{<:Number}`: `N × N` shortest path length matrix of the PMFG.
  - `Mv::SparseMatrixCSC{Int, Int}`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `Z::Matrix{<:Number}`: `(N-1)×3` linkage matrix in Matlab format.
  - `Z_hclust::Clustering.Hclust`: Dendrogram in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format.

# Related

  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleCluster8s`](@ref)
  - [`HierarchyConstruct4s`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
  - [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust)
"""
function DBHTs(D::MatNum, S::MatNum; branchorder::Symbol = :optimal,
               root::DBHTRootMethod = UniqueRoot(),
               sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing)
    @argcheck(!isempty(S), IsEmptyError)
    @argcheck(!isempty(D), IsEmptyError)
    @argcheck(size(S) == size(D), DimensionMismatch)
    Rpm = PMFG_T2s(S)[1]
    assert_pmfg_weights(Rpm, sim)
    Apm = copy(Rpm)
    Apm[Apm .!= 0] = D[Apm .!= 0]
    Dpm = distance_wei(Apm)[1]

    H1, Hb, Mb, CliqList, Sb = CliqHierarchyTree2s(Rpm, root)

    Mb = Mb[1:size(CliqList, 1), :]

    sRpm = size(Rpm, 1)
    Mv = SparseArrays.spzeros(Int, sRpm, 0)

    nMb = size(Mb, 2)
    for n in axes(Mb, 2)
        vc = SparseArrays.spzeros(Int, sRpm)
        vc[sort!(unique(CliqList[Mb[:, n] .!= 0, :]))] .= 1
        Mv = hcat(Mv, vc)
    end

    Adjv, T8 = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)

    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Mv)
    Z = turn_into_Hclust_merges(Z)

    n = size(Z, 1)
    hmer = Clustering.HclustMerges{eltype(D)}(n + 1)
    resize!(hmer.mleft, n) .= Int.(Z[:, 1])
    resize!(hmer.mright, n) .= Int.(Z[:, 2])
    resize!(hmer.heights, n) .= Z[:, 3]

    if branchorder == :barjoseph || branchorder == :optimal
        Clustering.orderbranches_barjoseph!(hmer, D)
    elseif branchorder == :r
        Clustering.orderbranches_r!(hmer)
    end

    Z_hclust = Clustering.Hclust(hmer, :DBHT)

    return T8, Rpm, Adjv, Dpm, Mv, Z, Z_hclust
end
"""
    jlogo!(jlogo::MatNum, sigma::MatNum, source::MatNum, sign::Integer)

Efficiently accumulate contributions to the sparse inverse covariance matrix for LoGo/DBHT.

This internal function updates the `jlogo` matrix in-place by iterating over a list of cliques or separators (`source`), extracting the corresponding submatrix from the covariance matrix `sigma`, inverting it, and adding (or subtracting) the result to the appropriate block in `jlogo`, scaled by `sign`.

Every row of `source` names the same number of vertices, because `tmp` is allocated once at `size(source, 2)` and reused. [`PMFG_T2s`](@ref) satisfies that: its 3-cliques all carry three vertices and its 4-cliques all carry four.

# Algorithm

 1. Allocate `tmp`, one square block of the width of a row of `source`.
 2. For each row `i` of `source`, read the index set `v` and gather the submatrix `sigma[v, v]` into `tmp`.
 3. Invert `tmp`.
 4. Add `sign` times each entry of the inverse into `jlogo`, at the pair of `v` that entry belongs to.

# Arguments

  - `jlogo`: `N × N` matrix to be updated in-place. It is added to, never cleared, so the caller sets what it starts from.
  - `sigma`: `N × N` covariance matrix. Only the blocks the rows of `source` name are read.
  - `source`: `Ns×k` index matrix. Each row holds the `k` vertices of one clique or separator, and `k` is `4` for the cliques and `3` for the separators of a PMFG.
  - `sign`: `+1` for cliques, `-1` for separators.

# Returns

  - `nothing`. Updates `jlogo` in-place.

# Related

  - [`J_LoGo`](@ref)
  - [`LoGo`](@ref)
"""
function jlogo!(jlogo::MatNum, sigma::MatNum, source::MatNum, sign::Integer)
    tmp = Matrix{eltype(sigma)}(undef, size(source, 2), size(source, 2))

    # Pre-compute indices for better cache locality
    for i in axes(source, 1)
        v = view(source, i, :)

        # Fill temp matrix directly
        idx = 1
        for b in axes(source, 2)
            for a in axes(source, 2)
                tmp[idx] = sigma[v[a], v[b]]
                idx += 1
            end
        end

        # Compute inverse once
        tmp_inv = inv(tmp)

        # Update jlogo matrix directly
        idx = 1
        for b in axes(source, 2)
            for a in axes(source, 2)
                jlogo[v[a], v[b]] += sign * tmp_inv[idx]
                idx += 1
            end
        end
    end
    return nothing
end
"""
    J_LoGo(sigma::MatNum, separators::MatNum, cliques::MatNum)

Compute the sparse inverse covariance matrix using the LoGo (Local-Global) algorithm [J_LoGo](@cite).

This function implements the LoGo sparse inverse covariance estimation by combining clique and separator contributions from a Planar Maximally Filtered Graph (PMFG) or similar clique tree structure. It efficiently accumulates the inverses of covariance submatrices corresponding to cliques and separators, producing a sparse precision (inverse covariance) matrix suitable for robust portfolio optimization and risk management.

# Mathematical definition

```math
J_{i,\\,j} = \\sum_{c \\in \\mathcal{C}} \\mathbf{1}\\left[i \\in c \\land j \\in c\\right] \\left(\\mathbf{\\Sigma}_{c,\\,c}\\right)^{-1}_{i,\\,j}
           - \\sum_{s \\in \\mathcal{S}} \\mathbf{1}\\left[i \\in s \\land j \\in s\\right] \\left(\\mathbf{\\Sigma}_{s,\\,s}\\right)^{-1}_{i,\\,j}\\,.
```

Where:

  - ``\\mathbf{J}``: LoGo precision matrix, ``N \\times N``.
  - ``\\mathbf{\\Sigma}``: Covariance matrix, ``N \\times N``.
  - ``\\mathbf{\\Sigma}_{c,\\,c}``: Its submatrix on the index set ``c``.
  - ``\\mathcal{C}``: Set of the cliques of the network.
  - ``\\mathcal{S}``: Set of its separators.
  - $(math_dict[:N])

``J_{i,\\,j}`` is exactly zero for a pair that no clique holds together, so the sparsity pattern of ``\\mathbf{J}`` is the edge set of the network. That is the conditional independence the filtering states, and it survives in the precision alone.

# Algorithm

 1. Set `jlogo` to a zero matrix of the size of `sigma`.
 2. Add the inverse of every clique block with [`jlogo!`](@ref) at `sign = 1`.
 3. Subtract the inverse of every separator block with [`jlogo!`](@ref) at `sign = -1`.

# Arguments

  - `sigma`: `N × N` covariance matrix.
  - `separators`: `Ns×3` index matrix. Each row holds the vertices of one separator, which are the 3-cliques of a PMFG.
  - `cliques`: `Nq×4` index matrix. Each row holds the vertices of one clique, which are the 4-cliques of a PMFG.

# Returns

  - `jlogo::Matrix{<:Number}`: `N × N` LoGo sparse precision matrix. The covariance it stands for is its inverse, and that inverse is dense.

# Related

  - [`jlogo!`](@ref)
  - [`LoGo`](@ref)
"""
function J_LoGo(sigma::MatNum, separators::MatNum, cliques::MatNum)
    jlogo = zeros(eltype(sigma), size(sigma))
    jlogo!(jlogo, sigma, cliques, 1)
    jlogo!(jlogo, sigma, separators, -1)
    return jlogo
end
"""
    clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
               branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)

Perform Direct Bubble Hierarchical Tree (DBHT) clustering using a `ClustersEstimator` configured with a `DBHT` algorithm.

This method computes the similarity and distance matrices from the input data matrix `X` using the estimator's configured estimators and algorithms, applies the DBHT clustering pipeline, and returns a [`Clusters`](@ref) result containing the hierarchical clustering, similarity and distance matrices, and the optimal number of clusters.

# Algorithm

 1. Take the correlation matrix `S` and the distance matrix `D` from `X` with `cle.ce` and `cle.de`, through [`cor_and_dist`](@ref).
 2. Check that `D` lies in the domain `cle.alg.sim` needs, with [`assert_similarity_domain`](@ref).
 3. Map `D` to the non-negative similarity `S` with [`distance_to_similarity`](@ref), through the branch `cle.alg.sim` selects.
 4. Run the pipeline with [`DBHTs`](@ref), and keep its last output alone, the `Clustering.Hclust` dendrogram `res`.
 5. Take the number of clusters `k` from `res` and `D` with [`optimal_number_clusters`](@ref), through the branch `cle.onc` selects.
 6. Wrap `res`, `S`, `D` and `k` in a [`Clusters`](@ref).

# Arguments

  - `cle`: A `ClustersEstimator` whose algorithm is a [`DBHT`](@ref) instance.
  - `X`: Data matrix (`observations × assets` or `assets × observations` depending on `dims`).
  - `branchorder`: Symbol specifying the dendrogram branch ordering method. Accepts `:optimal` (default), `:barjoseph`, or `:r`.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `clr::Clusters`: DBHT clustering result.

# Related

  - [`DBHT`](@ref)
  - [`Clusters`](@ref)
  - [`DBHTs`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
                    branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(cle.alg.sim, cle.de, D)
    S = distance_to_similarity(cle.alg.sim; S = S, D = D)
    res = DBHTs(D, S; branchorder = branchorder, root = cle.alg.root, sim = cle.alg.sim)[end]
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op fallback: return `nothing` when no LoGo algorithm is configured.

This is the branch a matrix processing pipeline takes when its sparsification field is `nothing`, so a caller composes the step in and out without a branch of its own.

# Algorithm

 1. Return `nothing`. No matrix is read and no matrix is written, and `args` and `kwargs` are discarded.

# Arguments

  - `::Nothing`: No LoGo algorithm configured.
  - `args...`: Optional arguments (ignored).
  - `kwargs...`: Optional keyword arguments (ignored).

# Returns

  - `nothing`. The caller's `sigma` is left as it stands.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function logo!(::Nothing, args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all inverse matrix sparsification algorithms.

A member of this family imposes a sparsity pattern on the **inverse** of a covariance matrix rather than on the matrix itself. The covariance that comes back is dense; what is sparse is its precision, and the zeros there are the conditional independences the information filtering network selected.

The family declares no seam of its own, and no method dispatches on this supertype. A concrete subtype is reached through the [`matrix_processing_algorithm!`](@ref) of [`AbstractMatrixProcessingAlgorithm`](@ref), which is the interface it inherits and which `src/07_MatrixProcessing.jl` owns. [`LoGo`](@ref) is the shipped member, and [`matrix_processing_algorithm!`](@ref) states the contract that method satisfies.

# Related

  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`LoGo`](@ref)

# References

  - $(ref_dict[:J_LoGo])
"""
abstract type InverseMatrixSparsificationAlgorithm <: AbstractMatrixProcessingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sparsifies the inverse covariance matrix on the cliques of an information filtering network.

`LoGo` is a composable algorithm type for estimating sparse inverse covariance matrices using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition, as described in [J_LoGo](@cite). It combines a distance estimator and a similarity matrix algorithm, both validated and extensible, to produce a robust, interpretable sparse precision matrix for use in portfolio optimization and risk management.

# What is sparse is the precision, not the covariance

[`J_LoGo`](@ref) sums the inverse of each clique block and subtracts the inverse of each separator block, and the matrix that comes out is **exactly zero** wherever the network carries no edge. Measured over a 20-asset sample, the triangulated maximally filtered graph holds `54` edges — the `3n - 6` of a maximal planar graph — and the largest absolute entry of the precision matrix away from those edges is `0.0`.

`sigma` is then replaced by the inverse of that precision matrix, so what the caller receives is dense. The filtering is a statement about which pairs are conditionally independent given the rest, and it survives only in the precision.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LoGo(;
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef()
    ) -> LoGo

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LoGo()
LoGo
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  sim ┼ MaximumDistanceSimilarity()
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`InverseMatrixSparsificationAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)

# References

  - $(ref_dict[:J_LoGo])
"""
@concrete struct LoGo <: InverseMatrixSparsificationAlgorithm
    """
    $(field_dict[:de])
    """
    de
    """
    $(field_dict[:sim])
    """
    sim <: AbstractNonNegativeSimilarityMatrixAlgorithm
    """
    $(field_dict[:pdm])
    """
    pdm
    function LoGo(de::AbstractDistanceEstimator,
                  sim::AbstractNonNegativeSimilarityMatrixAlgorithm,
                  pdm::Option{<:AbstractPosdefEstimator} = Posdef())
        return new{typeof(de), typeof(sim), typeof(pdm)}(de, sim, pdm)
    end
end
function LoGo(; de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
              sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              pdm::Option{<:AbstractPosdefEstimator} = Posdef())
    return LoGo(de, sim, pdm)
end
"""
    const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                     <:DistanceDistance{<:Any, <:VariationInfoDistance, <:Any,
                                                        <:Any, <:Any}}

Alias for distance types using variation of information metrics.

Matches either a [`VariationInfoDistance`](@ref)-based [`Distance`](@ref) or a [`VariationInfoDistance`](@ref)-based [`DistanceDistance`](@ref). Used for dispatch in DBHT-based phylogeny computation.

# Related

  - [`VariationInfoDistance`](@ref)
  - [`Distance`](@ref)
  - [`DistanceDistance`](@ref)
"""
const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                 <:DistanceDistance{<:Any, <:VariationInfoDistance, <:Any,
                                                    <:Any, <:Any}}
"""
    LoGo_dist_assert(de::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum)

Validate compatibility of the distance estimator and covariance matrix for LoGo sparse inverse covariance estimation by checking `size(sigma, 1) == size(X, 2)`.

The check runs for a [`VariationInfoDistance`](@ref) estimator alone, which is the only family that reads `X` rather than the correlation matrix. Every other estimator takes the no-op fallback, so a mismatched `X` passes. The narrow signature is what makes that so: it is bounded by [`DVarInfo_DDVarInfo`](@ref), and the configurations that reach it are a [`Distance`](@ref) or a [`DistanceDistance`](@ref) whose algorithm is a [`VariationInfoDistance`](@ref).

# Arguments

  - `de`: Distance estimator whose algorithm is a [`VariationInfoDistance`](@ref).
  - `sigma`: `N × N` covariance matrix.
  - `X`: `T × N` data matrix. `size(X, 2)` is the asset axis, which is the axis the check reads.

# Validation

  - `size(sigma, 1) == size(X, 2)`.

# Returns

  - `nothing`.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum)
    @argcheck(size(sigma, 1) == size(X, 2), DimensionMismatch)
    return nothing
end
"""
    LoGo_dist_assert(args...)

No-op fallback for other distance estimators.

Every distance estimator outside [`DVarInfo_DDVarInfo`](@ref) derives its distance from the correlation matrix and never reads `X`, so there is no shape of `X` for it to disagree with. This method makes that the default and leaves the check to the one family that owns it.

# Algorithm

 1. Return `nothing`. No shape is read, and `args` is discarded.

# Arguments

  - `args...`: The distance estimator, the covariance matrix and the data matrix (all ignored).

# Returns

  - `nothing`.

# Related

  - [`DVarInfo_DDVarInfo`](@ref)
  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(args...)
    return nothing
end
"""
    logo!(je::LoGo, sigma::MatNum, X::MatNum;
          dims::Int = 1, kwargs...)

Compute the LoGo (Local-Global) covariance matrix and update `sigma` in-place.

This method implements the LoGo algorithm for sparse inverse covariance estimation using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition. It validates inputs, computes the similarity and distance matrices, constructs the PMFG, identifies cliques and separators, and updates the input covariance matrix `sigma` in-place by inverting the LoGo sparse inverse covariance estimate. The result is projected to the nearest positive definite matrix if a `Posdef` estimator is not `nothing`.

# Algorithm

 1. Check that `sigma` is square, and check its asset axis against `X` through [`LoGo_dist_assert`](@ref).
 2. Read the diagonal of `sigma` into `s`. When any entry of `s` is not one, `sigma` is a covariance matrix: replace `s` with its square roots and derive the correlation matrix `S` with `StatsBase.cov2cor`. `sigma` itself stays a covariance, and it is what step 6 decomposes.
 3. Take the distance matrix `D` from `S` and `X` with `je.de`, and check that `D` lies in the domain `je.sim` needs.
 4. Map `D` to the non-negative similarity `S` with `je.sim`, through [`distance_to_similarity`](@ref).
 5. Build the TMFG on `S` with [`PMFG_T2s`](@ref) at `nargout = 4`, and take its 3-cliques as the separators and its 4-cliques as the cliques.
 6. Build the LoGo precision matrix from `sigma` with [`J_LoGo`](@ref), invert it, and write the result into `sigma`.
 7. Repair `sigma` with [`posdef!`](@ref) through `je.pdm`, which does nothing when `je.pdm` is `nothing`.

# Arguments

  - `je`: LoGo algorithm instance.
  - `sigma`: Covariance matrix (`N × N`), updated in-place with the LoGo sparse inverse covariance.
  - `X`: Data matrix (`T × N`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - `size(sigma, 1) == size(sigma, 2)`.
  - `size(sigma, 1) == size(X, 2)`, **only when `je.de` reads `X`**. [`LoGo_dist_assert`](@ref) carries the check, and it has a method for the variation-of-information estimators alone; every other estimator takes the no-op fallback. A default `LoGo()` therefore accepts a `20 × 20` `sigma` beside a `400 × 10` `X` and returns without raising, because [`CanonicalDistance`](@ref) derives the distance from the correlation matrix and never touches `X`.

# Returns

  - `nothing`. The input `sigma` is updated in-place.

# Related

  - [`LoGo`](@ref)
  - [`J_LoGo`](@ref)
  - [`LoGo_dist_assert`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`Posdef`](@ref)
"""
function logo!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    assert_matrix_issquare(sigma, :sigma)
    LoGo_dist_assert(je.de, sigma, X)
    s = LinearAlgebra.diag(sigma)
    iscov = any(!isone, s)
    S = if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor(sigma, s)
    else
        sigma
    end
    D = distance(je.de, S, X; dims = dims, kwargs...)
    assert_similarity_domain(je.sim, je.de, D)
    S = distance_to_similarity(je.sim; S = S, D = D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    sigma .= J_LoGo(sigma, separators, cliques) \ LinearAlgebra.I
    posdef!(je.pdm, sigma)
    return nothing
end
"""
    logo(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...) -> MatNum

Apply the LoGo (Local-Global) transformation to the covariance matrix and return the result as a new matrix.

This is the non-mutating variant of [`logo!`](@ref). It copies `sigma` before applying the transformation.

# Algorithm

 1. Copy `sigma`.
 2. Run [`logo!`](@ref) on the copy, which carries every step and every check of this transformation.
 3. Return the copy.

# Arguments

  - `je::LoGo`: LoGo algorithm configuration.
  - `sigma::MatNum`: `N × N` covariance matrix to transform (not mutated).
  - `X::MatNum`: `T × N` returns data matrix.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - Every check of [`logo!`](@ref) applies, and it raises from step 2.

# Returns

  - `sigma::MatNum`: `N × N` copy of the input with the LoGo transformation applied.

# Related

  - [`logo!`](@ref)
  - [`LoGo`](@ref)
  - [`J_LoGo`](@ref)
"""
function logo(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    sigma = copy(sigma)
    logo!(je, sigma, X; dims = dims, kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm!(je::LoGo, sigma::MatNum,
                                 X::MatNum; dims::Int = 1, kwargs...)

Apply the LoGo (Local-Global) transformation in-place to the covariance matrix, as a step of the matrix processing pipeline.

This method provides a standard interface for applying the LoGo algorithm to a covariance matrix within the matrix processing pipeline of `PortfolioOptimisers.jl`. It validates inputs, computes the LoGo sparse inverse covariance matrix, and updates `sigma` in-place. If a positive definite matrix estimator (`pdm`) is not `nothing`, the result is projected to the nearest positive definite matrix.

This is the contract [`LoGo`](@ref) satisfies as a member of [`AbstractMatrixProcessingAlgorithm`](@ref): the pipeline calls `matrix_processing_algorithm!(alg, sigma, X; dims, kwargs...)` on each of its algorithms in turn, each one writes into the same `sigma`, and each returns `nothing`. The family lives in `src/07_MatrixProcessing.jl`, and this method is the only one of it that this file declares.

# Algorithm

 1. Forward every argument to [`logo!`](@ref), which carries the steps and the checks of the transformation.

# Arguments

  - `je`: LoGo algorithm instance (`LoGo`). Its own `pdm` field carries the positive definite repair, so there is no `pdm` argument here.
  - `sigma`: Covariance matrix (`N × N`), updated in-place.
  - `X`: Data matrix (`T × N` or `N × T`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - Every check of [`logo!`](@ref) applies, and it raises from step 1.

# Returns

  - `nothing`. The input `sigma` is updated in-place.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
  - [`Posdef`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
"""
function matrix_processing_algorithm!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1,
                                      kwargs...)
    return logo!(je, sigma, X; dims = dims, kwargs...)
end

export UniqueRoot, EqualRoot, DBHT, LoGo, Clusters
