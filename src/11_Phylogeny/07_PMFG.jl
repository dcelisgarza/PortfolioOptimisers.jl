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
