"""
    PMFG_T2s(W::MatNum, nargout::Integer = 3)

Build a Triangulated Maximally Filtered Graph (TMFG), an approximation of the maximal planar graph of the largest total weight.

The function starts from a tetrahedron. At each step it inserts one vertex into a triangular face, the T2 move. The Planar Maximally Filtered Graph (PMFG) is an earlier approximation of the same graph. All weights must be non-negative.

[`DBHTs`](@ref), [`logo!`](@ref) and the network estimators call this function. It gives them the planar graph and its cliques. DBHT clusters the assets on them, and LoGo computes a sparse precision matrix on them.

`nargout` is a positional argument, and every caller passes it as one.

# The TMFG is not the PMFG

The problem of the weighted maximal planar graph asks for the maximal planar subgraph of the largest total weight, and it is NP-complete. Neither graph solves it exactly. The planar maximally filtered graph of [tumminello2005](@cite) is a greedy heuristic. It takes the edges in order of weight, and it keeps an edge while the graph stays planar.

The TMFG that this function builds is a different and cheaper heuristic, from [PMFG](@cite). It inserts vertices, not edges, so the two graphs are different in general. The name of the function names the PMFG, but the algorithm builds the TMFG. Both are maximal planar graphs, so each has exactly ``3N - 6`` edges. A minimum spanning tree has ``N - 1``.

Steps 1 and 2 below are a rule of the MATLAB code that accompanies [DBHTs](@cite). Algorithm 1 of [PMFG](@cite) starts from the tetrahedron of the largest total weight. Steps 9 to 11 and the bound `N >= 9` come from the same code, not from [PMFG](@cite).

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
  - ``g(v,\\, f)``: Gain of the insertion of vertex ``v`` into face ``f``.
  - $(math_dict[:N])

# Algorithm

 1. Score every vertex by `s`, the row sum of `W` over the entries above the mean of `W`.
 2. Take the four vertices of largest `s` as `in_v[1:4]`, and the rest as `ou_v`.
 3. Build the tetrahedron on those four vertices. Write its four faces into `tri[1:4, :]`, and its six edges into `A`.
 4. Build the gain table `gain[v, f]`, one entry for each vertex of `ou_v` and each face of `tri`.
 5. Take the pair of largest gain, which gives the vertex `ve` and the face `agm`. Remove `ve` from `ou_v` and record it in `in_v`.
 6. Join `ve` to the three vertices of face `agm` in `A`, and record that face in `clique3`. The face becomes a 3-clique that is not a face.
 7. Replace face `agm` and append two more, so each of the three faces of the split has `ve`.
 8. Compute the three changed columns of `gain` again, and set the row of `ve` to zero. Repeat from step 5 until `ou_v` is empty.
 9. Weight the structure with `A = W ⊙ ((A + A') .== 1)`, so a stored entry is an edge and its value is its weight.
10. When `nargout > 3`, build `cliques`. Its first row is the first tetrahedron. Each other row is the 4-clique of one inserted vertex, which holds the face that the vertex entered and the vertex.
11. When `nargout > 4`, build `cliqueTree`. For each 4-clique, count in `ss` how many of its first three vertices each 4-clique holds, and mark the 4-cliques whose count is `2`.

# Arguments

  - `W`: `N × N` matrix of non-negative, finite weights, such as a similarity matrix of an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) or an absolute correlation matrix.
  - `nargout`: Number of outputs to build. The function builds `cliques` when `nargout > 3`, and `cliqueTree` when `nargout > 4`. Each is `nothing` otherwise. The function always builds the first three outputs.

# Validation

  - `size(W, 1) >= 9`, raising `DimensionMismatch`.
  - No entry of `W` is `NaN`, raising `DomainError`.
  - Every entry of `W` is non-negative, raising `DomainError`.

An entry that is exactly zero passes all three checks, but the graph loses an edge for it. `A` holds the structure in its sparsity pattern, and this function refuses no edge. [`assert_pmfg_weights`](@ref) holds that check. The callers that read the weighted structure run it. This function does not run it, because [`logo!`](@ref) reads only the cliques, and a zero does not change its result.

# The estimators refuse a negative weight first

[`NetworkEstimator`](@ref), [`DBHT`](@ref) and [`LoGo`](@ref) reach this function. Each bounds its similarity field by [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), and each calls [`assert_similarity_domain`](@ref) before it maps the distances. Thus a configuration of the library that fails here fails earlier, in the constructor or in `assert_similarity_domain`. The message of that error names the configuration and not `W`.

The two checks of this function cover the case that the estimators cannot cover. Any package can subtype that family, so an extension can return a negative weight. Without the checks, the error downstream is silent. [`DirectHb`](@ref) sums signed weights, so a negative weight can change the set of converging bubbles, and the result is a wrong clustering with no error.

# Returns

  - `A::SparseMatrixCSC{<:Number, Int}`: Weighted adjacency matrix of the TMFG.
  - `tri::Matrix{Int}`: List of the triangular faces of the TMFG, one face for each row.
  - `clique3::Matrix{Int}`: List of the 3-cliques that are not triangular faces, one 3-clique for each row. `[tri; clique3]` holds every 3-clique.
  - `cliques::Option{Matrix{Int}}`: List of every 4-clique, a tetrahedron, one for each row, or `nothing` if `nargout <= 3`.
  - `cliqueTree::Option{SparseMatrixCSC{Int, Int}}`: Adjacency matrix of the tree of the 4-cliques, or `nothing` if `nargout <= 4`.

# Related

  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
  - [`LoGo`](@ref)

# References

  - $(ref_dict[:PMFG]) Section III.A, Algorithm 1, and Section I for the PMFG as a greedy heuristic.
  - $(ref_dict[:tumminello2005]) For the PMFG and its construction by edge insertion.
"""
function PMFG_T2s(W::MatNum, nargout::Integer = 3)
    N = size(W, 1)
    @argcheck(9 <= N, DimensionMismatch("9 <= size(W, 1) must hold. Got\nsize(W, 1) => $N"))
    # Split in two, because `0 <= NaN` is `false`: one check would report a `NaN` as a
    # negative weight and send the caller looking for the wrong thing.
    @argcheck(!any(isnan, W),
              DomainError(W,
                          "!any(isnan, W) must hold. Got\ncount(isnan, W) => $(count(isnan, W))."))
    @argcheck(all(x -> zero(x) <= x, W),
              DomainError(W,
                          "all(x -> x >= 0, W) must hold. Got\nminimum(W) => $(minimum(W))."))
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
        deleteat!(ou_v, v)
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
        cliques = @views vcat(transpose(in_v[1:4]), hcat(clique3, in_v[5:end]))
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

Check that no zero weight removed an edge from the graph that [`PMFG_T2s`](@ref) built.

A maximal planar graph on `N >= 3` vertices has exactly `3N - 6` edges. [`PMFG_T2s`](@ref) returns the structure and the weights in one matrix, `A = W ⊙ ((A + A') .== 1)`. Thus a weight that is exactly zero is an absent edge, not a weak edge, and the consumer does not get a PMFG. This function counts the stored edges, and it refuses a count that is not `3N - 6`.

# A zero weight is a valid input, but it removes an edge

The check of [`PMFG_T2s`](@ref) is `>= 0`, because zero is a correct similarity. [`LogDistance`](@ref) returns an infinite distance at a correlation of exactly zero, and [`ExponentialSimilarity`](@ref) maps it to `exp(-Inf)`, which is exactly `0`. [`ComplementSimilarity`](@ref) maps `D = 1` to `0`. The value is correct, but a zero weight cannot hold an edge.

Without this check, the error is a `BoundsError` about a matrix index. [`turn_into_Hclust_merges`](@ref) raises it much later, because [`HierarchyConstruct4s`](@ref) then builds fewer merges than the dendrogram needs.

# The callers that run the check

The three callers that read the weighted structure run it: [`DBHTs`](@ref), [`calc_weighted_adjacency_graph`](@ref) and [`calc_distance_weighted_graph`](@ref).

[`logo!`](@ref) is the fourth caller of [`PMFG_T2s`](@ref), and it does not run the check. It reads the separators and the cliques. [`PMFG_T2s`](@ref) derives them from the order of insertion and not from `A`, so a zero weight does not change the result of `logo!`. A refusal there would refuse a configuration that works.

# Algorithm

 1. Count the non-zero entries of `A`, and divide the count by two, which gives `edges`. `A` is symmetric, so it stores each edge two times.
 2. Build `source`, the part of the message that names the configuration, from the parts of `sim` and `de` that the caller gives.
 3. Raise a `DomainError` when `edges` is not `expected`, which is `3N - 6`.

# Arguments

  - `A`: `N × N` weighted adjacency matrix, the first output of [`PMFG_T2s`](@ref).
  - `sim`: Similarity matrix algorithm that made the weights. The message names it, and the function reads it for nothing else, as [`assert_similarity_domain`](@ref) reads its `de`.
  - `de`: Distance estimator from which the similarity comes. The message names it next to `sim`.

Each caller gives the parts that it holds, so the message names all of the configuration that the caller knows. [`calc_distance_weighted_graph`](@ref) holds both parts, [`calc_weighted_adjacency_graph`](@ref) and [`DBHTs`](@ref) hold the similarity, and a caller that holds only the matrices names neither.

# Validation

  - The number of stored edges is `3N - 6`, raising `DomainError`.

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
