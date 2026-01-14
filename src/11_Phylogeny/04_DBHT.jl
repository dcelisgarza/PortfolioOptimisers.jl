"""
    abstract type DBHTRootMethod <: AbstractAlgorithm end

Abstract supertype for all Direct Bubble Hierarchy Tree (DBHT) root selection methods in PortfolioOptimisers.jl.

# Related

  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
"""
abstract type DBHTRootMethod <: AbstractAlgorithm end
"""
    struct UniqueRoot <: DBHTRootMethod end

A DBHT root selection method that enforces a unique root in the hierarchy.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
"""
struct UniqueRoot <: DBHTRootMethod end
"""
    struct EqualRoot <: DBHTRootMethod end

A DBHT root selection method that creates a root from the adjacency tree of all root candidates. This can be used to represent multiple equally plausible roots in the DBHT hierarchy.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`DBHT`](@ref)
"""
struct EqualRoot <: DBHTRootMethod end
"""
    abstract type AbstractSimilarityMatrixAlgorithm <: AbstractAlgorithm end

Abstract supertype for all similarity matrix algorithms used in the creation of Planar Maximally Filtered Graph (PMFG) used in [`DBHT`](@ref) and [`LoGo`](@ref) methods.

# Related

  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`DBHT`](@ref)
  - [`LoGo`](@ref)
"""
abstract type AbstractSimilarityMatrixAlgorithm <: AbstractAlgorithm end
"""
    struct MaximumDistanceSimilarity <: AbstractSimilarityMatrixAlgorithm end

Similarity matrix algorithm using the maximum distance transformation.

```math
\\begin{align}
S_{i,\\,j} &= \\left\\lceil\\max(\\mathbf{D})^2\\right\\rceil - D_{i,\\,j}^2\\,,
\\end{align}
```

where `S` is the similarity, `\\mathbf{D}` the distance matrix, and each subscript denotes an asset.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`dbht_similarity`](@ref)
"""
struct MaximumDistanceSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
    struct ExponentialSimilarity <: AbstractSimilarityMatrixAlgorithm end

Similarity matrix algorithm using the exponential transformation.

```math
\\begin{align}
S_{i,\\,j} &= e^{-D_{i,\\,j}}\\,,
\\end{align}
```

where `S` is the similarity, `\\mathbf{D}` the distance matrix, and each subscript denotes an asset.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`dbht_similarity`](@ref)
"""
struct ExponentialSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
    struct GeneralExponentialSimilarity{T1, T2} <: AbstractSimilarityMatrixAlgorithm
        coef::T1
        power::T2
    end

Similarity matrix algorithm using a generalised exponential transformation.

```math
\\begin{align}
S_{i,\\,j} &= e^{-c \\cdot D_{i,\\,j}^p}\\,,
\\end{align}
```

where `S` is the similarity, `\\mathbf{D}` the distance matrix, ``c`` a scale factor, ``p`` an exponent, and each subscript denotes an asset.

# Fields

  - `coef`: Non-negative scaling coefficient.
  - `power`: Non-negative exponent applied to the distance matrix.

# Constructor

    GeneralExponentialSimilarity(; coef::Number = 1.0, power::Number = 1.0)

Keyword arguments correspond to the fields above.

## Validation

  - `coef > 0`.
  - `power > 0`.

# Examples

```jldoctest
julia> GeneralExponentialSimilarity()
GeneralExponentialSimilarity
   coef ┼ Float64: 1.0
  power ┴ Float64: 1.0
```

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`dbht_similarity`](@ref)
"""
struct GeneralExponentialSimilarity{T1, T2} <: AbstractSimilarityMatrixAlgorithm
    coef::T1
    power::T2
    function GeneralExponentialSimilarity(coef::Number, power::Number)
        @argcheck(zero(coef) < coef, DomainError)
        @argcheck(zero(power) < power, DomainError)
        return new{typeof(coef), typeof(power)}(coef, power)
    end
end
function GeneralExponentialSimilarity(; coef::Number = 1.0, power::Number = 1.0)
    return GeneralExponentialSimilarity(coef, power)
end
"""
    dbht_similarity(se::AbstractSimilarityMatrixAlgorithm; D::MatNum, kwargs...)

Compute a similarity matrix from a distance matrix using the specified similarity algorithm.

This function dispatches on the type of `se` to apply the appropriate similarity transformation to the distance matrix `D`. Used internally by DBHT and related clustering algorithms.

# Arguments

  - `se`: Similarity matrix algorithm.

      + `se::MaximumDistanceSimilarity`: Uses the maximum distance transformation.
      + `se::ExponentialSimilarity`: Uses the exponential transformation.
      + `se::GeneralExponentialSimilarity`: Uses a generalised exponential transformation.

  - `D`: Distance matrix.
  - `kwargs...`: Additional keyword arguments (not used).

# Returns

  - `S::Matrix{<:Number}`: Similarity matrix of the same size as `D`.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
"""
function dbht_similarity(::MaximumDistanceSimilarity; D::MatNum, kwargs...)
    return ceil(maximum(D)^2) .- D .^ 2
end
function dbht_similarity(::ExponentialSimilarity; D::MatNum, kwargs...)
    return exp.(-D)
end
function dbht_similarity(se::GeneralExponentialSimilarity; D::MatNum, kwargs...)
    power = se.power
    coef = se.coef
    return exp.(-coef * D .^ power)
end
"""
    struct DBHT{T1, T2} <: AbstractHierarchicalClusteringAlgorithm
        sim::T1
        root::T2
    end

Direct Bubble Hierarchical Tree (DBHT) clustering algorithm configuration.

`DBHT` is a composable clustering algorithm type for constructing hierarchical clusterings using the Direct Bubble Hierarchical Tree (DBHT) method, as described in [DBHTs](@cite).

# Fields

  - `sim`: Similarity matrix algorithm.
  - `root`: Root selection method.

# Constructor

    DBHT(; sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
         root::DBHTRootMethod = UniqueRoot())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DBHT()
DBHT
   sim ┼ MaximumDistanceSimilarity()
  root ┴ UniqueRoot()
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`]-(@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`DBHTRootMethod`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
"""
struct DBHT{T1, T2} <: AbstractHierarchicalClusteringAlgorithm
    sim::T1
    root::T2
    function DBHT(sim::AbstractSimilarityMatrixAlgorithm, root::DBHTRootMethod)
        return new{typeof(sim), typeof(root)}(sim, root)
    end
end
function DBHT(; sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              root::DBHTRootMethod = UniqueRoot())
    return DBHT(sim, root)
end
"""
    PMFG_T2s(W::MatNum; nargout::Integer = 3)

Constructs a Triangulated Maximally Filtered Graph (TMFG) starting from a tetrahedron and recursively inserting vertices inside existing triangles (T2 move) in order to approximate a Maximal Planar Graph with the largest total weight, also known as the Planar Maximally Filtered Graph (PMFG). All weights must be non-negative.

This function is a core step in the DBHT (Direct Bubble Hierarchical Tree) and LoGo algorithms, providing the planar graph structure and clique information required for hierarchical clustering and sparse inverse covariance estimation.

# Arguments

  - `W`: `N × N` matrix of non-negative weights (e.g., similarity or correlation matrix).
  - `nargout`: Number of output arguments. All outputs are always computed, but if `nargout <= 3`, `cliques` and `cliqueTree` are returned as `nothing`.

# Validation

  - `N >= 9` is required for a meaningful PMFG.
  - All entries in `W` must be non-negative.

# Details

  - The algorithm starts by selecting the four vertices with the largest strength to form an initial tetrahedron.
  - Vertices are recursively inserted into existing triangles to maximize the total weight, following the T2 move.
  - The resulting graph is planar and maximally filtered, preserving the most relevant connections for hierarchical clustering.
  - The function also identifies all 3-cliques and, optionally, all 4-cliques and their adjacency structure.

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
"""
function PMFG_T2s(W::MatNum, nargout::Integer = 3)
    N = size(W, 1)
    @argcheck(9 <= N, DimensionMismatch("9 <= size(W, 1) must hold. Got\nsize(W, 1) => $N"))
    @argcheck(all(x -> zero(x) <= x, W),
              DomainError("all(x -> x >= 0, W) must hold. Got\nall(x -> x >= 0, W) => $(all(x -> zero(x) <= x, W))."))
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
        clique3[k - 4, :] = tri[agm, :]

        # Update triangle list replacing 1 and adding 2 triangles
        tri[kk + 1, :] = vcat(tri[agm, [1, 3]], ve) # add
        tri[kk + 2, :] = vcat(tri[agm, [2, 3]], ve) # add
        tri[agm, :] = vcat(tri[agm, [1, 2]], ve)     # replace

        # # Update gain table
        gain[ve, :] .= 0
        gain[ou_v, agm] = sum(W[ou_v, tri[agm, :]]; dims = 2)
        gain[ou_v, kk + 1] = sum(W[ou_v, tri[kk + 1, :]]; dims = 2)
        gain[ou_v, kk + 2] = sum(W[ou_v, tri[kk + 2, :]]; dims = 2)

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
    distance_wei(L::MatNum)

Compute the shortest weighted path lengths between all node pairs in a network.

This function computes the distance matrix containing the lengths of the shortest paths between all node pairs in a (possibly weighted) network, using Dijkstra's algorithm. An entry `[u, v]` represents the length of the shortest path from node `u` to node `v`. The average shortest path length is the characteristic path length of the network.

# Inputs

  - `L`: Directed or undirected connection-length matrix.

      + Lengths between disconnected nodes should be set to `Inf`.
      + Lengths on the main diagonal should be set to `0`.

!!! note

    The input matrix must be a connection-length matrix, typically obtained by mapping weights to lengths (e.g., inverse of a similarity or correlation matrix). In weighted networks, shortest weighted paths may traverse more edges than shortest binary paths.

# Details

  - For each node, the function computes the shortest path to all other nodes using Dijkstra's algorithm.
  - The output `D` contains the minimal total length for each node pair, and `B` contains the number of edges in the corresponding shortest path.
  - Used internally for PMFG and DBHT clustering to compute geodesic distances on the graph.

# Returns

  - `D::Matrix{<:Number}`: Distance (shortest weighted path) matrix.
  - `B::Matrix{Int}`: Number of edges in the shortest weighted path matrix.

!!! note

    Based on a Matlab implementation by Mika Rubinov, Rick Betzel, and Andrea Avena.

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
            minD = !isempty(dus) ? minimum(dus) : Float64[]

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

# Inputs

  - `A`: `N × N` adjacency matrix of a Maximal Planar Graph (MPG).

# Details

  - The function searches for all triangles (3-cliques) by examining pairs of connected nodes and their shared neighbors.
  - Duplicates are removed and the resulting list is sorted for consistency.
  - The output `clique` matrix is used as the basis for further hierarchical and bubble structure construction in DBHT.

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

This function performs a breadth-first search (BFS) on a binary (directed or undirected) connection matrix, starting from a specified source vertex. It computes the shortest path distances from the source to all other vertices and records the predecessor (branch) for each node in the BFS tree.

# Inputs

  - `CIJ`: Binary (0/1) connection matrix representing the graph.
  - `source`: Index of the source vertex from which to start the search.

# Returns

  - `distance::VecNum`: Vector of shortest path distances from the source to each vertex (`0` for the source itself, `Inf` for unreachable nodes).
  - `branch::Vector{Int}`: Vector of predecessor indices for each vertex in the BFS tree (`-1` for the source).

# Details

  - The function explores the entire graph, layer by layer, starting from the source vertex.
  - For each node, it records the minimum number of steps required to reach it from the source.
  - The `branch` vector allows reconstruction of the BFS tree.
  - Used internally for component analysis and separating set identification in DBHT and related algorithms.

# Notes

  - The BFS tree does not contain all paths (or all shortest paths), but allows the determination of at least one path with minimum distance.
  - Original implementation by Olaf Sporns, Indiana University, 2002/2007/2008.

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

# Arguments

  - `Adj`: `N × N` adjacency matrix.
  - `Cliq`: `3×1` vector of node indices forming a 3-clique.

# Details

  - The function removes the clique nodes from the adjacency matrix and performs a breadth-first search to classify the remaining nodes.
  - Nodes unreachable from the first non-clique node are marked as disjoint.
  - Used internally by DBHT routines to determine separating sets and clique membership.

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

# Arguments

  - `M`: `N×Nc` binary matrix of node-to-3-clique memberships, where `M[i, n] = 1` if node `i` belongs to 3-clique `n`.

# Details

  - For each 3-clique, the function identifies its parent clique as the smallest superset among all cliques containing its nodes.
  - If multiple parent candidates exist, the one with the smallest overlap is chosen.
  - Root cliques (with no parent) are assigned a parent index of `0`.
  - Used internally by [`CliqHierarchyTree2s`](@ref) and DBHT clustering routines.

# Returns

  - `Pred::Vector{Int}`: `Nc×1` vector of predicted parent indices for each 3-clique. `Pred[n] = 0` indicates a root clique.

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
            length(a) == 1 ? Pred[n] = Parents[a[1]] : Pred = Int[]
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

This function computes the adjacency matrix among root candidate 3-cliques, identifying which root cliques are adjacent (i.e., share two vertices) in the graph. Used internally by [`CliqueRoot`](@ref) with [`EqualRoot`](@ref) to construct a root from the adjacency tree of all root candidates.

# Arguments

  - `A`: `N × N` adjacency matrix of the MPG.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.
  - `CliqRoot`: Vector of indices of root candidate cliques.

# Details

  - For each root candidate clique, the function checks which other root cliques share exactly two vertices (i.e., are adjacent in the clique graph).
  - The resulting adjacency matrix is symmetric and encodes the adjacency structure among root cliques.
  - Used to build a connected root structure when multiple root candidates exist in the DBHT hierarchy.

# Returns

  - `Adj::SparseMatrixCSC{Int, Int}`: `Nc×Nc` adjacency matrix of the cliques, where `Adj[i, j] = 1` if cliques `i` and `j` are adjacent among the root candidates.

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
        Indicator[CliqList[CliqRoot[n], :]] .= 1
        Indi = hcat(Indicator[CliqList[CliqRoot, 1]], Indicator[CliqList[CliqRoot, 2]],
                    Indicator[CliqList[CliqRoot, 3]])

        adjacent = CliqRoot[vec(sum(Indi; dims = 2)) .== 2]
        Adj[adjacent, n] .= 1
    end
    Adj = Adj + transpose(Adj)

    return Adj
end
"""
    BubbleHierarchy(Pred::VecNum, Sb::VecNum)

Build the bubble hierarchy from the clique hierarchy and separating set information.

This function constructs the bubble hierarchy tree and the bubble membership matrix for 3-cliques, given the predicted parent indices (`Pred`) and separating set vector (`Sb`). It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline, grouping 3-cliques into bubbles and building the adjacency structure among bubbles.

# Arguments

  - `Pred`: `Nc×1` vector of predicted parent indices for each 3-clique, as returned by [`BuildHierarchy`](@ref).
  - `Sb`: `Nc×1` vector indicating the size of the separating set for each 3-clique (`Sb[n] ≠ 0` means clique `n` is separating).

# Details

  - The function iteratively groups 3-cliques into bubbles, starting from root cliques and traversing the hierarchy.
  - For each bubble, the membership of 3-cliques is recorded in `Mb`.
  - The adjacency matrix `H` encodes the connections between bubbles, based on shared membership and hierarchical relationships.
  - If there are multiple root cliques, an initial bubble is created for each root.
  - Used internally by [`CliqHierarchyTree2s`](@ref) and DBHT clustering routines.

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

# Arguments

  - `::UniqueRoot`: Root selection method enforcing a unique root.
  - `Root`: Vector of indices of root candidate cliques.
  - `Pred`: Vector of predicted parent indices for each clique.
  - `Nc`: Number of 3-cliques.
  - `args...`: Additional arguments (ignored for this method).

# Details

  - If there is more than one root candidate, a synthetic root node is appended and all root candidates are connected to it.
  - The resulting matrix encodes the parent-child relationships among cliques, ensuring a single connected hierarchy.
  - Used internally by DBHT clustering and related routines.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix representing the hierarchical tree of 3-cliques.

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

# Arguments

  - `::EqualRoot`: Root selection method that creates a root from the adjacency tree of all root candidates.
  - `Root`: Vector of indices of root candidate cliques.
  - `Pred`: Vector of predicted parent indices for each clique.
  - `Nc`: Number of 3-cliques.
  - `A`: `N × N` adjacency matrix of the MPG.
  - `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.

# Details

  - If there are multiple root candidates, their adjacency structure is computed using [`AdjCliq`](@ref) and incorporated into the hierarchy.
  - The resulting matrix encodes both the parent-child relationships from `Pred` and the adjacency among root cliques.
  - Used internally by DBHT clustering to support alternative root strategies.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix representing the hierarchical tree of 3-cliques.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`CliqueRoot`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
"""
function CliqueRoot(::EqualRoot, Root::VecNum, Pred::VecNum, Nc::Integer, A::MatNum,
                    CliqList::MatNum)
    if length(Root) > 1
        Adj = AdjCliq(A, CliqList, Root)
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
    CliqHierarchyTree2s(Apm::MatNum; root::DBHTRootMethod = UniqueRoot())

Construct the clique and bubble hierarchy trees for a Maximal Planar Graph (MPG) using the DBHT (Direct Bubble Hierarchical Tree) approach.

This function builds the hierarchical structure of 3-cliques (triangles) and bubbles from the adjacency matrix of a planar graph, supporting different root selection strategies via the `root` argument. It is a core routine for DBHT clustering and related phylogenetic analyses.

# Arguments

  - `Apm`: Adjacency matrix of the MPG, where nonzero entries indicate edges.
  - `root`: Root selection method for the clique hierarchy.

# Details

  - The function first identifies all 3-cliques in the graph and computes their separating sets.
  - It then builds the clique hierarchy using the specified root selection method.
  - The bubble hierarchy is constructed from the clique hierarchy and separating sets.
  - Used internally by DBHT clustering and for extracting hierarchical structures from planar graphs.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix representing the hierarchical tree of 3-cliques.
  - `H2::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix representing the bubble hierarchy tree.
  - `Mb::Matrix{Int}`: Bubble membership matrix for 3-cliques (`Nc×Nb`), where `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `CliqList::Matrix{Int}`: List of 3-cliques (`Nc×3`), each row contains the vertex indices of a 3-clique.
  - `Sb::Vector{Int}`: Vector indicating the size of the separating set for each 3-clique.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`PMFG_T2s`](@ref)
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

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Hb`: Undirected bubble tree of the PMFG (as from [`BubbleHierarchy`](@ref)).
  - `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N×Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.

# Details

  - For each edge in the undirected bubble tree, the function determines the direction by removing the edge and comparing the sum of edge weights for the separating clique on each side.
  - The resulting directed tree encodes the flow of hierarchical structure among bubbles, which is used for cluster assignment and further phylogenetic analysis.
  - Used internally by [`BubbleCluster8s`](@ref) and DBHT clustering routines.

# Returns

  - `Hc::SparseMatrixCSC{Number, Int}`: `Nb×Nb` unweighted directed adjacency matrix of the DBHT. `Hc[i, j] = 1` indicates a directed edge from bubble `i` to bubble `j`.
  - `Sep::Vector{Int}`: Vector indicating the type of each bubble (e.g., converging, diverging, or neutral).

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
    Sep[vec(iszero.(sum(Hc; dims = 1))) .&& kb .> 1] .= 2

    return Hc, Sep
end
"""
    BubbleCluster8s(Rpm::MatNum, Dpm::MatNum,
                    Hb::MatNum, Mb::MatNum,
                    Mv::MatNum, CliqList::MatNum)

Obtain non-discrete and discrete clusterings from the bubble topology of the Planar Maximally Filtered Graph (PMFG).

This function assigns each vertex to a cluster based on the directed bubble hierarchy tree (DBHT) structure. It computes both a non-discrete cluster membership matrix and a discrete cluster assignment vector, using the converging bubbles identified in the directed bubble tree.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Dpm`: `N × N` shortest path lengths matrix of the PMFG.
  - `Hb`: Undirected bubble tree of the PMFG (from [`BubbleHierarchy`](@ref)).
  - `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicates 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N×Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique in the MPG.

# Details

  - The function first computes the directed bubble hierarchy tree using [`DirectHb`](@ref).
  - Converging bubbles are identified as cluster centers.
  - Non-discrete cluster membership (`Adjv`) is determined by traversing the directed bubble tree from each converging bubble.
  - Discrete cluster assignments (`Tc`) are made by resolving overlaps and assigning each vertex to the most strongly associated converging bubble, or, if ambiguous, to the closest converging bubble by shortest path.
  - Used internally by DBHT clustering and for extracting cluster assignments from the PMFG bubble structure.

# Returns

  - `Adjv::SparseMatrixCSC{Int, Int}`: `N×Nk` cluster membership matrix for vertices for non-discrete clustering via the bubble topology. `Adjv[n, k] = 1` indicates cluster membership of vertex `n` to the `k`-th non-discrete cluster.
  - `Tc::Vector{Int}`: `N × 1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`-th discrete cluster.

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

This function determines the bubble membership of each vertex, resolving ambiguities when a vertex could belong to multiple bubbles. Assignment is based on the strength of connections (edge weights) between the vertex and each candidate bubble.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Mv`: `N×Nb` bubble membership matrix for vertices. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `Mc`: Matrix indicating bubbles that coincide with clusters.

# Details

  - Vertices belonging to a single bubble are assigned directly.
  - For vertices that could belong to multiple bubbles, assignment is made to the bubble with the strongest normalized connection (fraction of edge weights).
  - Used internally for intra- and inter-cluster hierarchy construction in DBHT clustering.

# Returns

  - `Mvv::Matrix{Int}`: `N×Nb` matrix where `Mvv[n, bi] = 1` if vertex `n` is assigned to bubble `bi`.

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

# Inputs

  - `Zi`: Linkage matrix at iteration `i` in the same format as the output from Matlab.
  - `LabelVec1`: Label vector for the vertices in the bubble for the previous valid iteration.
  - `LabelVec2`: Label vector for the vertices in the bubble for the trial iteration.
  - `LinkageDist`: Linkage distance(s) for the current merge.

# Details

  - The function identifies which clusters have changed between `LabelVec1` and `LabelVec2` and appends a new row to the linkage matrix for the merge.
  - The linkage matrix `Z` can be converted to a format compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) using [`turn_into_Hclust_merges`](@ref).
  - Used internally by [`HierarchyConstruct4s`](@ref) and related routines for DBHT dendrogram construction.

# Returns

  - `Z::MatNum`: Linkage matrix at iteration `i + 1` in the same format as the output from Matlab.

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

Find the pair of clusters with the best linkage in a bubble.

This function searches for the pair of clusters (as indicated by `labelvec`) with the strongest linkage according to the provided distance matrix `d`. The best linkage is defined as the pair with the maximum inter-cluster distance among all pairs of clusters in the bubble. Used internally for hierarchical linkage construction in DBHT dendrogram routines.

# Inputs

  - `d`: `Nv×Nv` distance matrix for the vertices assigned to a bubble.
  - `labelvec`: Label vector for the vertices in the bubble.

# Details

  - For each unique pair of cluster labels, the function computes the maximum distance between their members.
  - Returns the pair with the largest such distance and the corresponding value.
  - Used in [`build_link_and_dendro`](@ref) and [`HierarchyConstruct4s`](@ref) to determine which clusters to merge at each step.

# Returns

  - `PairLink::Vector{Int}`: Pair of cluster labels with the best linkage.
  - `dvu::Number`: Value of the best linkage (maximum inter-cluster distance).

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
```
build_link_and_dendro(rg::AbstractRange, dpm::MatNum,
                      LabelVec::VecNum, LabelVec1::VecNum,
                      LabelVec2::VecNum, V::VecNum,
                      nc::Number, Z::MatNum)
```

Iteratively construct the linkage matrix for a bubble or cluster.

This function iterates over the vertices in a bubble or cluster, merging the pair of clusters with the best linkage at each step (as determined by [`LinkageFunction`](@ref)), and appending the corresponding row to the linkage matrix using [`DendroConstruct`](@ref). Used internally for building dendrograms in DBHT and related hierarchical clustering routines.

# Inputs

  - `rg`: Range of indices for the vertices in the bubble or cluster.
  - `dpm`: Distance matrix for the vertices assigned to the bubble or cluster.
  - `LabelVec`: Current label vector for the clusters.
  - `LabelVec1`: Label vector for the previous valid iteration.
  - `LabelVec2`: Label vector for the trial iteration.
  - `V`: Indices of the vertices in the bubble or cluster.
  - `nc::Number`: Inverse of the linkage distance (or a counter for the merge steps).
  - `Z`: Current linkage matrix.

# Details

  - At each iteration, finds the pair of clusters with the best linkage using [`LinkageFunction`](@ref).
  - Merges the pair by updating the label vector, and appends a new row to the linkage matrix using [`DendroConstruct`](@ref).
  - Continues until all clusters in the range are merged.

# Returns

  - `Z::MatNum`: Updated linkage matrix after all merges in the range.
  - `nc::Number`: Updated inverse linkage distance or merge counter.
  - `LabelVec1::VecNum`: Updated label vector for the next iteration.

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
        LabelVec[LabelVec .== PairLink[1] .|| LabelVec .== PairLink[2]] .= maximum(LabelVec1) +
                                                                           1  # Merge the cluster pair by updating the label vector with a same label.
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copy(LabelVec2)
    end
    return Z, nc, LabelVec1
end
"""
```
HierarchyConstruct4s(Rpm::MatNum, Dpm::MatNum,
                     Tc::VecNum, Mv::MatNum)
```

Constructs the intra- and inter-cluster hierarchy by utilizing the Bubble Hierarchy structure of a Maximal Planar Graph, specifically a Planar Maximally Filtered Graph (PMFG).

This function builds a hierarchical clustering (dendrogram) by first constructing intra-cluster linkages within each cluster (using the bubble structure), and then merging clusters to form the global hierarchy. It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline.

# Inputs

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Dpm`: `N × N` shortest path lengths matrix of the PMFG.
  - `Tc`: `N × 1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`-th discrete cluster.
  - `Mv`: `N×Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.

# Details

    - For each cluster, the function identifies the bubbles that coincide with the cluster and assigns each vertex to a specific bubble using [`BubbleMember`](@ref).
    - It constructs intra-bubble and intra-cluster linkages using [`build_link_and_dendro`](@ref).
    - After intra-cluster linkage, it merges clusters to form the global hierarchy using inter-cluster linkage steps.
    - The resulting linkage matrix can be converted to a format compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) using [`turn_into_Hclust_merges`](@ref).
    - Used internally by DBHT clustering routines for dendrogram construction.

# Returns

  - `Z::MatNum`: `(N-1)×3` linkage matrix in the same format as the output from Matlab, suitable for conversion to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

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
        LabelVec2[LabelVec1 .== PairLink[1] .|| LabelVec1 .== PairLink[2]] .= maximum(LabelVec1) +
                                                                              1
        dvu = unique(dcl[LabelVec1 .== PairLink[1]]) +
              unique(dcl[LabelVec1 .== PairLink[2]])
        dcl[LabelVec1 .== PairLink[1] .|| LabelVec1 .== PairLink[2]] .= dvu
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        LabelVec1 = copy(LabelVec2)
    end

    return Z
end
"""
    turn_into_Hclust_merges(Z::MatNum)

Convert a Matlab-style linkage matrix to a format compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

This function transforms a linkage matrix produced by DBHT or similar hierarchical clustering routines into the format required by [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), including proper indexing and cluster size tracking.

# Inputs

  - `Z`: Matlab-style linkage matrix, where each row represents a merge step with cluster indices and linkage heights.

# Details

  - For each merge, leaf indices are converted to negative values, and cluster sizes are accumulated in the fourth column.
  - Internal cluster indices are updated to reference the correct merged clusters.
  - The resulting matrix can be passed directly to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) for dendrogram construction and further analysis.

# Returns

  - `Z::MatNum`: Linkage matrix in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format, with updated indices and cluster sizes.

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
          root::DBHTRootMethod = UniqueRoot())

Perform Direct Bubble Hierarchical Tree clustering, a deterministic clustering algorithm [DBHTs](@cite). This version uses a graph-theoretic filtering technique called Triangulated Maximally Filtered Graph (TMFG).

This function implements the full DBHT clustering pipeline: it constructs a Planar Maximally Filtered Graph (PMFG) from the similarity matrix, extracts the clique and bubble hierarchies, assigns clusters, and builds a hierarchical clustering (dendrogram) compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Arguments

  - `D`: `N × N` dissimilarity matrix (e.g., a distance matrix). Must be symmetric and non-empty.
  - `S`: `N × N` non-negative similarity matrix. Must be symmetric and non-empty.
  - `branchorder`: Ordering method for the dendrogram branches. Accepts `:optimal`, `:barjoseph`, or `:r`.
  - `root`: Root selection method for the clique hierarchy.

# Validation

  - `!isempty(D) && LinearAlgebra.issymmetric(D)`.
  - `!isempty(S) && LinearAlgebra.issymmetric(S)`.
  - `size(D) == size(S)`.

# Details

  - Validates that `D` and `S` are non-empty, symmetric, and of equal size.
  - Constructs the PMFG using [`PMFG_T2s`](@ref).
  - Computes shortest path distances on the PMFG.
  - Extracts clique and bubble hierarchies using [`CliqHierarchyTree2s`](@ref) and [`BubbleHierarchy`](@ref).
  - Assigns clusters using [`BubbleCluster8s`](@ref).
  - Builds the hierarchical clustering using [`HierarchyConstruct4s`](@ref) and converts it to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format.
  - Supports different root selection strategies and dendrogram branch orderings.

# Returns

  - `T8::Vector{Int}`: `N × 1` cluster membership vector.
  - `Rpm::SparseMatrixCSC{<:Number, Int}`: `N × N` adjacency matrix of the Planar Maximally Filtered Graph (PMFG).
  - `Adjv::SparseMatrixCSC{Int, Int}`: Bubble cluster membership matrix from [`BubbleCluster8s`](@ref).
  - `Dpm::Matrix{<:Number}`: `N × N` shortest path length matrix of the PMFG.
  - `Mv::SparseMatrixCSC{Int, Int}`: `N×Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
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
               root::DBHTRootMethod = UniqueRoot())
    @argcheck(!isempty(S), IsEmptyError)
    @argcheck(!isempty(D), IsEmptyError)
    @argcheck(size(S) == size(D), DimensionMismatch)
    Rpm = PMFG_T2s(S)[1]
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

# Arguments

  - `jlogo`: The matrix to be updated in-place.
  - `sigma`: The full covariance matrix.
  - `source`: Each row contains indices of a clique or separator (e.g., 4-cliques or 3-cliques).
  - `sign`: +1 for cliques, -1 for separators.

# Details

  - For each row in `source`, the function extracts the submatrix of `sigma` corresponding to the clique/separator.
  - The inverse of this submatrix is computed and added to (or subtracted from) the corresponding block in `jlogo`.
  - Used internally by [`J_LoGo`](@ref) to efficiently compute the sparse inverse covariance matrix for LoGo/DBHT.

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

# Arguments

  - `sigma`: The covariance matrix (`N × N`).
  - `separators`: Each row contains indices of a separator (typically 3-cliques).
  - `cliques`: Each row contains indices of a clique (typically 4-cliques).

# Details

  - For each clique, the inverse of the corresponding submatrix of `sigma` is added to the output.
  - For each separator, the inverse of the corresponding submatrix is subtracted.
  - The resulting matrix is the sparse inverse covariance estimate, as described in the LoGo methodology.
  - Used internally by [`LoGo`](@ref) and related estimators.

# Returns

  - `jlogo::Matrix{<:Number}`: The LoGo sparse inverse covariance matrix.

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

# Arguments

  - `cle`: A `ClustersEstimator` whose algorithm is a [`DBHT`](@ref) instance.
  - `X`: Data matrix (`observations × assets` or `assets × observations` depending on `dims`).
  - `branchorder`: Symbol specifying the dendrogram branch ordering method. Accepts `:optimal` (default), `:barjoseph`, or `:r`.
  - `dims`: Integer specifying the dimension along which to compute statistics (`1` for columns/assets, `2` for rows).
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Details

  - Computes the similarity and distance matrices using the estimator's configured correlation and distance estimators.
  - Applies the selected similarity transformation via [`dbht_similarity`](@ref).
  - Runs the full DBHT clustering pipeline via [`DBHTs`](@ref), including PMFG construction, clique and bubble hierarchy extraction, and dendrogram construction.
  - Determines the optimal number of clusters using the estimator's cluster selection method.
  - Returns a [`Clusters`](@ref) result encapsulating all relevant outputs.

# Returns

  - `clr::Clusters`: DBHT clustering result.

# Related

  - [`DBHT`](@ref)
  - [`Clusters`](@ref)
  - [`DBHTs`](@ref)
  - [`dbht_similarity`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
                    branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    S = dbht_similarity(cle.alg.sim; S = S, D = D)
    res = DBHTs(D, S; branchorder = branchorder, root = cle.alg.root)[end]
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
function logo!(::Nothing, args...; kwargs...)
    return nothing
end
"""
    abstract type InverseMatrixSparsificationAlgorithm <: AbstractMatrixProcessingAlgorithm end

Abstract supertype for all inverse matrix sparsification algorithms in PortfolioOptimisers.jl.

# Related

  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`LoGo`](@ref)
"""
abstract type InverseMatrixSparsificationAlgorithm <: AbstractMatrixProcessingAlgorithm end
"""
    struct LoGo{T1, T2, T3} <: InverseMatrixSparsificationAlgorithm
        dist::T1
        sim::T2
        pdm::T3
    end

LoGo (Local-Global) sparse inverse covariance estimation algorithm.

`LoGo` is a composable algorithm type for estimating sparse inverse covariance matrices using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition, as described in [J_LoGo](@cite). It combines a distance estimator and a similarity matrix algorithm, both validated and extensible, to produce a robust, interpretable sparse precision matrix for use in portfolio optimization and risk management.

# Fields

  - `dist`: Distance matrix estimator.
  - `sim`: Similarity matrix algorithm.
  - `pdm`: Optional Positive definite matrix estimator. If provided, ensures the output is positive definite.

# Constructor

    LoGo(; dist::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
         sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
         pdm::Option{<:Posdef} = Posdef())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> LoGo()
LoGo
  dist ┼ Distance
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
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
"""
struct LoGo{T1, T2, T3} <: InverseMatrixSparsificationAlgorithm
    dist::T1
    sim::T2
    pdm::T3
    function LoGo(dist::AbstractDistanceEstimator, sim::AbstractSimilarityMatrixAlgorithm,
                  pdm::Option{<:Posdef} = Posdef())
        return new{typeof(dist), typeof(sim), typeof(pdm)}(dist, sim, pdm)
    end
end
function LoGo(; dist::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
              sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              pdm::Option{<:Posdef} = Posdef())
    return LoGo(dist, sim, pdm)
end
const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                 <:DistanceDistance{<:Any, <:VariationInfoDistance, <:Any,
                                                    <:Any, <:Any}}
"""
    LoGo_dist_assert(dist::AbstractDistanceEstimator, sigma::MatNum, X::MatNum)

Validate compatibility of the distance estimator and covariance matrix for LoGo sparse inverse covariance estimation by checking `size(sigma, 1) == size(X, 2)`.

# Arguments

  - `dist`: Distance estimator, typically a subtype of `AbstractDistanceEstimator`.
  - `sigma`: Covariance matrix (`N × N`).
  - `X`: Data matrix (`T × N` or `N × T`).

# Returns

  - `nothing`.

# Validation

  - `size(sigma, 1) == size(X, 2)`.

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

# Returns

  - `nothing`.
"""
function LoGo_dist_assert(args...)
    return nothing
end
"""
    logo!(je::LoGo, sigma::MatNum, X::MatNum;
          dims::Int = 1, kwargs...)

Compute the LoGo (Local-Global) covariance matrix and update `sigma` in-place.

This method implements the LoGo algorithm for sparse inverse covariance estimation using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition. It validates inputs, computes the similarity and distance matrices, constructs the PMFG, identifies cliques and separators, and updates the input covariance matrix `sigma` in-place by inverting the LoGo sparse inverse covariance estimate. The result is projected to the nearest positive definite matrix if a `Posdef` estimator is not `nothing`.

# Arguments

  - `je`: LoGo algorithm instance.
  - `sigma`: Covariance matrix (`N × N`), updated in-place with the LoGo sparse inverse covariance.
  - `X`: Data matrix (`T × N`).
  - `dims`: Dimension along which to compute statistics (`1` for columns/assets, `2` for rows). Default: `1`.
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.
  - Computes the distance matrix using the configured distance estimator.
  - Computes the similarity matrix using the configured similarity algorithm.
  - Constructs the PMFG and extracts cliques and separators.
  - Computes the LoGo sparse inverse covariance matrix via [`J_LoGo`](@ref).
  - Updates `sigma` in-place with the inverse of the LoGo estimate.
  - Projects the result to the nearest positive definite matrix if `pdm` is not `nothing`.

# Validation

    - `size(sigma, 1) == size(sigma, 2)`.
    - `size(sigma, 1) == size(X, 2)`.

# Returns

  - `nothing`. The input `sigma` is updated in-place.

# Related

  - [`LoGo`](@ref)
  - [`J_LoGo`](@ref)
  - [`LoGo_dist_assert`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`dbht_similarity`](@ref)
  - [`Posdef`](@ref)
"""
function logo!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    assert_matrix_issquare(sigma, :sigma)
    LoGo_dist_assert(je.dist, sigma, X)
    s = LinearAlgebra.diag(sigma)
    iscov = any(!isone, s)
    S = if iscov
        s .= sqrt.(s)
        StatsBase.StatsBase.cov2cor(sigma, s)
    else
        sigma
    end
    D = distance(je.dist, S, X; dims = dims, kwargs...)
    S = dbht_similarity(je.sim; S = S, D = D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    sigma .= J_LoGo(sigma, separators, cliques) \ LinearAlgebra.I
    posdef!(je.pdm, sigma)
    return nothing
end
function logo(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    sigma = copy(sigma)
    logo!(je, sigma, X; dims = dims, kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm!(je::LoGo, sigma::MatNum,
                                 X::MatNum; dims::Int = 1, kwargs...)

Apply the LoGo (Local-Global) transformation in-place to the covariance matrix as a matrix processing algorithm to.

This method provides a standard interface for applying the LoGo algorithm to a covariance matrix within the matrix processing pipeline of PortfolioOptimisers.jl. It validates inputs, computes the LoGo sparse inverse covariance matrix, and updates `sigma` in-place. If a positive definite matrix estimator (`pdm`) is not `nothing`, the result is projected to the nearest positive definite matrix.

# Arguments

  - `je`: LoGo algorithm instance (`LoGo`).
  - `pdm`: Optional positive definite matrix estimator (e.g., `Posdef()`), or `nothing`.
  - `sigma`: Covariance matrix (`N × N`), updated in-place.
  - `X`: Data matrix (`T × N` or `N × T`).
  - `dims`: Dimension along which to compute statistics (`1` for columns/assets, `2` for rows). Default: `1`.
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Details

  - Internally, it calls [`logo!`](@ref) to perform the LoGo sparse inverse covariance estimation and update `sigma` in-place.
  - Used in composable workflows for covariance matrix estimation.

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

export ExponentialSimilarity, GeneralExponentialSimilarity, MaximumDistanceSimilarity,
       UniqueRoot, EqualRoot, DBHT, LoGo, Clusters
