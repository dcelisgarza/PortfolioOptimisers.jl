"""
    clique3(A::MatNum)

List the 3-cliques of a Maximal Planar Graph (MPG).

The function finds every 3-clique, a triangle, of the graph that `A` holds. It returns the candidate edges, the third vertices that close a triangle on each candidate edge, and the list of the distinct 3-cliques.

# Algorithm

 1. Remove the diagonal of `A` and reduce it to a binary matrix.
 2. Form `A2 = A * A`, whose entry counts the paths of length two between a pair.
 3. Keep the upper triangle of the entries where `A2` and `A` are both non-zero, which gives `P`. A stored entry of `P` is an edge whose two ends share at least one neighbour.
 4. Read the row and the column index of each stored entry of `P` into the two columns of `E`, one row for each candidate edge.
 5. For each candidate edge, intersect the neighbourhoods of its two ends, which gives `K3[n]`, the third vertices that close a triangle on the edge.
 6. Sort each triple `(E[n, 1], E[n, 2], K3[n][m])` and keep it when it is new. The function finds a triangle from each of its three edges, and it stores the triangle once.
 7. Sort the triples, and write them as the rows of `clique`.

# Arguments

  - `A`: `N × N` adjacency matrix of a Maximal Planar Graph (MPG). A non-zero entry is an edge.

# Returns

  - `K3::Vector{Vector{Int}}`: `K3[n]` holds the vertices that close a triangle on the candidate edge of row `n` of `E`.
  - `E::Matrix{Int}`: `Ne×2` matrix of the candidate edges, one edge for each row, with the smaller vertex in the first column.
  - `clique::Matrix{Int}`: `Nc×3` matrix. Each row lists the three vertices of one 3-clique of the MPG in ascending order, and the rows are in lexicographic order.

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

    triples = Set{NTuple{3, Int}}()
    for n in eachindex(r)
        for k in K3[n]
            # The triple in ascending order, so each triangle has one key.
            lo, hi = minmax(E[n, 1], E[n, 2])
            lo, mid = minmax(lo, k)
            mid, hi = minmax(mid, hi)
            push!(triples, (lo, mid, hi))
        end
    end

    rows = sort!(collect(triples))
    clique = [t[col] for t in rows, col in 1:3]

    return K3, E, clique
end
"""
    FindDisjoint(Adj::MatNum, Cliq::VecNum)

Split the vertices of a graph into a 3-clique and the sides that the clique separates.

The function cuts the three vertices of `Cliq` out of the graph. It then labels each vertex as a vertex of the clique, a vertex that `IndxNot[1]` can reach, or a vertex that `IndxNot[1]` cannot reach.

# Algorithm

 1. Copy `Adj` into `Temp`, and collect in `IndxNot` every vertex that is not one of the three of `Cliq`.
 2. Set the rows and the columns of `Temp` at `Cliq` to zero. This cuts the clique out of the graph, and it separates the sides that the clique joins.
 3. Run [`breadth`](@ref) from `IndxNot[1]`, which gives `d`, and mark with `-1` each vertex at an infinite distance.
 4. Write `1` into `T` at every vertex marked `-1`, and `2` at every other vertex, so `2` is the side that holds `IndxNot[1]`.
 5. Write `0` into `T` at the three vertices of `Cliq`.

# Arguments

  - `Adj`: `N × N` adjacency matrix of the MPG. A non-zero entry is an edge.
  - `Cliq`: `3×1` vector of the vertices of a 3-clique.

# Returns

  - `T::Vector{Int}`: `N × 1` vector of the side of each vertex.

      + `0` for a vertex of the clique.
      + `1` for a vertex that `IndxNot[1]` cannot reach in the graph without the clique.
      + `2` for a vertex that `IndxNot[1]` can reach, `IndxNot[1]` included.

  - `IndxNot::Vector{Int}`: `(N - 3)×1` vector of every vertex that is not a vertex of `Cliq`, in ascending order.

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

Find the parent of each 3-clique of a Maximal Planar Graph (MPG).

The function builds the parent vector `Pred` from the matrix `M`. Column `n` of `M` marks the vertices of 3-clique `n` and of its interior. The parent of a 3-clique is the 3-clique with the fewest marked vertices whose marks hold every mark of it. [`CliqHierarchyTree2s`](@ref) calls this function, and it builds the clique hierarchy from `Pred`.

# Algorithm

 1. For each 3-clique `n`, read `Children`, the vertices that column `n` of `M` marks.
 2. Sum the rows of `M` over `Children`, and take as `Parents` every clique whose sum equals `length(Children)`. The column of such a clique marks every vertex that column `n` marks, so the clique is a superset of clique `n`. Drop `n` from that list.
 3. Set `Pred[n] = 0` when `Parents` is empty, which makes clique `n` a root candidate.
 4. Otherwise set `Pred[n]` to the parent with the fewest marked vertices, the smallest superset.
 5. When two parents have the same smallest count, return an empty vector. The empty vector tells the caller that the function built no hierarchy, and the loop stops at that point.

# Arguments

  - `M`: `N × Nc` binary matrix. `M[i, n] = 1` when vertex `i` is a vertex of 3-clique `n` or of its interior. Step 3 of [`CliqHierarchyTree2s`](@ref) states which side of the graph is the interior.

# Returns

  - `Pred::Vector{Int}`: `Nc×1` vector of the parent of each 3-clique, an index into the columns of `M`. `Pred[n] = 0` marks a root candidate. The vector is empty when step 5 of the algorithm returns.

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
        Children = findall(==(1), view(M, :, n))
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

Find the pairs of root candidates of a Maximal Planar Graph (MPG) that share an edge.

The function computes the adjacency matrix of the root candidates, which are 3-cliques. Two root candidates are adjacent when they share exactly two vertices. The [`EqualRoot`](@ref) method of [`CliqueRoot`](@ref) calls it to join the root candidates to each other.

The function reads only the size of `A`, and no edge of the MPG changes the result. The result loses nothing by that. Every row of `CliqList` is a 3-clique of the MPG, so two rows that share two vertices both hold the edge between those two vertices. Thus the count of the shared vertices is the test for adjacency in the graph.

# Algorithm

 1. Clear `Indicator`, then mark in it the three vertices of root candidate `n`.
 2. Read `Indicator` back at the three vertex columns of every root candidate, which gives `Indi`.
 3. Take the root candidates whose row of `Indi` sums to `2`, and set their entries of column `CliqRoot[n]` of `Adj` to one.
 4. Repeat from step 1 for the next candidate. The test of step 3 is symmetric, so `Adj` needs no symmetrisation.

# Arguments

  - `A`: `N × N` adjacency matrix of the MPG. The function reads only `size(A, 1)`, which sets the length of `Indicator`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique of the MPG.
  - `CliqRoot`: Vector of the indices of the root candidates, which index the rows of `CliqList`.

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
    RootList = CliqList[CliqRoot, 1:3]
    Indi = similar(Indicator, length(CliqRoot), 3)
    for n in eachindex(CliqRoot)
        Indicator .= 0
        Indicator[CliqList[CliqRoot[n], :]] .= 1
        Indi .= view(Indicator, RootList)

        adjacent = CliqRoot[vec(sum(Indi; dims = 2)) .== 2]
        Adj[adjacent, CliqRoot[n]] .= 1
    end

    return Adj
end
"""
    CliqueRoot(::UniqueRoot, Root::VecNum, Pred::VecNum, Nc::Integer, args...)

Build the adjacency matrix of the clique hierarchy of a Maximal Planar Graph (MPG) with one root.

When the hierarchy has more than one root candidate, this method adds a synthetic root clique and makes it the parent of every root candidate. [`CliqHierarchyTree2s`](@ref) calls it when its `root` is [`UniqueRoot`](@ref).

# Algorithm

 1. When more than one root candidate exists, append a synthetic clique to `Pred` and set the parent of every root candidate to it. The method changes `Pred` in place, so the vector of the caller gets that entry.
 2. Allocate `H` over `Nc + 1` rows and columns, which is the room the synthetic clique of step 1 needs.
 3. Write `H[n, Pred[n]] = 1` for every clique that has a parent.
 4. Symmetrise `H`.

# Arguments

  - `::UniqueRoot`: Root selection method that makes one root.
  - `Root`: Vector of the indices of the root candidates, which index the entries of `Pred`.
  - `Pred`: `Nc×1` vector of the parent of each 3-clique, from [`BuildHierarchy`](@ref).
  - `Nc`: Number of 3-cliques.
  - `args...`: Positional arguments, which this method ignores. [`CliqHierarchyTree2s`](@ref) passes the adjacency matrix and the clique list here. The [`EqualRoot`](@ref) method reads them, and this method does not.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `(Nc + 1)×(Nc + 1)` symmetric adjacency matrix of the hierarchy tree of the 3-cliques. Row and column `Nc + 1` hold the synthetic root. They are empty when the hierarchy has one root candidate.

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

Build the adjacency matrix of the clique hierarchy of a Maximal Planar Graph (MPG), with the root candidates joined to each other.

This method adds no root clique. It joins two root candidates when they share two vertices, so the hierarchy keeps every root candidate. [`CliqHierarchyTree2s`](@ref) calls it when its `root` is [`EqualRoot`](@ref).

# Algorithm

 1. When more than one root candidate exists, build the adjacency `Adj` of the candidates with [`AdjCliq`](@ref). One candidate has no other candidate to join, so `Adj` is a zero matrix in that case, and step 4 adds nothing.
 2. Allocate `H` over `Nc` rows and columns. This method adds no synthetic clique, so it needs no extra row.
 3. Write `H[n, Pred[n]] = 1` for every clique that has a parent.
 4. Return a `0 × 0` matrix when `Pred` is empty. Otherwise symmetrise `H` and add `Adj` to it, which joins the root candidates to each other.

# Arguments

  - `::EqualRoot`: Root selection method that joins the root candidates to each other.
  - `Root`: Vector of the indices of the root candidates, which index the entries of `Pred`.
  - `Pred`: `Nc×1` vector of the parent of each 3-clique, from [`BuildHierarchy`](@ref).
  - `Nc`: Number of 3-cliques.
  - `A`: `N × N` adjacency matrix of the MPG. The method passes it to [`AdjCliq`](@ref), which reads only its size.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique of the MPG.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `Nc×Nc` symmetric adjacency matrix of the hierarchy of the 3-cliques, or a `0 × 0` matrix when `Pred` is empty.

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

Build the clique hierarchy and the bubble tree of a Maximal Planar Graph (MPG), as the Direct Bubble Hierarchical Tree (DBHT) method needs them.

The function builds the hierarchy of the 3-cliques and the tree of the bubbles from the adjacency matrix of a planar graph. The `root` argument selects the rule for the root of the clique hierarchy.

`root` is a positional argument, and every caller passes it as one.

# Algorithm

 1. Reduce `Apm` to the binary adjacency `A`, and list every 3-clique of it with [`clique3`](@ref), which gives `CliqList`.
 2. For each 3-clique, cut it out of the graph with [`FindDisjoint`](@ref), which splits the vertices into the clique `indx0`, the side `indx1` that the cut separated, and the side `indx2` that stayed connected.
 3. Take the smaller of the two sides, together with the clique, as the separated set `indx_s`. A tie takes `indx1`, the side that the cut separated. The smaller side is the interior of the clique in [NHPG](@cite), Definition 2.2, and that definition does not decide a tie. Record `length(indx_s) - 3` in `Sb[n]`, and mark the vertices of `indx_s` in column `n` of `M`.
 4. Build the parent vector `Pred` from `M` with [`BuildHierarchy`](@ref), and read the root candidates `Root` off it.
 5. Build the clique hierarchy `H` with [`CliqueRoot`](@ref), through the branch `root` selects.
 6. When `H` is non-empty, build the bubble hierarchy `H2` and the bubble membership `Mb` with [`BubbleHierarchy`](@ref), reduce `H2` to binary, and trim `Mb` to the rows of `CliqList`.
 7. Return `0 × 0` matrices for `H2` and `Mb` when `H` is empty.

# Arguments

  - `Apm`: `N × N` adjacency matrix of the MPG. A non-zero entry is an edge. The function reads only the sparsity pattern, so a weighted matrix and its binary form give the same result.
  - `root`: Root selection method for the clique hierarchy.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: Symmetric adjacency matrix of the hierarchy of the 3-cliques. The [`CliqueRoot`](@ref) method that `root` selects sets its size.
  - `H2::SparseMatrixCSC`: `Nb×Nb` symmetric adjacency matrix of the bubble tree with `Bool` entries, where `Nb` is the number of bubbles. It is a `0 × 0` matrix of `Int` when `H` is empty.
  - `Mb::AbstractMatrix{Int}`: `Nc×Nb` dense bubble membership matrix of the 3-cliques. `Mb[n, bi] = 1` means that 3-clique `n` belongs to bubble `bi`. It is a `0 × 0` sparse matrix when `H` is empty.
  - `CliqList::Matrix{Int}`: `Nc×3` list of the 3-cliques. Each row holds the three vertices of one 3-clique.
  - `Sb::Vector{Int}`: `Nc×1` vector of the number of vertices in the interior of each 3-clique. An entry is positive exactly when its 3-clique separates the graph.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`PMFG_T2s`](@ref)

# References

  - $(ref_dict[:NHPG]) Definition 2.2 for the interior, Definition 3.2 and Corollary 3.6 for `H`, and Definitions 4.3 and 5.1 for the bubbles and `H2`.
"""
function CliqHierarchyTree2s(Apm::MatNum, root::DBHTRootMethod = UniqueRoot())
    N = size(Apm, 1)::Int
    A = Apm .!= 0
    K3, E, clique = clique3(A)

    Nc = size(clique, 1)
    M = SparseArrays.spzeros(Int, N, Nc)
    CliqList = copy(clique)
    Sb = zeros(Int, Nc)
    indx_s = Int[]

    for n in axes(clique, 1)
        cliq_vec = CliqList[n, :]
        T = FindDisjoint(A, cliq_vec)[1]
        indx0 = findall(T .== 0)
        indx1 = findall(T .== 1)
        indx2 = findall(T .== 2)

        append!(empty!(indx_s), length(indx1) > length(indx2) ? indx2 : indx1, indx0)

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
