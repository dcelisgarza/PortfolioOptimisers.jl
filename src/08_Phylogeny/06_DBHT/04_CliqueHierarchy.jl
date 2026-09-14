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
