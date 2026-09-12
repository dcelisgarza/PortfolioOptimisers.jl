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
