"""
    BubbleHierarchy(Pred::VecNum, Sb::VecNum)

Build the bubble tree from the parents of the 3-cliques and the sizes of their interiors.

The function puts the 3-cliques into bubbles, from the parent vector `Pred` and the interior sizes `Sb`. It builds the adjacency matrix of the bubbles and the bubble membership matrix of the 3-cliques.

# Algorithm

 1. Take `Root`, the cliques whose entry of `Pred` is zero, and mark them in `CliqCount`.
 2. When more than one root exists, open one bubble that holds all of them, as the first column of `Mb`.
 3. For each root `n`, open a bubble that holds `n` and its direct children, the cliques whose parent is `n`. Append the bubble as a column of `Mb`, and mark those children in `CliqCount`.
 4. Collect as the next roots the direct children whose interior is not empty, where `Sb` is not zero.
 5. Repeat from step 3 until `CliqCount` marks every clique.
 6. Build `H`. Two bubbles are neighbours when at least one 3-clique belongs to both.
 7. Symmetrise `H` and clear its diagonal, so a bubble is not its own neighbour.

# Arguments

  - `Pred`: `Nc×1` vector of the parent of each 3-clique, from [`BuildHierarchy`](@ref).
  - `Sb`: `Nc×1` vector of the number of vertices in the interior of each 3-clique. `Sb[n] ≠ 0` means that clique `n` separates the graph.

# Returns

  - `H::SparseMatrixCSC{Int, Int}`: `Nb×Nb` symmetric adjacency matrix of the bubble tree, where `Nb` is the number of bubbles.
  - `Mb::Matrix{Int}`: `Nc×Nb` bubble membership matrix of the 3-cliques. `Mb[n, bi] = 1` means that 3-clique `n` belongs to bubble `bi`.

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
    # The cliques of each bubble, one entry per column of `Mb`.
    bubbles = Vector{Vector{Int}}(undef, 0)

    if length(Root) > 1
        push!(bubbles, Root)
    end

    while sum(CliqCount) < Nc
        NxtRoot = Int[]
        for n in eachindex(Root)
            DirectChild = findall(Pred .== Root[n])
            push!(bubbles, [Root[n]; DirectChild])
            CliqCount[DirectChild] .= 1

            append!(NxtRoot, Iterators.filter(c -> !iszero(Sb[c]), DirectChild))
        end
        Root = sort!(unique(NxtRoot))
    end
    Mb = zeros(Int, Nc, length(bubbles))
    for (bi, b) in pairs(bubbles)
        Mb[b, bi] .= 1
    end
    Nb = size(Mb, 2)
    H = SparseArrays.spzeros(Int, Nb, Nb)

    for n in axes(Mb, 2)
        Indx = @view(Mb[:, n]) .== 1
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

Give a direction to each edge of the bubble tree of a Maximal Planar Graph (MPG).

Each edge of the undirected bubble tree of a Planar Maximally Filtered Graph (PMFG) holds one separating 3-clique. The function gives each edge a direction, and the result is the directed bubble hierarchical tree (DBHT). The sums of the edge weights on the two sides of the clique set the direction. The directions then show the converging and the diverging bubbles.

# Mathematical definition

Each edge of the bubble tree holds one separating 3-clique. A cut of the edge splits the bubbles into two sides, and the mass that each side draws through the clique sets the direction.

```math
m(\\mathcal{V}) = \\sum_{u \\in \\mathcal{V}_{0}} \\sum_{v \\in \\mathcal{V}} R_{u,\\,v}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``\\mathcal{V}_{0}``: The three vertices of the separating clique.
  - ``\\mathcal{V}``: Vertices of one side of the cut, with ``\\mathcal{V}_{0}`` removed.
  - ``m(\\mathcal{V})``: Mass the clique draws from that side.

The function directs the edge towards the heavier side. A bubble that is the heavier side of each of its edges has no outgoing edge, and it is a converging bubble.

# Algorithm

 1. Reduce `Hb` to binary, and read the row and column index of each edge of its upper triangle.
 2. For each such edge, find the 3-cliques that both of its bubbles hold, and record `(row, column, clique)` as a row of `CliqEdge`.
 3. For each row of `CliqEdge`, remove that edge from a copy of `Hb`, run [`breadth`](@ref) from bubble `1`, and mark every bubble it did not reach with `-1`.
 4. Split the two bubbles of the edge into `bleft`, the one on the reached side, and `bright`, the one on the cut side.
 5. Collect `vleft` and `vright`, the vertices of the bubbles of each side, and remove from both the three vertices `vo` of the separating clique.
 6. Sum the PMFG weights from `vo` into each side, which gives `left` and `right`. Write the heavier of the two into `Hc` as an edge directed towards the heavier side. A tie directs the edge towards `bright`.
 7. Set `Sep[b] = 1` for a bubble with no outgoing edge in `Hc`, then set `Sep[b] = 2` for a bubble with no incoming edge that has more than one neighbour in `Hb`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Hb`: `Nb×Nb` undirected bubble tree of the PMFG, from [`BubbleHierarchy`](@ref). A non-zero entry joins two bubbles.
  - `Mb`: `Nc×Nb` bubble membership matrix of the 3-cliques. `Mb[n, bi] = 1` means that 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N × Nb` bubble membership matrix of the vertices. `Mv[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique of the MPG.

# Returns

  - `Hc::SparseMatrixCSC{Float64, Int}`: `Nb×Nb` directed adjacency matrix of the DBHT. A non-zero `Hc[i, j]` is a directed edge from bubble `i` to bubble `j`, and its value is the mass of the heavier side.
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
    CliqEdge = Matrix{Int}(undef, length(r), 3)
    for n in eachindex(r)
        CliqEdge[n, 1] = r[n]
        CliqEdge[n, 2] = c[n]
        CliqEdge[n, 3:3] = findall(Mb[:, r[n]] .!= 0 .&& Mb[:, c[n]] .!= 0)
    end

    kb = vec(sum(Hb; dims = 1))
    sMv = size(Mv, 2)
    Hc = SparseArrays.spzeros(sMv, sMv)

    sCE = size(CliqEdge, 1)
    Temp = copy(Hb)
    for n in axes(CliqEdge, 1)
        copyto!(Temp, Hb)
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

Get a non-discrete and a discrete clustering of the vertices from the bubble tree of the Planar Maximally Filtered Graph (PMFG).

Each converging bubble of the directed bubble hierarchical tree (DBHT) is one cluster. The function computes a non-discrete membership matrix, in which a vertex can belong to more than one cluster. It also computes a discrete membership vector, in which each vertex belongs to one cluster.

# Mathematical definition

The function assigns a vertex that more than one converging bubble holds to the bubble of the largest ``\\chi``. ``\\chi`` is the weight that joins the vertex to the bubble, divided by the edge count of the bubble.

```math
\\chi(v,\\, b) = \\frac{\\displaystyle\\sum_{u \\in b} R_{u,\\,v}}{3\\left(\\left|b\\right| - 2\\right)}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``b``: Vertex set of a converging bubble, and ``\\left|b\\right|`` its vertex count.
  - ``\\chi(v,\\, b)``: Association of vertex ``v`` with bubble ``b``.

The denominator is the edge count of a maximal planar graph on ``\\left|b\\right|`` vertices, the same ``3n - 6`` that the PMFG has. It removes the effect of the size of the bubble, so a large bubble does not win only because it is large.

# Algorithm

 1. Direct the bubble tree with [`DirectHb`](@ref), which gives `Hc` and `Sep`.
 2. Take `indx`, the converging bubbles, `Sep .== 1`. When one or none exists, put every vertex in cluster `1`, leave `Adjv` at `0 × 0`, and stop.
 3. For each converging bubble, run [`breadth`](@ref) on the transpose of `Hc`, and mark in column `n` of `Adjv` every vertex of every bubble that the search reaches. The function can mark a vertex in more than one column, so `Adjv` is non-discrete.
 4. Gather `Bubv`, the vertex membership of the converging bubbles alone. Copy into `Mdjv` the rows of `Bubv` for the vertices `cv` that exactly one converging bubble holds.
 5. For each vertex of `uv`, the vertices that more than one converging bubble holds, take the converging bubble of largest ``\\chi`` and mark it in `Mdjv`.
 6. Read the discrete assignment `Tc` off the stored entries of `Mdjv`.
 7. For a vertex that no converging bubble holds, compute `Udjv`, the mean shortest path length from the vertex to the vertices that `Mdjv` assigns to each converging bubble. Set `Udjv` to `typemax` at each cluster of `Adjv` that does not hold the vertex, and assign the vertex to the closest of the other clusters.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Dpm`: `N × N` matrix of the shortest path lengths of the PMFG.
  - `Hb`: `Nb×Nb` undirected bubble tree of the PMFG, from [`BubbleHierarchy`](@ref).
  - `Mb`: `Nc×Nb` bubble membership matrix of the 3-cliques. `Mb[n, bi] = 1` means that 3-clique `n` belongs to bubble `bi`.
  - `Mv`: `N × Nb` bubble membership matrix of the vertices. `Mv[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi`.
  - `CliqList`: `Nc×3` matrix. Each row lists the three vertices of a 3-clique of the MPG.

# Returns

  - `Adjv::SparseMatrixCSC{Int, Int}`: `N × Nk` membership matrix of the non-discrete clustering, where `Nk` is the number of converging bubbles. `Adjv[n, k] = 1` means that vertex `n` belongs to the `k`-th non-discrete cluster, and a vertex can belong to more than one. It is `0 × 0` when step 2 of the algorithm stops.
  - `Tc::Vector{Int}`: `N × 1` cluster membership vector. `Tc[n] = k` means that vertex `n` belongs to the `k`-th discrete cluster. Each vertex belongs to exactly one.

# Related

  - [`DirectHb`](@ref)
  - [`BubbleHierarchy`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)
"""
function BubbleCluster8s(Rpm::MatNum, Dpm::MatNum, Hb::MatNum, Mb::MatNum, Mv::MatNum,
                         CliqList::MatNum)
    Hc, Sep = DirectHb(Rpm, Hb, Mb, Mv, CliqList)   # Assign directions on the bubble tree

    N = size(Rpm, 1)::Int    # Number of vertices in the PMFG
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

        # Compute the distance between a vertex and the converging bubbles. Scale the sparse
        # `Mdjv` first: both products then read only its stored entries, so an unreachable
        # vertex keeps its `Inf` and never meets a zero (#1314).
        Udjv = Dpm * (Mdjv * LinearAlgebra.Diagonal(1 ⊘ vec(sum(!iszero, Mdjv; dims = 1))))
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

Assign each vertex of a cluster to one bubble.

A vertex can be a vertex of more than one bubble. The function assigns such a vertex to one of them, from the edge weights that join the vertex to each candidate bubble. [`HierarchyConstruct4s`](@ref) calls it once for each cluster.

# Mathematical definition

The function assigns a vertex that more than one bubble of the cluster holds to the bubble of the largest ``\\phi``.

```math
\\phi(v,\\, b) = \\frac{\\displaystyle\\sum_{u \\in b} R_{u,\\,v}}{\\displaystyle\\frac{1}{2}\\sum_{u \\in b} \\sum_{u' \\in b} R_{u,\\,u'}}\\,.
```

Where:

  - ``R_{u,\\,v}``: Weight of the PMFG edge between vertices ``u`` and ``v``.
  - ``b``: Vertex set of a bubble.
  - ``\\phi(v,\\, b)``: Fraction of the internal weight of bubble ``b`` that vertex ``v`` draws.

The denominator has the factor one half because the PMFG weights are symmetric, and the double sum counts each edge two times. The ``\\chi`` of [`BubbleCluster8s`](@ref) is different. It divides by the edge count of a maximal planar bubble, and ``\\phi`` divides by the edge weight that the bubble holds.

# Algorithm

 1. Split the vertices that `Mc` marks into `v`, the vertices that exactly one bubble holds, and `vu`, the vertices that more than one bubble holds.
 2. Copy the rows of `Mc` at `v` into `Mvv`. Each of these vertices has one bubble, so the copy assigns it.
 3. For each vertex of `vu`, read its candidate bubbles `bub` off its row of `Mc`, score each with ``\\phi``, and mark the largest in `Mvv`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG.
  - `Mv`: `N × Nb` bubble membership matrix of the vertices. `Mv[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi`.
  - `Mc`: `N × Nb` bubble membership matrix of the vertices of one cluster. `Mc[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi` and that it belongs to the cluster. Every other entry is zero, so a bubble that holds no vertex of the cluster has an empty column.

# Returns

  - `Mvv::Matrix{Int}`: `N × Nb` matrix. `Mvv[n, bi] = 1` when the function assigns vertex `n` to bubble `bi`. Each row has at most one non-zero entry, so the assignment is discrete.

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
        all_bub = map(b -> LinearAlgebra.dot(view(Mv, :, b), Rpm, view(Mv, :, b)) / 2, bub)
        frac = vu_bub ⊘ all_bub
        imx = vec(argmax(frac; dims = 1))
        Mvv[vu[n], bub[imx]] .= 1
    end

    return Mvv
end
