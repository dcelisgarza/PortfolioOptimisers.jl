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
                # `findmin(...; dims = 1)` returns one `CartesianIndex` per column, so
                # component 1 is the row that won and component 2 is the column, which
                # is the position in `T`. The winner is what decides the edge count, so
                # read component 1. Row 2 is the path through `v`, one edge longer than
                # the path to `v`. Reading component 2 and comparing it with `3` left
                # `B` unrelated to any path. Issue #470.
                wi = vec(getindex.(wi, 1))
                D[u, T] = vec(d)   # Smallest of old/new path lengths
                ind = T[wi .== 2]   # Indices of lengthened paths
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
    breadth(CIJ::MatNum, source::Integer)

Breadth-first search.

This function performs a breadth-first search (BFS) on a binary (directed or undirected) connection matrix, starting from a specified source vertex. It computes the shortest path distances from the source to all other vertices and records the predecessor (branch) for each node in the BFS tree. The tree holds one shortest path per reachable vertex, and not every shortest path, so `branch` reconstructs one route of minimum length rather than all of them.

!!! note

    Original implementation by Olaf Sporns, Indiana University, 2002/2007/2008.

# Algorithm

 1. Colour every vertex white, set every entry of `distance` to `Inf`, and set every entry of `branch` to zero.
 2. Colour `source` grey, set its distance to zero and its branch to `-1`, and put it in the queue `Q`.
 3. Take the head `u` of `Q`, and read its out-neighbours `ns` from the stored entries of row `u`.
 4. For each neighbour `v` whose distance is still zero, set it to `distance[u] + 1`. The source starts at zero, so this arm also fires on the source itself as soon as one of its own neighbours is expanded.
 5. For each white neighbour `v`, colour it grey, set `distance[v]` to `distance[u] + 1`, set `branch[v]` to `u`, and append `v` to `Q`.
 6. Drop `u` from `Q` and colour it black. Repeat from step 3 until `Q` is empty.

!!! warning

    `distance[source]` does not stay `0`. Step 4 fires on the source itself the first time one of the source's own neighbours is expanded, so on an undirected graph without a self-loop the entry ends at `2`, and at `1` where the graph carries a self-loop on the source. This is the behaviour of the MATLAB original. All three callers reset the entry after the call: [`FindDisjoint`](@ref), [`DirectHb`](@ref) and [`BubbleCluster8s`](@ref) each write a `0` into it. Measured in issue #470 on a five-vertex path with a leaf, where the source reads `2.0`.

# Arguments

  - `CIJ`: `N × N` binary (0/1) connection matrix representing the graph. Row `u` holds the out-neighbours of vertex `u`.
  - `source`: Index of the source vertex from which to start the search.

# Returns

  - `distance::VecNum`: `N × 1` vector of shortest path distances from the source to each vertex. `Inf` marks a vertex no path reaches. The source's own entry is **not** `0`; see the warning above.
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
