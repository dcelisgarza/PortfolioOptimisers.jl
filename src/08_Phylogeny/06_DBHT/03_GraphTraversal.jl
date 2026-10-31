"""
    distance_wei(L::MatNum)

Compute the length of the shortest weighted path between each pair of vertices of a network.

The function runs the algorithm of Dijkstra on a weighted or unweighted network. Entry `[u, v]` of the result is the length of the shortest path from vertex `u` to vertex `v`. The mean of the shortest path lengths is the characteristic path length of the network.

!!! note

    It ports the function `distance_wei` of the Brain Connectivity Toolbox of [rubinovsporns2010](@citet).

The algorithm is Problem 2 of [dijkstra1959](@cite). It settles the vertices in the order of their distance from the source, and by its Remark 1 it holds on a directed graph. The toolbox runs it from every source, settles every vertex at the least distance in one step, and counts the edges of each path in `B`. [dijkstra1959](@cite) states none of the three.

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

  - `L`: `N × N` connection-length matrix of a directed or undirected network.

      + A pair with no edge holds `0` or `Inf`. The function reads only the non-zero entries, and an `Inf` length never gives a shorter path, so the two values give the same result.
      + The main diagonal holds `0`.

!!! note

    The input matrix must hold lengths, not weights. A caller maps each weight to a length, for example with the reciprocal of a similarity. [`DBHTs`](@ref) puts the entries of its distance matrix on the edges of the TMFG. In a weighted network, a shortest weighted path can have more edges than a shortest binary path.

# Returns

  - `D::Matrix{<:Number}`: `N × N` matrix of the lengths of the shortest weighted paths. `D[u, v]` is the length of the shortest path from vertex `u` to vertex `v`.
  - `B::Matrix{Int}`: `N × N` matrix of the edge count of each shortest weighted path.

# Related

  - [`PMFG_T2s`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`DBHT`](@ref)

# References

  - $(ref_dict[:rubinovsporns2010])
  - $(ref_dict[:dijkstra1959]) Problem 2 and Remark 1.
"""
function distance_wei(L::MatNum)
    N = size(L, 1)
    D = fill(typemax(eltype(L)), N, N)
    D[LinearAlgebra.diagind(D)] .= 0  # Distance matrix
    B = zeros(Int, N, N)     # Number of edges matrix
    S = fill(true, N)   # Distance permanence (true is temporary)
    L1 = copy(L)

    for u in axes(L, 1)
        fill!(S, true)
        copyto!(L1, L)
        V = [u]
        while true
            S[V] .= false   # Distance u -> V is now permanent
            L1[:, V] .= 0   # No inside edges as already shortest
            SparseArrays.dropzeros!(L1)
            for v in V
                T = SparseArrays.findnz(L1[v, :])[1] # neighbours of shortest nodes
                # The path through `v` wins only when it is strictly shorter, so a tie keeps
                # the old path and its edge count, and the path that won is one edge longer
                # than the path to `v`. Issue #470.
                dT = D[u, v] .+ L1[v, T]
                won = dT .< view(D, u, T)
                D[u, T[won]] = dT[won]
                B[u, T[won]] .= B[u, v] + 1
            end

            dus = D[u, S]
            minD = !isempty(dus) ? minimum(dus) : eltype(D)[]

            # isempty: all nodes reached
            # isinf: some nodes cannot be reached
            if isempty(minD) || isinf(minD)
                break
            end

            V = findall(==(minD), view(D, u, :))
        end
    end

    return D, B
end
"""
    breadth(CIJ::MatNum, source::Integer)

Run a breadth-first search from one source vertex.

The function searches a binary connection matrix of a directed or undirected graph, from the vertex `source`. It computes the shortest path distance from the source to each other vertex. It records in `branch` the predecessor of each vertex in the search tree. The tree holds one shortest path for each reachable vertex, and not every shortest path, so `branch` gives one path of minimum length to each vertex.

!!! note

    It ports the function `breadth` of the Brain Connectivity Toolbox of [rubinovsporns2010](@citet).

Breadth-first search is older than the toolbox. The library has no copy of the work that first states it, so this function follows the toolbox alone.

# Algorithm

 1. Colour every vertex white, set every entry of `distance` to `Inf`, and set every entry of `branch` to zero.
 2. Colour `source` grey, set its distance to zero and its branch to `-1`, and put it in the queue `Q`.
 3. Take the head `u` of `Q`, and read its out-neighbours `ns` from the stored entries of row `u`.
 4. For each neighbour `v` whose distance is still zero, set it to `distance[u] + 1`. The distance of the source starts at zero, so this step also changes it when the search expands a neighbour of the source.
 5. For each white neighbour `v`, colour it grey, set `distance[v]` to `distance[u] + 1`, set `branch[v]` to `u`, and append `v` to `Q`.
 6. Drop `u` from `Q` and colour it black. Repeat from step 3 until `Q` is empty.

!!! warning

    `distance[source]` does not stay `0`. Step 4 changes it the first time that the search expands a neighbour of the source. On an undirected graph with no self-loop, the entry ends at `2`. When the graph has a self-loop on the source, it ends at `1`. The MATLAB original has the same behaviour. [`FindDisjoint`](@ref), [`DirectHb`](@ref) and [`BubbleCluster8s`](@ref) call this function, and each writes `0` into the entry after the call.

# Arguments

  - `CIJ`: `N × N` binary connection matrix of the graph. A non-zero entry is an edge. Row `u` holds the out-neighbours of vertex `u`.
  - `source`: Index of the vertex where the search starts.

# Returns

  - `distance::VecNum`: `N × 1` vector of the shortest path distance from the source to each vertex. Its element type is `float_if_integer(eltype(CIJ))`, so an integer or `Bool` matrix gives a floating-point vector. `Inf` marks a vertex that no path reaches. The entry of the source is not `0`, as the warning states.
  - `branch::Vector{Int}`: `N × 1` vector of the predecessor of each vertex in the search tree. It is `-1` at the source and `0` at an unreachable vertex.

# Related

  - [`FindDisjoint`](@ref)
  - [`CliqHierarchyTree2s`](@ref)

# References

  - $(ref_dict[:rubinovsporns2010]) Page 1065, for the toolbox that holds `breadth`. $(ref_dict[:no_original_source])
"""
function breadth(CIJ::MatNum, source::Integer)
    N = size(CIJ, 1)
    # Colours
    white = 0
    gray = 1
    black = 2
    # Initialise colours
    color = zeros(Int, N)
    # Initialise distances. A distance must hold `Inf` for a vertex no path reaches, so a
    # `Bool` or an integer matrix takes the floating-point type that represents it, and a
    # `Float32` one stays `Float32`.
    Td = float_if_integer(eltype(CIJ))
    distance = fill(convert(Td, Inf), N)
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
                push!(Q, v)
            end
        end
        popfirst!(Q)
        color[u] = black
    end

    return distance, branch
end
