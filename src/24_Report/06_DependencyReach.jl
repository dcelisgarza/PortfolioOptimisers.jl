"""
    reach_hops(dir::AbstractEdgeDirection, At::MatNum, ka::VecInt) -> Matrix{Int}

Count the hops of the shortest path from the node of each asset to each node of a graph, along one part of a reach.

The function walks the graph of the steps of [`reach_weights`](@ref) with one breadth-first traversal from each node of `ka`, as [`separation_matrix`](@ref) does for [`HopCount`](@ref). The weights of the edges have no effect.

# Arguments

  - `dir`: The part of the reach.
  - `At`: The adjacency of the graph, with the sources in its columns.
  - `ka`: The node position of each asset.

# Returns

  - `H::Matrix{Int}`: The hops, `nodes × length(ka)`. The node of the asset has zero hops, and a node that the part does not reach holds `typemax(Int)`.

# Related

  - [`snapshot_hops`](@ref)
  - [`walk_graph`](@ref)
"""
function reach_hops(dir::AbstractEdgeDirection, At::MatNum, ka::VecInt)
    g = walk_graph(reach_weights(dir, At))
    H = Matrix{Int}(undef, size(At, 1), length(ka))
    for (c, s) in enumerate(ka)
        H[:, c] = Graphs.gdistances(g, s)
    end
    return H
end
"""
    snapshot_hops(dirs::Tuple, gs::Tuple, ka::VecInt) -> Matrix{Int}

Count the hops of the shortest path from the node of each asset to each node, over every part of a reach: the smaller count of the parts.

So [`BothEdges`](@ref) takes the shorter of the two directed paths, and it does not mix the two directions at an intermediate node.

# Arguments

  - `dirs`: The parts of the reach, from [`exposure_parts`](@ref).
  - `gs`: The graph of each part.
  - `ka`: The node position of each asset.

# Returns

  - `H::Matrix{Int}`: The hops, `nodes × length(ka)`.

# Related

  - [`reach_hops`](@ref)
  - [`dependency_reach`](@ref)
"""
function snapshot_hops(dirs::Tuple, gs::Tuple, ka::VecInt)
    return mapreduce((d, g) -> reach_hops(d, g.At, ka), (a, b) -> min.(a, b), dirs, gs)
end
"""
    reaches_any(H::AbstractMatrix{<:Integer}, held::AbstractVector{Bool}) -> Bool

Tell whether a held asset reaches a node other than its own.

# Arguments

  - `H`: The hops, from [`snapshot_hops`](@ref).
  - `held`: `true` for each column of a held asset.

# Returns

  - `tf::Bool`: `true` when a held asset reaches another node.

# Related

  - [`dependency_reach`](@ref)
"""
function reaches_any(H::AbstractMatrix{<:Integer}, held::AbstractVector{Bool})
    return any(h -> 0 < h < typemax(h), view(H, :, held))
end
"""
    resolve_separation(sep::HopCount{<:CountOf}, reach::AbstractDependencyReach, H::AbstractMatrix{<:Integer}, held::AbstractVector{Bool}) -> HopCount
    resolve_separation(sep::HopCount{<:FractionOf}, reach::AbstractDependencyReach, H::AbstractMatrix{<:Integer}, held::AbstractVector{Bool}) -> HopCount

Resolve the hop budget of the reach of a [`DependencyExposure`](@ref) over one snapshot.

A count is the budget. A fraction takes its quantile over the hops of the pairs of a held asset and another node that the direction reaches, and rounds it to the nearest `Int`. So `FractionOf(1.0)` is the largest hop count of those pairs, and the budget reaches every node that a path reaches.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `sep`: The hop budget.
  - `reach`: The reach that holds it.
  - `H`: The hops, from [`snapshot_hops`](@ref).
  - `held`: `true` for each column of a held asset.

# Validation

  - A fraction needs one pair of a held asset and another node that the direction reaches. Raises an `ArgumentError`. [`dependency_reach`](@ref) calls it only when a pair exists.

# Returns

  - `sep::HopCount`: The resolved budget.

# Related

  - [`dependency_reach`](@ref)
  - [`HopCount`](@ref)
"""
function resolve_separation(sep::HopCount{<:CountOf}, ::AbstractDependencyReach,
                            ::AbstractMatrix{<:Integer}, ::AbstractVector{Bool})::HopCount
    return sep
end
function resolve_separation(sep::HopCount{<:FractionOf}, ::AbstractDependencyReach,
                            H::AbstractMatrix{<:Integer},
                            held::AbstractVector{Bool})::HopCount
    v = [h for h in view(H, :, held) if 0 < h < typemax(h)]
    @argcheck(!isempty(v),
              ArgumentError("a FractionOf hop budget of a dependency exposure takes a quantile of the hops from each held asset to each node that the direction reaches, and the held assets reach no other node"))
    return HopCount(round(Int, Statistics.quantile(v, sep.n.val)))
end
"""
    reach_scores(reach::AbstractDependencyReach, H::AbstractMatrix{<:Integer}, held::AbstractVector{Bool}) -> Vector

Resolve the budget of a reach over one snapshot, and score each hop count of `1:n`.

# Algorithm

 1. Resolve the budget `n` with [`resolve_separation`](@ref).
 2. Check the decay over `0:n` with [`assert_separation_decay`](@ref).
 3. Score each hop count with [`separation_decay`](@ref).

# Arguments

  - `reach`: The reach, with the fields `sep` and `decay`.
  - `H`: The hops, from [`snapshot_hops`](@ref).
  - `held`: `true` for each column of a held asset.

# Validation

  - The rules of [`resolve_separation`](@ref) and [`assert_separation_decay`](@ref).

# Returns

  - `f::Vector`: The score of each hop count `h`, at `f[h]`.

# Related

  - [`dependency_reach`](@ref)
"""
function reach_scores(reach::AbstractDependencyReach, H::AbstractMatrix{<:Integer},
                      held::AbstractVector{Bool})
    n = resolve_separation(reach.sep, reach, H, held).n.val
    assert_separation_decay(reach.decay, 0:n, n)
    return [separation_decay(reach.decay, h, n) for h in 1:n]
end
"""
    dependency_reach(reach::ShortestHops, dirs::Tuple, gs::Tuple, ka::VecInt, held::AbstractVector{Bool}) -> Tuple{Matrix, Matrix}
    dependency_reach(reach::WalkShares, dirs::Tuple, gs::Tuple, ka::VecInt, held::AbstractVector{Bool}) -> Tuple{Matrix, Matrix}

Return the reach from the node of each asset to each node of one snapshot, the part at one hop and the part at two hops or more.

When no held asset reaches another node, the two parts are zero, and the budget is not resolved.

# Algorithm

The method that Julia selects is the algorithm.

 1. Count the hops with [`snapshot_hops`](@ref).
 2. [`ShortestHops`](@ref): score each node at `h` hops in `1:n` with `f(h)`, from [`reach_scores`](@ref), the part at one hop in `Rd` and the rest in `Ri`.
 3. [`WalkShares`](@ref): build the adjacency `M̃` of each part, [`reach_weights`](@ref) read under `wgt` and normalised under `nrm`. From the unit vectors of the nodes of the assets, multiply by `M̃` from the left `n` times, and add `f(h)` times the walk of `h` hops to `Rd` at `h = 1` and to `Ri` otherwise, over each part. Set the node of each asset to zero.

# Arguments

  - `reach`: The reach.
  - `dirs`: The parts of the direction, from [`exposure_parts`](@ref).
  - `gs`: The graph of each part.
  - `ka`: The node position of each asset that is a node of the snapshot.
  - `held`: `true` for each asset of `ka` that holds a weight.

# Validation

  - The rules of [`reach_scores`](@ref).

# Returns

  - `(Rd, Ri)`: The two parts, `nodes × length(ka)`. The element type comes from the scores, and for [`WalkShares`](@ref) also from the adjacency.

# Related

  - [`AbstractDependencyReach`](@ref)
  - [`snapshot_reach`](@ref)
"""
function dependency_reach(reach::ShortestHops, dirs::Tuple, gs::Tuple, ka::VecInt,
                          held::AbstractVector{Bool})
    H = snapshot_hops(dirs, gs, ka)
    T = typeof(separation_decay(reach.decay, 1, 1))
    Rd, Ri = zeros(T, size(H)), zeros(T, size(H))
    if !reaches_any(H, held)
        return Rd, Ri
    end
    f = reach_scores(reach, H, held)
    Rd, Ri = zeros(eltype(f), size(H)), zeros(eltype(f), size(H))
    for k in eachindex(H)
        h = H[k]
        if h == 1
            Rd[k] = f[1]
        elseif 1 < h <= length(f)
            Ri[k] = f[h]
        end
    end
    return Rd, Ri
end
function dependency_reach(reach::WalkShares, dirs::Tuple, gs::Tuple, ka::VecInt,
                          held::AbstractVector{Bool})
    H = snapshot_hops(dirs, gs, ka)
    Ms = map((d, g) -> normalise_edge_weights(reach.nrm,
                                              read_edge_weights(reach.wgt,
                                                                reach_weights(d, g.At))),
             dirs, gs)
    T = promote_type(map(eltype, Ms)..., typeof(separation_decay(reach.decay, 1, 1)))
    Rd, Ri = zeros(T, size(H)), zeros(T, size(H))
    if !reaches_any(H, held)
        return Rd, Ri
    end
    f = reach_scores(reach, H, held)
    for M in Ms
        V = walk_starts(ka, size(H, 1), T)
        for h in eachindex(f)
            V = M * V
            (h == 1 ? Rd : Ri) .+= f[h] .* V
        end
    end
    for (c, s) in enumerate(ka)
        Rd[s, c] = zero(T)
        Ri[s, c] = zero(T)
    end
    return Rd, Ri
end
"""
    snapshot_reach(de::DependencyExposure, gs::Tuple, nx::VecStr, nodes::VecStr, none::Integer, held::AbstractVector{Bool}) -> Tuple{Matrix{Bool}, Matrix, Matrix}

Return the reach of one snapshot on the reported nodes and on every asset: the own node, the part at one hop and the part at two hops or more.

# Algorithm

 1. Find the node of each asset of `nx` in the snapshot with [`walk_positions`](@ref), and keep the assets that are nodes.
 2. Find the two parts of the reach with [`dependency_reach`](@ref).
 3. Put them on the reported nodes: a reported node that the snapshot does not hold gets a zero row, and an asset that is not a node a zero column.
 4. Mark the own node of each asset in `R0`. When `none` is not zero, mark the row `none` of each asset that is not a node.

# Arguments

  - `de`: The configuration.
  - `gs`: The graph of each part of the direction.
  - `nx`: The asset names.
  - `nodes`: The reported nodes.
  - `none`: The row of `(none)` in `nodes`, or `0`.
  - `held`: `true` for each asset of `nx` that holds a weight at a row of the snapshot.

# Returns

  - `(R0, Rd, Ri)`: Three matrices, `length(nodes) × length(nx)`.

# Related

  - [`dependency_reach`](@ref)
  - [`exposure_result`](@ref)
"""
function snapshot_reach(de::DependencyExposure, gs::Tuple, nx::VecStr, nodes::VecStr,
                        none::Integer, held::AbstractVector{Bool})
    ia = walk_positions(first(gs).nodes, nx)
    j = findall(>(0), ia)
    Rd, Ri = dependency_reach(de.reach, exposure_parts(de.dir), gs, ia[j], held[j])
    pj = walk_positions(first(gs).nodes, nodes)
    K = findall(>(0), pj)
    R0 = zeros(Bool, length(nodes), length(nx))
    Dd = zeros(eltype(Rd), length(nodes), length(nx))
    Di = zeros(eltype(Ri), length(nodes), length(nx))
    Dd[K, j] = view(Rd, pj[K], :)
    Di[K, j] = view(Ri, pj[K], :)
    row = Dict(pj[k] => k for k in K)
    for c in j
        r = get(row, ia[c], 0)
        if r > 0
            R0[r, c] = true
        end
    end
    if none > 0
        R0[none, ia .== 0] .= true
    end
    return R0, Dd, Di
end

public dependency_reach
