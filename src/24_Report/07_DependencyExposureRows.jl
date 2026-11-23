"""
    exposure_row_key(s::ExogenousGraphSeries{<:VecInt}, kint::Integer, kts) -> Integer
    exposure_row_key(s::ExogenousGraphSeries{<:VecDate}, kint::Integer, kts::Dates.AbstractTime) -> Dates.AbstractTime
    exposure_row_key(s::ExogenousGraphSeries{<:VecDate}, kint::Integer, kts::Nothing) -> Union{}

Return the key at which a row of weights reads a series: the position of the row for integer keys, and its timestamp for date keys. A static graph is in force at each row, so it needs no key.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `g`, `s`: The graph or the series.
  - `kint`: The position of the row in the frame of the integer keys.
  - `kts`: The timestamp of the row, or `nothing`.

# Validation

  - A series with date keys needs the timestamp of the row. Raises an [`IsNothingError`](@ref).

# Returns

  - The key of the row.

# Related

  - [`exposure_row_snapshot`](@ref)
"""
function exposure_row_key(::ExogenousGraphSeries{<:VecInt}, kint::Integer, ::Any)
    return kint
end
function exposure_row_key(::ExogenousGraphSeries{<:VecDate}, ::Integer,
                          kts::Dates.AbstractTime)
    return kts
end
function exposure_row_key(::ExogenousGraphSeries{<:VecDate}, ::Integer, ::Nothing)
    return throw(IsNothingError("the Exogenous Graph Series has dates as keys, and the rows of weights carry no timestamps. Give the timestamps `ts` of the returns data, or key the series by row positions"))
end
"""
    exposure_row_snapshot(parts::Tuple, kint::Integer, kts) -> Tuple{Int, Any}

Return the state of the graph of each part of a reach at one row of weights, and the snapshot in force.

The parts of a reach share the keys of their series, so the key of the first part finds the snapshot of each part.

# Algorithm

 1. A static graph of each part: return `1` and the graphs.
 2. A series: find the last key at or before the row. Return `-1` when the row comes before the first key, `0` when the key holds `nothing`, and `1` with the graph of each part otherwise.

# Arguments

  - `parts`: The graph or the series of each part of the reach.
  - `kint`, `kts`: The position and the timestamp of the row, see [`exposure_row_key`](@ref).

# Returns

  - `(state, snap)`: The state of the row, and the tuple of the graphs in force, or `nothing`.

# Related

  - [`exposure_snapshots`](@ref)
"""
function exposure_row_snapshot(parts::Tuple{Vararg{ExogenousGraph}}, ::Integer, ::Any)
    return 1, parts
end
function exposure_row_snapshot(parts::Tuple{Vararg{ExogenousGraphSeries}}, kint::Integer,
                               kts)
    s = first(parts)
    p = searchsortedlast(s.ts, exposure_row_key(s, kint, kts))
    if p == 0
        return -1, nothing
    end
    if isnothing(s.graphs[p])
        return 0, nothing
    end
    return 1, map(x -> x.graphs[p], parts)
end
"""
    exposure_snapshots(blocks::AbstractVector) -> Tuple{Vector, Vector{Int}, Vector}

Find the snapshot that each row of weights reads, and the distinct snapshots.

A block is the rows of one fold, or of one call: `(W, nx, kint, kts, labels, parts)`. Two rows that read equal graphs share one snapshot, so the reach of a snapshot is found one time.

# Arguments

  - `blocks`: The blocks of rows.

# Returns

  - `snaps::Vector`: The distinct snapshots, each a tuple of one graph for each part of the reach.
  - `ridx::Vector{Int}`: The snapshot of each row, `0` at a `nothing` key and `-1` before the first key.
  - `first_keys::Vector`: The first key of the series of each row before the first key.

# Related

  - [`exposure_row_snapshot`](@ref)
  - [`exposure_result`](@ref)
"""
function exposure_snapshots(blocks::AbstractVector)
    st = [[exposure_row_snapshot(b[6], b[3][r], b[4][r]) for r in eachindex(b[3])]
          for b in blocks]
    rows = [x for s in st for x in s]
    snaps = unique(last(x) for x in rows if first(x) == 1)
    pos = Dict(s => k for (k, s) in enumerate(snaps))
    ridx = [first(x) == 1 ? pos[last(x)] : first(x) for x in rows]
    # Only a series has a row before its first key, so its parts have keys.
    first_keys = [let p = first(b[6])
                      isa(p, ExogenousGraphSeries) ? first(p.ts) : nothing
                  end
                  for (b, s) in zip(blocks, st) if any(x -> first(x) == -1, s)]
    return snaps, ridx, first_keys
end
"""
    exposure_names(nxs::AbstractVector) -> Vector{String}

Return the union of the asset names of the blocks of rows, in the order of their first use.

# Related

  - [`exposure_matrix`](@ref)
"""
function exposure_names(nxs::AbstractVector)
    return unique(String[n for nx in nxs for n in nx])
end
"""
    exposure_matrix(Ws::AbstractVector, nxs::AbstractVector, nx::VecStr) -> Matrix

Stack the weights of the blocks of rows on the union of their asset names, with a zero where a block does not hold an asset.

# Arguments

  - `Ws`: The weights of each block, `rows × assets` on its names.
  - `nxs`: The asset names of each block.
  - `nx`: The union of the names, from [`exposure_names`](@ref).

# Validation

  - Each row of weights is finite or `NaN` at every asset. Raises an `ArgumentError` that names the row.

# Returns

  - `W::Matrix`: The weights, `rows × length(nx)`, of the element type of the blocks.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_matrix(Ws::AbstractVector, nxs::AbstractVector, nx::VecStr)
    col = Dict(n => k for (k, n) in enumerate(nx))
    T = mapreduce(eltype, promote_type, Ws)
    W = zeros(T, sum(Wb -> size(Wb, 1), Ws), length(nx))
    r = 0
    for (Wb, nxb) in zip(Ws, nxs)
        for (i, n) in enumerate(nxb)
            view(W, (r + 1):(r + size(Wb, 1)), col[n]) .= view(Wb, :, i)
        end
        r += size(Wb, 1)
    end
    k = findfirst(i -> any(isnan, view(W, i, :)) && !all(isnan, view(W, i, :)), axes(W, 1))
    @argcheck(isnothing(k),
              ArgumentError("each row of weights of a Dependency Exposure is finite or NaN at every asset, and row $k mixes the two"))
    return W
end
"""
    exposure_unplaced_rows!(de::DependencyExposure, ridx::VecInt, labels::AbstractVector, first_keys::AbstractVector, gname::AbstractString) -> ridx

Apply the rule of a row before the first key of a series: refuse it under `strict`, and warn one time and give `NaN` otherwise.

A row at a `nothing` key gives `NaN` with no warning, because the user marked the snapshot as unknown. The rows before the first key leave `ridx` as rows with no graph, `0`.

# Arguments

  - `de`: The configuration.
  - `ridx`: The snapshot of each row, from [`exposure_snapshots`](@ref).
  - `labels`: The label of each row.
  - `first_keys`: The first key of the series of each row before the first key.
  - `gname`: The name of the graph, which the messages quote.

# Validation

  - Under `de.strict`, no row comes before the first key. Raises an `ArgumentError` that names the graph, the rows and the first key.

# Returns

  - `ridx::VecInt`: The snapshots, with `0` at each row before the first key.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_unplaced_rows!(de::DependencyExposure, ridx::VecInt,
                                 labels::AbstractVector, first_keys::AbstractVector,
                                 gname::AbstractString)
    k = findall(==(-1), ridx)
    if isempty(k)
        return ridx
    end
    msg = "the $(length(k)) row(s) $(labels[first(k)]) to $(labels[last(k)]) of the Dependency Exposure come before the first key $(first(first_keys)) of the Exogenous Graph Series \"$gname\", so no graph is in force at them"
    @argcheck(!de.strict, ArgumentError("$msg. Start the rows at or after the first key"))
    @warn("$msg, and their exposure is NaN. Pass `strict = true` to refuse them.",
          maxlog = log_limit(:warn), _id = @log_id())
    ridx[k] .= 0
    return ridx
end
"""
    exposure_nodes(de::DependencyExposure, snaps::AbstractVector, nrows::Integer) -> Vector{String}

Return the nodes that a Dependency Exposure reports: `de.nodes`, or every node of the snapshots in force, in the order of their first use.

# Arguments

  - `de`: The configuration.
  - `snaps`: The distinct snapshots.
  - `nrows`: The number of rows of weights.

# Validation

  - With `de.nodes = nothing`, a snapshot is in force at a row, so a node is known. Raises an `ArgumentError`.
  - With `de.nodes`, each name is a node of a snapshot in force, when one is in force. Raises an `ArgumentError` that names the unknown names.

# Returns

  - `nodes::Vector{String}`: The reported nodes.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_nodes(de::DependencyExposure, snaps::AbstractVector, nrows::Integer)
    known = unique(String[n for s in snaps for n in first(s).nodes])
    if isnothing(de.nodes)
        @argcheck(nrows == 0 || !isempty(snaps),
                  ArgumentError("no graph is in force at any row of the Dependency Exposure, so no node is known. Name the nodes to report with `nodes`"))
        return known
    end
    lost = isempty(snaps) ? String[] : setdiff(de.nodes, known)
    @argcheck(isempty(lost),
              ArgumentError("the nodes $(pinned_repr(first(lost, 5)))$(length(lost) > 5 ? " and $(length(lost) - 5) more" : "") of `nodes` are not nodes of any graph in force"))
    return convert(Vector{String}, de.nodes)
end
"""
    exposure_held(W::MatNum, ridx::VecInt, p::Integer) -> BitVector

Mark the assets that hold a weight that is not zero at a finite row of weights that reads the snapshot `p`.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_held(W::MatNum, ridx::VecInt, p::Integer)
    rows = [r for r in eachindex(ridx) if ridx[r] == p && all(isfinite, view(W, r, :))]
    return vec(any(!iszero, view(W, rows, :); dims = 1))
end
"""
    exposure_unplaced_assets(de::DependencyExposure, W::MatNum, nx::VecStr, ridx::VecInt, labels::AbstractVector, snap::Tuple, p::Integer, gname::AbstractString) -> Bool

Apply the rule of a held asset that is not a node of its snapshot: refuse it under `strict`, and warn one time otherwise. An asset with a zero weight is never refused.

# Arguments

  - `de`: The configuration.
  - `W`: The weights of the rows.
  - `nx`: The asset names.
  - `ridx`: The snapshot of each row.
  - `labels`: The label of each row.
  - `snap`: The snapshot `p`.
  - `p`: The position of the snapshot.
  - `gname`: The name of the graph, which the messages quote.

# Validation

  - Under `de.strict`, each held asset is a node. Raises an `ArgumentError` that names the assets, the graph and the first row.

# Returns

  - `tf::Bool`: `true` when a held asset is not a node, so the Result needs the row `(none)`.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_unplaced_assets(de::DependencyExposure, W::MatNum, nx::VecStr,
                                  ridx::VecInt, labels::AbstractVector, snap::Tuple,
                                  p::Integer, gname::AbstractString)
    lost = exposure_held(W, ridx, p) .& (walk_positions(first(snap).nodes, nx) .== 0)
    if !any(lost)
        return false
    end
    r = findfirst(k -> ridx[k] == p && any(!iszero, view(W, k, lost)), eachindex(ridx))
    names = nx[lost]
    msg = "the held assets $(pinned_repr(first(names, 5)))$(length(names) > 5 ? " and $(length(names) - 5) more" : "") are not nodes of the Exogenous Graph \"$gname\" in force at the row $(labels[r])"
    @argcheck(!de.strict,
              ArgumentError("$msg. Add them to the graph, or pass `strict = false`"))
    @warn("$msg, so their weight goes in the row `(none)`, which reaches no node.",
          maxlog = log_limit(:warn), _id = @log_id())
    return true
end
"""
    exposure_none_label(nodes::VecStr) -> String

Return the name of the row of the held assets that are not nodes: `(none)`, or `(none)_k` with the first suffix `k`, from `1` up, that no node holds.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_none_label(nodes::VecStr)
    k = 0
    name = "(none)"
    while name in nodes
        k += 1
        name = "(none)_$k"
    end
    return name
end
"""
    exposure_parts!(L::AbstractArray, S::AbstractArray, W::MatNum, ridx::VecInt, p::Integer, R::Tuple) -> nothing

Write the long and the short parts of the exposure of each finite row of weights that reads the snapshot `p`.

The long part of `weight`, `direct` and `indirect` is the product of the positive weights and the transpose of `R0`, `Rd` and `Ri`, and the short part the same product of the negative weights.

# Related

  - [`exposure_result`](@ref)
"""
function exposure_parts!(L::AbstractArray, S::AbstractArray, W::MatNum, ridx::VecInt,
                         p::Integer, R::Tuple)
    rows = [r for r in eachindex(ridx) if ridx[r] == p && all(isfinite, view(W, r, :))]
    Wp = max.(view(W, rows, :), zero(eltype(W)))
    Wn = min.(view(W, rows, :), zero(eltype(W)))
    for k in 1:3
        L[rows, :, k] = Wp * transpose(R[k])
        S[rows, :, k] = Wn * transpose(R[k])
    end
    return nothing
end
"""
    exposure_result(de::DependencyExposure, blocks::AbstractVector, gname::AbstractString) -> DependencyExposureResult

Compute the Dependency Exposure of the rows of weights of a set of blocks.

A block is `(W, nx, kint, kts, labels, parts)`: the weights of its rows on its asset names, the position and the timestamp of each row for the graph lookup, the label of each row, and the graph or the series of each part of the reach.

# Algorithm

 1. Stack the weights on the union of the asset names, [`exposure_matrix`](@ref).
 2. Find the snapshot of each row, [`exposure_snapshots`](@ref), and apply the rule of a row before the first key, [`exposure_unplaced_rows!`](@ref).
 3. Find the reported nodes, [`exposure_nodes`](@ref), and apply the rule of a held asset that is not a node, [`exposure_unplaced_assets`](@ref). Add the row `(none)` when an asset is not a node.
 4. Find the reach of each snapshot, [`snapshot_reach`](@ref).
 5. Write the long and the short parts of each row, [`exposure_parts!`](@ref). A row with no graph in force, or a row of `NaN` weights, stays `NaN`.

# Arguments

  - `de`: The configuration.
  - `blocks`: The blocks of rows.
  - `gname`: The name of the graph, which the messages quote.

# Validation

  - The rules of the functions of each step.

# Returns

  - `res::DependencyExposureResult`: The exposure.

# Related

  - [`dependency_exposure`](@ref)
  - [`DependencyExposureResult`](@ref)
"""
function exposure_result(de::DependencyExposure, blocks::AbstractVector,
                         gname::AbstractString)
    nxs = [b[2] for b in blocks]
    nx = exposure_names(nxs)
    W = exposure_matrix([b[1] for b in blocks], nxs, nx)
    labels = [l for b in blocks for l in b[5]]
    snaps, ridx, first_keys = exposure_snapshots(blocks)
    exposure_unplaced_rows!(de, ridx, labels, first_keys, gname)
    nodes = exposure_nodes(de, snaps, length(ridx))
    lost = [exposure_unplaced_assets(de, W, nx, ridx, labels, snaps[p], p, gname)
            for p in eachindex(snaps)]
    none = any(lost) ? length(nodes) + 1 : 0
    if none > 0
        nodes = [nodes; exposure_none_label(nodes)]
    end
    R = [snapshot_reach(de, snaps[p], nx, nodes, none, exposure_held(W, ridx, p))
         for p in eachindex(snaps)]
    T = mapreduce(r -> promote_type(eltype(r[2]), eltype(r[3])), promote_type, R;
                  init = eltype(W))
    L = fill(convert(T, NaN), length(ridx), length(nodes), 3)
    S = fill(convert(T, NaN), length(ridx), length(nodes), 3)
    for p in eachindex(snaps)
        exposure_parts!(L, S, W, ridx, p, R[p])
    end
    # With no graph in force the reach is empty, and an empty vector keeps a matrix type.
    R0 = isempty(R) ? Matrix{Bool}[] : [r[1] for r in R]
    Rd = isempty(R) ? Matrix{T}[] : [r[2] for r in R]
    Ri = isempty(R) ? Matrix{T}[] : [r[3] for r in R]
    return DependencyExposureResult(nodes, nx, labels, W, L, S, R0, Rd, Ri, ridx)
end
