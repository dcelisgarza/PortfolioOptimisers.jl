"""
    exposure_weights(src::TargetWeights, pred::PredictionResult, j::Integer = 1) -> Tuple{Matrix, Vector{Int}}
    exposure_weights(src::DriftedWeights, pred::PredictionResult, j::Integer = 1) -> Tuple{Matrix, Vector{Int}}

Return the rows of weights of member `j` of one fold that a [`DependencyExposure`](@ref) reads, and the row of the fold at which each holds.

This is the second job of the family [`AbstractPreviousWeightsSource`](@ref): the source that threads weights into the next fold also names the weights of a fold that the exposure describes. The weights are on the names `pred.rd.nx` of the fold, viewed at its Investable Mask as [`fold_target`](@ref) views them.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`TargetWeights`](@ref): one row, the target of the fold from [`fold_target`](@ref), at the first row of the fold. This is the portfolio of the fold, as the Weights section of [`portfolio_report`](@ref) reads it.
 2. [`DriftedWeights`](@ref): one row for each observation of the fold, the weight path of its Held Weights record from [`weight_path`](@ref), or its target at each row on a fold with no record. A failed fold that holds the weights it received under a drift gives those weights.

# Arguments

  - `src`: The source.
  - `pred`: One fold.
  - `j`: The index of the member.

# Returns

  - `W::Matrix`: The rows of weights, `rows × assets`. A failed fold gives a row of `NaN`.
  - `t::Vector{Int}`: The row of the fold of each row of `W`.

# Related

  - [`AbstractPreviousWeightsSource`](@ref)
  - [`DependencyExposure`](@ref)
  - [`fold_target`](@ref)
  - [`weight_path`](@ref)
"""
function exposure_weights(::TargetWeights, pred::PredictionResult, j::Integer = 1)
    return permutedims(collect(fold_target(pred, j))), [1]
end
function exposure_weights(::DriftedWeights, pred::PredictionResult, j::Integer = 1)
    U = fold_weight_path(pred, pred.hw, j)
    return U, collect(axes(U, 1))
end
"""
    fold_weight_path(pred::PredictionResult, hw::Nothing, j::Integer) -> Matrix
    fold_weight_path(pred::PredictionResult, hw::HeldWeightsResult, j::Integer) -> Matrix

Return the weights of member `j` of a fold at the start of each of its observations, on the names of the fold.

# Algorithm

The method that Julia selects is the algorithm.

 1. No record: repeat the target of [`fold_target`](@ref) over the rows of the fold.
 2. A record: read its path with [`weight_path`](@ref), take member `j`, and view it at the Investable Mask of the fold.

# Related

  - [`exposure_weights`](@ref)
"""
function fold_weight_path(pred::PredictionResult, ::Nothing, j::Integer)
    return repeat(permutedims(collect(fold_target(pred, j))),
                  observation_count(pred.rd.X, pred.rd.F))
end
function fold_weight_path(pred::PredictionResult, hw::HeldWeightsResult, j::Integer)
    U = path_member(weight_path(hw, pred.res.w), j)
    return collect(investable_weights_view(result_investable_mask(pred.res), U))
end
"""
    path_member(U::MatNum, j::Integer) -> MatNum
    path_member(U::VecMatNum, j::Integer) -> MatNum

Return the weight path of member `j`: the one path of a fold with one member, or entry `j` of a population of members.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`fold_weight_path`](@ref)
  - [`member_vector`](@ref)
"""
function path_member(U::MatNum, ::Integer)
    return U
end
function path_member(U::VecMatNum, j::Integer)
    return U[j]
end
"""
    exposure_graphs(de::DependencyExposure, rd::Union{ReturnsResult, PredictionReturnsResult}) -> Tuple{Tuple, String}
    exposure_graphs(de::DependencyExposure, name::Option{<:AbstractString}, ng, G) -> Tuple{Tuple, String}
    exposure_graphs(de::DependencyExposure, gu::GraphUnion, ng, G) -> Tuple{Tuple, String}
    exposure_graphs(de::DependencyExposure, G::ExG_ExGS) -> Tuple{Tuple, String}

Return the graph or the series of each part of the reach of `de`, and the name of the graph, after a check of each snapshot against [`graph_requirements`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. Returns data: read the entry `de.ng` of its graphs `ng` and `G`.
 2. A name, or `nothing` for the only graph: check each snapshot of the graph with [`assert_consumer_entry`](@ref), and give the graph to each part.
 3. A [`GraphUnion`](@ref): check each snapshot of each part, and build the union along each part of the direction with [`graph_union`](@ref).
 4. An explicit graph: refuse a `de.ng` that is not `nothing`, check each snapshot, and give the graph to each part.

# Arguments

  - `de`: The configuration.
  - $(arg_dict[:rd])
  - `name`, `gu`: The entry `de.ng`.
  - `ng`, `G`: The names and the graphs of the returns data.
  - `G`: The explicit graph or series.

# Validation

  - The rules of [`graph_position`](@ref), [`assert_consumer_entry`](@ref) and [`graph_union`](@ref).
  - With an explicit graph, `de.ng` is `nothing`, because the graph replaces the lookup. Raises an `ArgumentError`.

# Returns

  - `parts::Tuple`: The graph or the series of each part, from [`exposure_parts`](@ref).
  - `gname::String`: The name of the graph.

# Related

  - [`dependency_exposure`](@ref)
  - [`exposure_result`](@ref)
"""
function exposure_graphs(de::DependencyExposure,
                         rd::Union{ReturnsResult, PredictionReturnsResult})
    return exposure_graphs(de, de.ng, rd.ng, rd.G)
end
function exposure_graphs(de::DependencyExposure, name::Option{<:AbstractString}, ng, G)
    k = graph_position(ng, name)
    assert_consumer_entry(de, name, ng, G, "")
    return ntuple(_ -> G[k], length(exposure_parts(de.dir))), String(ng[k])
end
function exposure_graphs(de::DependencyExposure, gu::GraphUnion, ng, G)
    assert_consumer_entry(de, gu, ng, G, "")
    gs = [G[graph_position(ng, name)] for name in gu.ng]
    parts = map(d -> graph_union(gs...; w = gu.w, nrm = gu.nrm, dir = d,
                                 unknown = gu.unknown), exposure_parts(de.dir))
    return parts, String(graph_entry_label(gu))
end
function exposure_graphs(de::DependencyExposure, G::ExG_ExGS)
    @argcheck(isnothing(de.ng),
              ArgumentError("an explicit graph replaces the lookup of `de.ng` in the returns data, so `de.ng` must be `nothing`, got $(repr(de.ng))"))
    for (key, g) in graph_snapshot_list(G)
        assert_graph_requirements(de, g, "G", key, "")
    end
    return ntuple(_ -> G, length(exposure_parts(de.dir))), "G"
end
"""
    exposure_fold_block(de::DependencyExposure, pred::PredictionResult, parts::Tuple, koff::Integer, loff::Integer, j::Integer) -> Tuple

Return the block of rows of one fold for [`exposure_result`](@ref): its weights, its names, the keys of its rows, their labels and the graphs of the parts.

# Arguments

  - `de`: The configuration, whose `src` names the weights through [`exposure_weights`](@ref).
  - `pred`: The fold.
  - `parts`: The graph or the series of each part of the reach.
  - `koff`: The rows before the fold in the frame of the integer keys: `0` for the graphs of the fold, and the rows of the earlier folds for an explicit graph.
  - `loff`: The rows of the earlier folds, for the label of a row with no timestamp.
  - `j`: The index of the member.

# Returns

  - `(W, nx, kint, kts, labels, parts)`: The block. The label of a row is its timestamp, or its position in the prediction.

# Related

  - [`exposure_fold_blocks`](@ref)
"""
function exposure_fold_block(de::DependencyExposure, pred::PredictionResult, parts::Tuple,
                             koff::Integer, loff::Integer, j::Integer)
    W, t = exposure_weights(de.src, pred, j)
    nx = something(pred.rd.nx, string.(axes(W, 2)))
    ts = pred.rd.ts
    kts = isnothing(ts) ? fill(nothing, length(t)) : ts[t]
    labels = isnothing(ts) ? loff .+ t : kts
    return W, nx, koff .+ t, kts, labels, parts
end
"""
    exposure_fold_blocks(de::DependencyExposure, folds::AbstractVector, G::Nothing, j::Integer) -> Tuple{Vector, String}
    exposure_fold_blocks(de::DependencyExposure, folds::AbstractVector, G::ExG_ExGS, j::Integer) -> Tuple{Vector, String}

Return the block of rows of each fold of a prediction, and the name of the graph.

# Algorithm

The method that Julia selects is the algorithm.

 1. No explicit graph: each fold reads its own graphs, the view of its rows, with integer keys that count the rows of the fold.
 2. An explicit graph: each fold reads it, with integer keys that count the rows of the prediction in order, the positional form of the oracle.

# Related

  - [`exposure_fold_block`](@ref)
  - [`dependency_exposure`](@ref)
"""
function exposure_fold_blocks(de::DependencyExposure, folds::AbstractVector, ::Nothing,
                              j::Integer)
    offs = fold_row_offsets(folds)
    srcs = [exposure_graphs(de, f.rd) for f in folds]
    blocks = [exposure_fold_block(de, folds[k], first(srcs[k]), 0, offs[k], j)
              for k in eachindex(folds)]
    return blocks, last(first(srcs))
end
function exposure_fold_blocks(de::DependencyExposure, folds::AbstractVector, G::ExG_ExGS,
                              j::Integer)
    offs = fold_row_offsets(folds)
    parts, gname = exposure_graphs(de, G)
    return [exposure_fold_block(de, folds[k], parts, offs[k], offs[k], j)
            for k in eachindex(folds)], gname
end
"""
    fold_row_offsets(folds::AbstractVector) -> Vector{Int}

Return the number of rows of the folds before each fold of a prediction.

# Related

  - [`exposure_fold_blocks`](@ref)
"""
function fold_row_offsets(folds::AbstractVector)
    n = [observation_count(f.rd.X, f.rd.F) for f in folds]
    return cumsum([0; n[1:(end - 1)]])
end
"""
    prediction_folds(pred::PredictionResult) -> Vector
    prediction_folds(pred::MultiPeriodPredictionResult) -> AbstractVector

Return the folds of a prediction: the one fold, or the folds of a path.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`dependency_exposure`](@ref)
"""
function prediction_folds(pred::PredictionResult)
    return [pred]
end
function prediction_folds(pred::MultiPeriodPredictionResult)
    return pred.pred
end
"""
    exposure_weight_rows(w::VecNum, T::Integer, N::Integer) -> Matrix
    exposure_weight_rows(W::MatNum, T::Integer, N::Integer) -> MatNum

Return the weights of each row of the returns data: a vector at every row, as the oracle broadcasts a 1D array, or a matrix as it is.

`dependency_exposure` is an entry point of integer data, so an integer weight converts once to its floating-point type through [`float_if_integer`](@ref), because a row with no graph in force is `NaN`.

# Algorithm

The method that Julia selects is the algorithm.

# Validation

  - A vector has one weight for each asset, and a matrix the size `(T, N)`. Raises a `DimensionMismatch`.

# Related

  - [`dependency_exposure`](@ref)
"""
function exposure_weight_rows(w::VecNum, T::Integer, N::Integer)
    @argcheck(length(w) == N,
              DimensionMismatch("a weight vector holds one weight for each asset of the returns data, so its length must be $N, got $(length(w))"))
    return repeat(permutedims(float_if_integer(w)), T)
end
function exposure_weight_rows(W::MatNum, T::Integer, N::Integer)
    @argcheck(size(W) == (T, N),
              DimensionMismatch("a weight matrix holds one row for each row of the returns data and one column for each asset, so its size must be ($T, $N), got $(size(W))"))
    return float_if_integer(W)
end
"""
    population_exposure(rs::AbstractVector{<:DependencyExposureResult}, keys::AbstractVector) -> DependencyExposureResult

Reduce the Dependency Exposure of each path of a population to one row: the mean of the rows of the path.

The nodes and the asset names are the unions over the paths, in order, with a zero where a path does not report a node or hold an asset. A row is the mean of rows that read different snapshots, so it has no reach, and `ridx` is `0`.

# Arguments

  - `rs`: The exposure of each path.
  - `keys`: The key of each path, [`population_path_key`](@ref).

# Returns

  - `res::DependencyExposureResult`: One row for each path.

# Related

  - [`dependency_exposure`](@ref)
"""
function population_exposure(rs::AbstractVector{<:DependencyExposureResult},
                             keys::AbstractVector)
    nodes = unique(String[n for r in rs for n in r.nodes])
    nx = exposure_names([r.nx for r in rs])
    W = exposure_matrix([mean(r.W; dims = 1) for r in rs], [r.nx for r in rs], nx)
    T = mapreduce(r -> eltype(r.L), promote_type, rs)
    L, S = (zeros(T, length(rs), length(nodes), 3) for _ in 1:2)
    col = Dict(n => c for (c, n) in enumerate(nodes))
    for (k, r) in enumerate(rs), (i, n) in enumerate(r.nodes), p in 1:3
        L[k, col[n], p] = mean(view(r.L, :, i, p))
        S[k, col[n], p] = mean(view(r.S, :, i, p))
    end
    R = Matrix{T}[]
    return DependencyExposureResult(nodes, nx, collect(keys), W, L, S, Matrix{Bool}[], R,
                                    copy(R), zeros(Int, length(rs)))
end
"""
    dependency_exposure(pred::Union{PredictionResult, MultiPeriodPredictionResult},
                        de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(pred::Union{PredictionResult, MultiPeriodPredictionResult},
                        G::ExG_ExGS, de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(ppred::PopulationPredictionResult,
                        de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(ppred::PopulationPredictionResult, G::ExG_ExGS,
                        de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        G::ExG_ExGS, de::DependencyExposure = DependencyExposure(), j::Integer = 1)
    dependency_exposure(W::VecNum_MatNum, rd::ReturnsResult,
                        de::DependencyExposure = DependencyExposure())
    dependency_exposure(w::VecNum, nx::VecStr, g::ExogenousGraph,
                        de::DependencyExposure = DependencyExposure())

Compute the Dependency Exposure of the weights of a Result to each node of an Exogenous Graph, under the configuration `de`.

Each method gives a [`DependencyExposureResult`](@ref) with one row for each row of weights, and the graph in force at each row. [`DependencyExposure`](@ref) states the reach, the direction, the reported nodes and the rules of a missing graph and of a held asset that is not a node.

# The methods

  - **A prediction.** A fold, a path or a population reads the graphs of each fold, `pred.rd.G`, the view of the rows of the fold. `de.src` names the weights, through [`exposure_weights`](@ref): one row for each fold under [`TargetWeights`](@ref), with the graph in force at the first row of the fold, or one row for each observation under [`DriftedWeights`](@ref). A population gives one row for each path, the mean of the rows of the path. `j` is the member, as in [`prediction_weights`](@ref).
  - **An explicit graph.** `G` replaces the lookup in `pred.rd.G`, so a prediction made with no graph has an exposure too, and `de.ng` must be `nothing`. A date key matches the timestamps of each fold. An integer key counts the rows of the prediction in order, the positional form of the oracle. A population counts the rows of each path from one.
  - **An optimisation result.** The portfolio of `res` over the returns data `rd`, with no cross-validation, is one fold over `rd`, from `predict(res, rd)`, with no drift.
  - **Bare weights.** `W` is a vector, which applies at each row of `rd`, as the oracle broadcasts a 1D array, or a `rows × assets` matrix. The names come from `rd.nx`, and the graph in force at each row from `rd.G`. The last method is one vector `w` on the asset names `nx` and one graph `g`, with no returns data. `de.src` does not apply, because the weights state the rows.

# The rows with no graph

A row at a `nothing` key of a series gives `NaN`, with no warning, because the user marked the snapshot as unknown. A row before the first key gives one warning that names the graph, the rows and the first key, and `NaN`, or refuses before any compute under `strict = true`. The view of a fold that lies before the first key holds one `nothing` key, see [`graph_view`](@ref), so its rows give `NaN` with no warning. A row of `NaN` weights, a failed fold, gives `NaN`. A node that a snapshot does not hold has zero exposure at its rows.

# Arguments

  - `pred`, `ppred`: The prediction.
  - `res`: The optimisation result.
  - $(arg_dict[:rd])
  - `G`: The explicit graph, an [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref).
  - `de`: The configuration.
  - `j`: The index of the member.
  - `W`, `w`: The weights, on the asset names of `rd.nx` or `nx`.
  - `nx`: The asset names of `w`.
  - `g`: The graph.

# Validation

  - Each snapshot of the graph has non-negative edge weights, [`NonNegativeWeights`](@ref). Raises an `ArgumentError` that names the edge.
  - The rules of [`graph_position`](@ref) for `de.ng`, and `de.ng` is `nothing` with an explicit graph. Raises an `ArgumentError` or an [`IsNothingError`](@ref).
  - A row of weights is finite or `NaN` at every asset. Raises an `ArgumentError`.
  - The weights have one entry for each asset, and a matrix one row for each row of `rd`. Raises a `DimensionMismatch`.
  - The rules of a row before the first key, of a held asset that is not a node and of `nodes`, as [`DependencyExposure`](@ref) states them. Each raises an `ArgumentError`.
  - A population holds one path or more. Raises an [`IsEmptyError`](@ref).
  - Returns data of the bare method holds the names `nx` and rows. Raises an [`IsNothingError`](@ref).

# Returns

  - `res::DependencyExposureResult`: The exposure.

# Examples

```jldoctest
julia> g = ExogenousGraph(; edges = [(\"Token A\", \"Protocol\", 1), (\"Token B\", \"Token A\", 3)],
                          nodes = [\"Token A\", \"Token B\", \"Protocol\"]);

julia> res = dependency_exposure([0.6, 0.4], [\"Token A\", \"Token B\"], g);

julia> res.nodes
3-element Vector{String}:
 \"Token A\"
 \"Token B\"
 \"Protocol\"

julia> dropdims(sum(res.L; dims = 3); dims = 3)
1×3 Matrix{Float64}:
 1.0  0.4  1.0
```

# Related

  - [`DependencyExposure`](@ref)
  - [`DependencyExposureResult`](@ref)
  - [`exposure_weights`](@ref)
  - [`exposure_contributions`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`ExogenousGraphSeries`](@ref)
  - [`portfolio_report`](@ref)
"""
function dependency_exposure(pred::PredRes_MultiPredRes,
                             de::DependencyExposure = DependencyExposure(),
                             j::Integer = 1)::DependencyExposureResult
    return exposure_result(de,
                           exposure_fold_blocks(de, prediction_folds(pred), nothing, j)...)
end
function dependency_exposure(pred::PredRes_MultiPredRes, G::ExG_ExGS,
                             de::DependencyExposure = DependencyExposure(),
                             j::Integer = 1)::DependencyExposureResult
    return exposure_result(de, exposure_fold_blocks(de, prediction_folds(pred), G, j)...)
end
function dependency_exposure(ppred::PopulationPredictionResult,
                             args...)::DependencyExposureResult
    paths = ppred.pred
    @argcheck(!isempty(paths),
              IsEmptyError("the population holds no path, so it holds no weights"))
    return population_exposure([dependency_exposure(p, args...) for p in paths],
                               [population_path_key(paths[k], k) for k in eachindex(paths)])
end
function dependency_exposure(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                             args...)::DependencyExposureResult
    return dependency_exposure(StatsAPI.predict(res, rd), args...)
end
function dependency_exposure(W::VecNum_MatNum, rd::ReturnsResult,
                             de::DependencyExposure = DependencyExposure())::DependencyExposureResult
    @argcheck(!isnothing(rd.nx),
              IsNothingError("the Dependency Exposure finds each asset among the nodes of the graph by its name, and the returns data names no asset. Give the ReturnsResult the names `nx`"))
    T = observation_count(rd.X, rd.F)
    @argcheck(!isnothing(T),
              IsNothingError("the weights of a Dependency Exposure hold at the rows of the returns data, and the returns data holds no rows. Give the returns X or F"))
    parts, gname = exposure_graphs(de, rd)
    t = collect(1:T)
    kts = isnothing(rd.ts) ? fill(nothing, T) : rd.ts
    block = (exposure_weight_rows(W, T, length(rd.nx)), rd.nx, t, kts, something(rd.ts, t),
             parts)
    return exposure_result(de, [block], gname)
end
function dependency_exposure(w::VecNum, nx::VecStr, g::ExogenousGraph,
                             de::DependencyExposure = DependencyExposure())::DependencyExposureResult
    parts, gname = exposure_graphs(de, g)
    block = (exposure_weight_rows(w, 1, length(nx)), nx, [1], [nothing], [1], parts)
    return exposure_result(de, [block], gname)
end

export dependency_exposure, exposure_weights
