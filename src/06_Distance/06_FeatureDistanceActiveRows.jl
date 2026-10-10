"""
    active_window(amsk, Z::Arr3Num, dims::Integer)

Check the active mask of a window against the window, and give it only when a cell of it is inactive.

The active mask marks, for each observation of the window and each asset, whether the asset is in the universe. A window with no inactive cell reads as `nothing`, so the collapse takes the path that reads every row, and its result is the one it gives with no mask.

# Algorithm

 1. `amsk` is `nothing`: return `nothing`.
 2. Otherwise check that `amsk` is `observations × assets` for the layout that `dims` declares, and return `nothing` when every entry is `true`, or `amsk` otherwise.

# Arguments

  - `amsk`: The active mask of the window, `observations × assets`, or `nothing` when every cell is active.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Validation

  - `size(amsk) == (size(Z, 1), assets)`. Raises a `DimensionMismatch`.

# Returns

  - `A::Option{<:AbstractMatrix{Bool}}`: The active mask, or `nothing` when every cell is active.

# Related

  - [`distance`](@ref)
  - [`active_feature_distance`](@ref)
"""
function active_window(::Nothing, ::Arr3Num, ::Integer)
    return nothing
end
function active_window(amsk::AbstractMatrix{Bool}, Z::Arr3Num, dims::Integer)
    ax = (size(Z, 1), size(Z, dims == 1 ? 2 : 3))
    @argcheck(size(amsk) == ax,
              DimensionMismatch("the active mask of a window of time-varying features is observations × assets, so size(amsk) == $(ax) must hold. Got\nsize(amsk) => $(size(amsk))"))
    return all(amsk) ? nothing : amsk
end
"""
    feature_readable(alg, A::AbstractMatrix{Bool}) -> BitVector

Mark the assets that a collapse can read in a window, from the active mask of the window.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`LastObservation`](@ref) forwards to its rule.
 2. [`LastRow`](@ref): an asset is readable when it is active at the last row.
 3. Every other rule and collapse: an asset is readable when it is active at one row of the window at least.

# Arguments

  - `alg`: The collapse algorithm, or the rule of a [`LastObservation`](@ref).
  - `A`: The active mask of the window, `observations × assets`.

# Returns

  - `r::BitVector`: One entry per asset, `true` where the collapse can read the asset.

# Related

  - [`assert_feature_readable`](@ref)
  - [`feature_readable_mask`](@ref)
"""
function feature_readable(alg::LastObservation, A::AbstractMatrix{Bool})
    return feature_readable(alg.alg, A)
end
function feature_readable(::LastRow, A::AbstractMatrix{Bool})
    return BitVector(view(A, size(A, 1), :))
end
function feature_readable(::Union{<:LastActiveRow, <:AbstractFeatureCollapseAlgorithm},
                          A::AbstractMatrix{Bool})
    return BitVector(vec(any(A; dims = 1)))
end
"""
    feature_asset_labels(nx, i::AbstractVector{<:Integer}) -> String
    feature_asset_labels(nx, E::AbstractVector{<:Tuple}) -> String

Name the assets at the positions `i`, or the pairs of positions `E`, for a refusal: by their names when `nx` holds them, and by their positions otherwise.

# Related

  - [`assert_feature_readable`](@ref)
  - [`empty_pair_distance!`](@ref)
"""
function feature_asset_labels(::Nothing, i::AbstractVector)
    return "the assets at the positions $(i)"
end
function feature_asset_labels(nx::AbstractVector, i::AbstractVector)
    return "the assets $(nx[i])"
end
function feature_asset_labels(::Nothing, E::AbstractVector{<:Tuple})
    return "the assets at the positions $(join(E, ", "))"
end
function feature_asset_labels(nx::AbstractVector, E::AbstractVector{<:Tuple})
    return "the assets $(join(((nx[i], nx[j]) for (i, j) in E), ", "))"
end
"""
    assert_feature_readable(alg, A, nx)

Refuse a window in which the collapse has no value to read for an asset, and name each such asset.

Inside a fit, [`feature_readable_mask`](@ref) drops such an asset at the entry of the optimiser, so the collapse never meets it. A direct call of [`distance`](@ref), [`cor_and_dist`](@ref) or [`clusterise`](@ref) has no entry of a fit, so it refuses here.

# Arguments

  - `alg`: The collapse algorithm.
  - `A`: The active mask of the window, `observations × assets`, or `nothing` when every cell is active.
  - `nx`: The asset names, or `nothing`.

# Validation

  - Every asset is readable by [`feature_readable`](@ref). Raises an `ArgumentError` that names each other asset and the remedy.

# Returns

  - `nothing`.

# Related

  - [`feature_readable`](@ref)
  - [`feature_readable_mask`](@ref)
"""
function assert_feature_readable(::Any, ::Nothing, ::Any)::Nothing
    return nothing
end
function assert_feature_readable(alg::AbstractFeatureCollapseAlgorithm,
                                 A::AbstractMatrix{Bool}, nx)::Nothing
    i = findall(!, feature_readable(alg, A))
    @argcheck(isempty(i),
              ArgumentError("FeatureDistance has no value to read for $(feature_asset_labels(nx, i)): $(unreadable_message(alg))"))
    return nothing
end
"""
    unreadable_message(alg::AbstractFeatureCollapseAlgorithm) -> String

State why a collapse has no value to read for an asset, and the remedy, for the refusal of [`assert_feature_readable`](@ref). [`LastRow`](@ref) reads the last row alone, so it names [`LastActiveRow`](@ref) as a remedy too.

# Related

  - [`assert_feature_readable`](@ref)
  - [`feature_readable`](@ref)
"""
function unreadable_message(::AbstractFeatureCollapseAlgorithm)
    return "each has no readable row in the window: no row at which it is active and every value column of `sel` holds data, observed or filled. Inside a fit, the entry of the optimiser drops such an asset as a non-investable asset. A direct call has no entry of a fit. Pass a view of the returns data at the assets to cluster, `port_opt_view(rd, i)`."
end
function unreadable_message(::LastObservation{<:LastRow})
    return "each is inactive at the last row of the window, or a value column of `sel` holds a placeholder there. Inside a fit, the entry of the optimiser drops such an asset as a non-investable asset. A direct call has no entry of a fit. Pass a view of the returns data at the assets to cluster, `port_opt_view(rd, i)`, or read each asset at its last readable row, `LastObservation(; alg = LastActiveRow())`."
end
"""
    asset_major(Z::Arr3Num, dims::Integer)

Give a window of time-varying features in the layout `observations × assets × features`, whichever trailing axis `dims` says the assets occupy.

# Related

  - [`active_feature_distance`](@ref)
"""
function asset_major(Z::Arr3Num, dims::Integer)
    return dims == 1 ? Z : PermutedDimsArray(Z, (1, 3, 2))
end
"""
    active_collapse(alg::AbstractCollapseAlgorithm, Zd::Arr3Num, w, A, nx)

Collapse each asset's features over its own active rows, with the weights of those rows.

The weights are resolved once against the whole window. Each asset keeps the entries of its active rows, and the collapse divides by their sum, so an inactive cell has no weight.

```math
\\begin{align}
\\bar{z}_{i,\\,k} &= \\dfrac{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i} z_{t,\\,i,\\,k}}{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i}}\\,,
\\end{align}
```

Where:

  - $(math_dict[:zbar_ik_feature])
  - $(math_dict[:z_tik_feature])
  - ``a_{t,\\,i}``: Entry of the active mask, `1` when asset ``i`` is active at observation ``t`` and `0` otherwise.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

[`MedianCollapse`](@ref) takes the same restricted weights.

# Algorithm

 1. Find the active rows of each asset.
 2. Check that the weights of each asset's active rows sum to more than zero.
 3. Collapse each asset over its active rows with [`collapse_features`](@ref), under the weights of those rows, and stack the results by asset.

# Arguments

  - `alg`: The collapse algorithm.
  - `Zd`: The window, `observations × assets × features`.
  - `w`: The resolved observation weights of the window, or `nothing`.
  - `A`: The active mask of the window, `observations × assets`.
  - `nx`: The asset names, or `nothing`.

# Validation

  - The weights of the active rows of each asset sum to more than zero. Raises an `ArgumentError` that names each other asset.

# Returns

  - `Zc::Matrix{<:Number}`: The collapsed features, `assets × features`.

# Related

  - [`AggregateFeatures`](@ref)
  - [`FeatureFallback`](@ref)
  - [`collapse_features`](@ref)
"""
function active_collapse(alg::AbstractCollapseAlgorithm, Zd::Arr3Num, w, A, nx)
    rows = [findall(view(A, :, i)) for i in axes(A, 2)]
    i = findall(r -> !(isnothing(w) || sum(active_weights(w, r)) > zero(eltype(w))), rows)
    @argcheck(isempty(i),
              ArgumentError("FeatureDistance collapses each asset over its active rows, and the observation weights of those rows sum to zero for $(feature_asset_labels(nx, i)), so the collapse has no value to read. Give a positive weight to one active row of each asset."))
    zs = [vec(collapse_features(alg, view(Zd, rows[i], i:i, :), active_weights(w, rows[i])))
          for i in axes(A, 2)]
    return permutedims(reduce(hcat, zs))
end
"""
    active_weights(w, r::AbstractVector{<:Integer})

Restrict the resolved observation weights of a window to the rows `r` at which an asset is active, keeping the kind of the weights through [`nothing_scalar_array_getindex`](@ref). No weights stay `nothing`.

# Related

  - [`active_collapse`](@ref)
  - [`nothing_scalar_array_getindex`](@ref)
"""
function active_weights(::Nothing, ::AbstractVector)
    return nothing
end
function active_weights(w::AbstractVector, r::AbstractVector)
    return nothing_scalar_array_getindex(w, r)
end
"""
    pair_distance(metric::Distances.SemiMetric, a::AbstractVector, b::AbstractVector)

Measure one pair of feature vectors, and apply the zero-feature-vector convention of [`patch_zero_feature_vectors!`](@ref) to the result.

# Related

  - [`patch_zero_feature_vectors!`](@ref)
  - [`active_feature_distance`](@ref)
"""
function pair_distance(metric::Distances.SemiMetric, a::AbstractVector, b::AbstractVector)
    d = metric(a, b)
    za = all(iszero, a)
    zb = all(iszero, b)
    if !isnan(d) || !(za || zb)
        return d
    end
    return za && zb ? zero(d) : one(d)
end
"""
    stack_metric(metric::Distances.SemiMetric, idx::AbstractVector{<:Integer})

Restrict a metric to the coordinates `idx` of a stack. A weighted metric keeps the weights of those coordinates, and every other metric is returned unchanged.

# Related

  - [`stack_rescale`](@ref)
  - [`StackObservations`](@ref)
"""
function stack_metric(metric::Distances.SemiMetric, ::AbstractVector)
    return metric
end
function stack_metric(metric::Distances.WeightedEuclidean, idx::AbstractVector)
    return Distances.WeightedEuclidean(metric.weights[idx])
end
function stack_metric(metric::Distances.WeightedSqEuclidean, idx::AbstractVector)
    return Distances.WeightedSqEuclidean(metric.weights[idx])
end
function stack_metric(metric::Distances.WeightedCityblock, idx::AbstractVector)
    return Distances.WeightedCityblock(metric.weights[idx])
end
function stack_metric(metric::Distances.WeightedMinkowski, idx::AbstractVector)
    return Distances.WeightedMinkowski(metric.weights[idx], metric.p)
end
"""
    stack_rescale(metric::Distances.SemiMetric, d::Number, idx::AbstractVector{<:Integer}, len::Integer)

Rescale the distance `d` of a pair, measured on the coordinates `idx` of a stack, to all `len` coordinates of the stack.

[`StackObservations`](@ref) measures a pair on the rows at which both assets are active. A metric that sums over the coordinates then sums over fewer of them, and gives a smaller distance than the whole window would. The rescale is the one that makes the sum an estimate of the sum over the whole window.

| Metric                                                                                                                                         | Rescale                                            |
|:---------------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------- |
| `Distances.Cityblock`, `Distances.TotalVariation`, `Distances.SqEuclidean`, `Distances.ChiSqDist`                                              | ``d \\cdot L / n``                                 |
| `Distances.Euclidean`                                                                                                                          | ``d \\cdot \\sqrt{L / n}``                         |
| `Distances.Minkowski`                                                                                                                          | ``d \\cdot (L / n)^{1/p}``                         |
| A weighted form                                                                                                                                | the ratio of the weight sums in place of ``L / n`` |
| [`AngularDist`](@ref), `Distances.CosineDist`, `Distances.CorrDist`, `Distances.Jaccard`, `Distances.BrayCurtis`, and the four mean deviations | none                                               |

Where ``L`` is `len` and ``n`` is the length of `idx`. Every coordinate of a row carries the features of one observation, so ``L / n`` is ``T / n`` in rows. Another metric has no method, and it refuses. The message names the method to add.

# Arguments

  - `metric`: The distance metric.
  - `d`: The distance of the pair on the coordinates `idx`.
  - `idx`: The coordinates of the stack that the pair shares.
  - `len`: The number of coordinates of the stack.

# Returns

  - `d`: The rescaled distance.

# Related

  - [`StackObservations`](@ref)
  - [`stack_metric`](@ref)
"""
function stack_rescale(metric::Distances.SemiMetric, ::Number, ::AbstractVector, ::Integer)
    return throw(ArgumentError("StackObservations measures a pair of assets on the rows at which both are active, and rescales the distance of a pair that shares fewer rows than the window to the whole window. The metric $(nameof(typeof(metric))) has no rescale. Add a method `PortfolioOptimisers.stack_rescale(metric::$(nameof(typeof(metric))), d, idx, len)` that returns `d`, measured on the coordinates `idx` of the stack, rescaled to its `len` coordinates. Or use AggregateDistances, which needs no rescale."))
end
function stack_rescale(::Union{<:AngularDist, <:Distances.CosineDist, <:Distances.CorrDist,
                               <:Distances.Jaccard, <:Distances.BrayCurtis,
                               <:Distances.MeanAbsDeviation, <:Distances.MeanSqDeviation,
                               <:Distances.RMSDeviation, <:Distances.NormRMSDeviation},
                       d::Number, ::AbstractVector, ::Integer)
    return d
end
function stack_rescale(::Union{<:Distances.Cityblock, <:Distances.TotalVariation,
                               <:Distances.SqEuclidean, <:Distances.ChiSqDist}, d::Number,
                       idx::AbstractVector, len::Integer)
    return d * len / length(idx)
end
function stack_rescale(::Distances.Euclidean, d::Number, idx::AbstractVector, len::Integer)
    return d * sqrt(len * one(d) / length(idx))
end
function stack_rescale(metric::Distances.Minkowski, d::Number, idx::AbstractVector,
                       len::Integer)
    return d * (len * one(d) / length(idx))^inv(metric.p)
end
function stack_rescale(metric::Union{<:Distances.WeightedCityblock,
                                     <:Distances.WeightedSqEuclidean}, d::Number,
                       idx::AbstractVector, ::Integer)
    return d * sum(metric.weights) / sum(view(metric.weights, idx))
end
function stack_rescale(metric::Distances.WeightedEuclidean, d::Number, idx::AbstractVector,
                       ::Integer)
    return d * sqrt(sum(metric.weights) / sum(view(metric.weights, idx)))
end
function stack_rescale(metric::Distances.WeightedMinkowski, d::Number, idx::AbstractVector,
                       ::Integer)
    return d * (sum(metric.weights) / sum(view(metric.weights, idx)))^inv(metric.p)
end
"""
    empty_pair_distance!(D::MatNum, pair::AbstractEmptyPairAlgorithm, E, de::FeatureDistance, win)

Give the distance of each pair of assets in `E`, which share no active row, by the rule `pair`.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`RefusePair`](@ref): refuse, and name each pair and the two other rules.
 2. [`DropFewerRows`](@ref): refuse, because the rule drops an asset at the entry of a fit, and a direct call has none. Name each pair and the remedy.
 3. [`FeatureFallback`](@ref): collapse each asset over its own active rows with [`active_collapse`](@ref), under the weights of the rule. Fill the distance of each pair with [`pair_distance`](@ref) of the two collapsed vectors, laid out by [`fallback_features`](@ref) as the collapse of `de` reads a row.

# Arguments

  - `D`: The distance matrix, `assets × assets`, written in place at each pair of `E`.
  - `pair`: The rule for an empty pair.
  - `E`: The empty pairs, `(i, j)` with `i < j`.
  - `de`: The feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`active_feature_distance`](@ref)
"""
function empty_pair_distance!(::MatNum, ::RefusePair, E::AbstractVector, ::FeatureDistance,
                              win::NamedTuple)
    return throw(ArgumentError("FeatureDistance reads each pair of assets at the rows at which both are active, and $(feature_asset_labels(win.nx, E)) share no active row. Three ways forward:\n  1. Drop the asset of each such pair with fewer active rows at the entry of a fit, `pair = DropFewerRows()`.\n  2. Measure each such pair by the features of its two assets, each over its own active rows, `pair = FeatureFallback()`.\n  3. Pass a view of the returns data at the assets to cluster, `port_opt_view(rd, i)`."))
end
function empty_pair_distance!(::MatNum, ::DropFewerRows, E::AbstractVector,
                              ::FeatureDistance, win::NamedTuple)
    return throw(ArgumentError("FeatureDistance reads each pair of assets at the rows at which both are active, and $(feature_asset_labels(win.nx, E)) share no active row. DropFewerRows drops an asset at the entry of a fit, and a direct call has no entry. Pass a view of the returns data at the assets to cluster, `port_opt_view(rd, i)`, or measure each such pair by the features of its two assets, `pair = FeatureFallback()`."))
end
function empty_pair_distance!(D::MatNum, fb::FeatureFallback, E::AbstractVector,
                              de::FeatureDistance, win::NamedTuple)
    (; Z, dims, A, nx) = win
    Zf = active_collapse(fb.alg, asset_major(Z, dims), collapse_weights(fb.w, Z), A, nx)
    for (i, j) in E
        D[i, j] = D[j, i] = pair_distance(de.metric,
                                          fallback_features(de.alg, view(Zf, i, :),
                                                            size(Z, 1)),
                                          fallback_features(de.alg, view(Zf, j, :),
                                                            size(Z, 1)))
    end
    return D
end
"""
    fallback_features(alg::AbstractFeatureCollapseAlgorithm, z::AbstractVector, T::Integer)

Lay out an asset's collapsed features as the collapse reads one asset. [`AggregateDistances`](@ref) reads the features of one row, `z` itself. [`StackObservations`](@ref) reads the stack of every row, so `z` fills each of the `T` rows.

# Related

  - [`FeatureFallback`](@ref)
  - [`empty_pair_distance!`](@ref)
"""
function fallback_features(::AggregateDistances, z::AbstractVector, ::Integer)
    return z
end
function fallback_features(::StackObservations, z::AbstractVector, T::Integer)
    return repeat(z; inner = T)
end
"""
    resolve_empty_pairs!(D::MatNum, pair::AbstractEmptyPairAlgorithm, E, de::FeatureDistance, win)

Hand the empty pairs `E` to [`empty_pair_distance!`](@ref), or return `D` unchanged when no pair is empty.

# Related

  - [`empty_pair_distance!`](@ref)
"""
function resolve_empty_pairs!(D::MatNum, pair::AbstractEmptyPairAlgorithm,
                              E::AbstractVector, de::FeatureDistance, win::NamedTuple)
    return isempty(E) ? D : empty_pair_distance!(D, pair, E, de, win)
end
"""
    active_feature_distance(de::FeatureDistance, win::NamedTuple)

Turn a window of time-varying features that holds an inactive cell into a distance matrix, reading each asset or each pair at its own active rows.

`win` holds the window `Z`, its layout `dims`, its active mask `A` and the asset names `nx`. [`distance`](@ref) calls this function only when a cell of the window is inactive, and only after [`assert_feature_readable`](@ref) checked that the collapse can read each asset. A window with no inactive cell takes the path of [`feature_distance`](@ref), so its result does not change.

# Algorithm

The type of `de.alg` selects one of four methods.

 1. [`LastObservation`](@ref): under [`LastRow`](@ref), or when every asset is active at the last row, measure the last row. Otherwise measure each asset at its last active row.
 2. [`AggregateFeatures`](@ref): resolve the weights with [`collapse_weights`](@ref), collapse each asset over its own active rows with [`active_collapse`](@ref), and measure the collapsed matrix.
 3. [`AggregateDistances`](@ref): measure each row, and add each entry of a pair only at a row where both assets are active, with that row's weight. Divide each entry by its own weight total, and give each pair with a zero total to [`resolve_empty_pairs!`](@ref).
 4. [`StackObservations`](@ref): measure the whole stack. Measure each pair that shares fewer rows than the window again, on the coordinates of its shared rows, with [`stack_metric`](@ref), and rescale it with [`stack_rescale`](@ref). Give each pair with no shared row to [`resolve_empty_pairs!`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`FeatureDistance`](@ref)
  - [`feature_distance`](@ref)
  - [`active_collapse`](@ref)
  - [`stack_rescale`](@ref)
  - [`empty_pair_distance!`](@ref)
"""
function active_feature_distance(de::FeatureDistance{<:Any, <:LastObservation},
                                 win::NamedTuple)
    (; Z, dims, A) = win
    if isa(de.alg.alg, LastRow) || all(view(A, size(A, 1), :))
        return feature_distance(de.metric, view(Z, size(Z, 1), :, :), dims)
    end
    Zd = asset_major(Z, dims)
    zs = [view(Zd, findlast(view(A, :, i))::Int, i, :) for i in axes(A, 2)]
    return feature_distance(de.metric, permutedims(reduce(hcat, zs)), 1)
end
function active_feature_distance(de::FeatureDistance{<:Any, <:AggregateFeatures},
                                 win::NamedTuple)
    (; Z, dims, A, nx) = win
    Zc = active_collapse(de.alg.alg, asset_major(Z, dims), collapse_weights(de.alg.w, Z), A,
                         nx)
    return feature_distance(de.metric, Zc, 1)
end
function active_feature_distance(de::FeatureDistance{<:Any, <:AggregateDistances},
                                 win::NamedTuple)
    (; Z, dims, A) = win
    w = collapse_weights(de.alg.w, Z)
    metric = de.metric
    T = Distances.result_type(metric, eltype(Z), eltype(Z))
    N = size(A, 2)
    D = zeros(T, N, N)
    S = zeros(T, N, N)
    Dt = Matrix{T}(undef, N, N)
    for t in axes(Z, 1)
        Zt = view(Z, t, :, :)
        Distances.pairwise!(metric, Dt, Zt; dims = dims)
        patch_zero_feature_vectors!(Dt, Zt, dims)
        wt = isnothing(w) ? one(T) : T(w[t])
        accumulate_active_pairs!(D, S, Dt, wt, view(A, t, :))
    end
    E = [(i, j) for j in axes(S, 2) for i in 1:(j - 1) if iszero(S[i, j])]
    D ./= S
    return resolve_empty_pairs!(D, de.alg.pair, E, de, win)
end
"""
    accumulate_active_pairs!(D::MatNum, S::MatNum, Dt::MatNum, wt::Number, m::AbstractVector{Bool})

Add one row of [`AggregateDistances`](@ref) to its accumulators: `wt * Dt[i, j]` to `D[i, j]` and `wt` to the weight total `S[i, j]` of each pair whose two assets are both active at the row, as `m` marks them.

# Related

  - [`active_feature_distance`](@ref)
  - [`AggregateDistances`](@ref)
"""
function accumulate_active_pairs!(D::MatNum, S::MatNum, Dt::MatNum, wt::Number,
                                  m::AbstractVector{Bool})
    @inbounds for j in axes(D, 2), i in axes(D, 1)
        if m[i] && m[j]
            D[i, j] += wt * Dt[i, j]
            S[i, j] += wt
        end
    end
    return nothing
end
function active_feature_distance(de::FeatureDistance{<:Any, <:StackObservations},
                                 win::NamedTuple)
    (; Z, dims, A) = win
    Zs = stack_observations(Z, dims)
    D = feature_distance(de.metric, Zs, 1)
    E = Tuple{Int, Int}[]
    for j in axes(A, 2), i in 1:(j - 1)
        rows = findall(t -> A[t, i] && A[t, j], axes(A, 1))
        if isempty(rows)
            push!(E, (i, j))
        elseif length(rows) < size(A, 1)
            D[i, j] = D[j, i] = shared_stack_distance(de.metric, Zs, (i, j), rows,
                                                      size(A, 1))
        end
    end
    return resolve_empty_pairs!(D, de.alg.pair, E, de, win)
end
"""
    shared_stack_distance(metric::Distances.SemiMetric, Zs::MatNum, (i, j), rows::AbstractVector, T::Integer)

Measure the pair `(i, j)` of a stack `Zs`, `assets × (observations · features)`, on the coordinates of its shared active `rows`, and rescale the distance to the whole window of `T` rows.

# Algorithm

 1. Name the coordinates of the shared rows, `t + (k - 1) T` for each shared row `t` and each feature `k`.
 2. Measure the two assets on those coordinates with [`pair_distance`](@ref), under the metric restricted by [`stack_metric`](@ref).
 3. Rescale the distance to every coordinate of the stack with [`stack_rescale`](@ref).

# Related

  - [`StackObservations`](@ref)
  - [`active_feature_distance`](@ref)
"""
function shared_stack_distance(metric::Distances.SemiMetric, Zs::MatNum,
                               (i, j)::Tuple{<:Integer, <:Integer}, rows::AbstractVector,
                               T::Integer)
    idx = [t + (k - 1) * T for k in 1:div(size(Zs, 2), T) for t in rows]
    d = pair_distance(stack_metric(metric, idx), view(Zs, i, idx), view(Zs, j, idx))
    return stack_rescale(metric, d, idx, size(Zs, 2))
end
"""
    feature_readable_mask(x, imsk::Option{<:BitVector}, rd) -> Option{BitVector}

Compose the Investable Mask of a prior with the assets that each [`FeatureDistance`](@ref) inside `x` can read, at the entry of a fit.

A fit reduces its prior, its optimiser and its returns data to the Investable Mask once, at its entry, with [`investable_reduction`](@ref). An asset that the prior admits can still have no value that a [`FeatureDistance`](@ref) of the fit can read in the window of its Asset Panel. A cell is readable where the asset is active and every value column of the Feature Selector holds data, observed or filled. [`LastRow`](@ref) reads nothing for an asset whose last cell is not readable, and every collapse reads nothing for an asset with no readable row. A static panel has one row, so an asset with a placeholder has none. This function drops such an asset from the mask before the reduction. The asset then departs as a non-investable asset: the reduction announces it, the fit gives it a zero weight, and it is listed on the Non-Investable Axis. Under [`DropFewerRows`](@ref) it also drops, from each pair with no shared active row, the asset with fewer active rows, until no pair is empty.

A default prior drops every asset that is not active over its whole window, so under it this function drops only an asset with a placeholder.

Every fit that clusters or builds a phylogeny calls it: [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref), [`SchurComplementHierarchicalRiskParity`](@ref), [`NestedClustered`](@ref), and a [`JuMPOptimiser`](@ref) through its phylogeny and centrality estimators. [`Stacking`](@ref) and [`SubsetResampling`](@ref) cluster only inside their inner optimisers, and each of those calls it at its own entry.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `Tuple` or a vector: compose the mask of each entry in turn.
 2. An estimator that holds a distance estimator, such as a [`ClustersEstimator`](@ref): forward to the estimator it holds.
 3. A [`FeatureDistance`](@ref) with no producer, on returns data that holds a panel: take the mask of readable cells in the rows that the collapse reads, with [`entry_activity`](@ref). Mark the readable assets with [`feature_readable`](@ref), keep those that `imsk` keeps, and hand the result to [`drop_empty_pairs!`](@ref). Return `nothing` when `imsk` is `nothing` and every asset stays.
 4. Every other case, a producer included, a panel whose every cell is readable, and no panel: return `imsk` unchanged.

# Arguments

  - `x`: An estimator, a vector or a `Tuple` of estimators.
  - `imsk`: The Investable Mask of the prior, or `nothing` when every asset is investable.
  - $(arg_dict[:rd])

# Validation

  - At least one asset stays. Raises an [`IsEmptyError`](@ref).

# Returns

  - `imsk::Option{BitVector}`: The composed mask, or `nothing` when every asset stays.

# Related

  - [`investable_reduction`](@ref)
  - [`investable_mask`](@ref)
  - [`feature_readable`](@ref)
  - [`drop_empty_pairs!`](@ref)
  - [`FeatureDistance`](@ref)
"""
function feature_readable_mask(::Any, imsk::Option{<:BitVector}, ::Any)
    return imsk
end
function feature_readable_mask(xs::Union{<:Tuple, <:AbstractVector},
                               imsk::Option{<:BitVector}, rd)
    for x in xs
        imsk = feature_readable_mask(x, imsk, rd)
    end
    return imsk
end
function feature_readable_mask(de::FeatureDistance{<:Any, <:Any, <:Any, Nothing},
                               imsk::Option{<:BitVector}, rd::ReturnsResult)
    return compose_readable_mask(de.alg, entry_activity(de, rd.pnl), imsk)
end
"""
    compose_readable_mask(alg::AbstractFeatureCollapseAlgorithm, A, imsk::Option{<:BitVector})

Compose the Investable Mask `imsk` with the assets that the collapse `alg` can read in a window whose active mask is `A`. This is step 3 of [`feature_readable_mask`](@ref), which states it. A window with no active mask, `A === nothing`, keeps `imsk`.

# Related

  - [`feature_readable_mask`](@ref)
  - [`feature_readable`](@ref)
  - [`drop_empty_pairs!`](@ref)
"""
function compose_readable_mask(::AbstractFeatureCollapseAlgorithm, ::Nothing,
                               imsk::Option{<:BitVector})
    return imsk
end
function compose_readable_mask(alg::AbstractFeatureCollapseAlgorithm,
                               A::AbstractMatrix{Bool}, imsk::Option{<:BitVector})
    keep = feature_readable(alg, A)
    if !isnothing(imsk)
        keep .&= imsk
    end
    drop_empty_pairs!(keep, empty_pair_rule(alg), A)
    @argcheck(any(keep),
              IsEmptyError("FeatureDistance reads each asset at its readable rows, where it is active and every value column of `sel` holds data, and no asset of the investable universe has a value to read in the window of the Asset Panel, so the fit has no asset left."))
    return isnothing(imsk) && all(keep) ? nothing : keep
end
"""
    entry_activity(de::FeatureDistance, pnl)

Give the mask of the cells that `de` can read in the window of the panel `pnl`, with [`feature_window_mask`](@ref), for the entry of a fit. Returns data with no panel gives `nothing`.

# Related

  - [`feature_readable_mask`](@ref)
  - [`feature_window_mask`](@ref)
"""
function entry_activity(::FeatureDistance, ::Nothing)
    return nothing
end
function entry_activity(de::FeatureDistance, pnl::AssetPanel)
    return last(feature_window_mask(de.alg, pnl, select_fields(pnl, de.sel, de.strict)))
end
"""
    feature_window_mask(alg, pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}}) -> (rows, A)

Name the rows that a collapse reads, and cut the mask of the cells it can read to them.

A cell is readable where the asset is active and every value column of `cols` holds data: [`readable_cells`](@ref) joins the active mask of the panel with [`feature_data_cells`](@ref). A cell that holds the placeholder of a blank that no fill reached holds no data. So a collapse reads it as it reads an inactive cell: each asset and each pair at its own readable rows. [`readable_window_rows`](@ref) names the rows on this mask, so [`LastActiveRow`](@ref) reads the last readable row of each asset. [`window_activity`](@ref) cuts the mask to the rows.

# Arguments

  - `alg`: The collapse algorithm.
  - `pnl`: The Asset Panel.
  - `cols`: The resolved columns, from [`select_fields`](@ref).

# Returns

  - `rows`: The `rows` keyword of [`feature_matrix`](@ref).
  - `A::Option{<:AbstractMatrix{Bool}}`: The mask of readable cells in those rows, see [`window_activity`](@ref).

# Related

  - [`feature_window`](@ref)
  - [`entry_activity`](@ref)
  - [`readable_cells`](@ref)
  - [`readable_window_rows`](@ref)
"""
function feature_window_mask(alg::AbstractFeatureCollapseAlgorithm, pnl::AssetPanel,
                             cols::AbstractVector{Tuple{Int, Int, Symbol}})
    R = readable_cells(pnl.amsk, feature_data_cells(pnl, cols))
    rows = readable_window_rows(alg, pnl, R)
    return rows, window_activity(R, rows)
end
"""
    readable_cells(amsk, o)

Join the active mask `amsk` of a panel with the mask `o` of its cells that hold data: a cell is readable where both are `true`. Either mask can be `nothing`, which marks every cell `true`.

# Related

  - [`feature_window_mask`](@ref)
  - [`feature_data_cells`](@ref)
"""
function readable_cells(amsk, ::Nothing)
    return amsk
end
function readable_cells(::Nothing, o::AbstractArray{Bool})
    return o
end
function readable_cells(amsk::AbstractMatrix{Bool}, o::AbstractMatrix{Bool})
    return amsk .& o
end
"""
    readable_window_rows(alg, pnl::AssetPanel, R)

Name the rows that a collapse reads on the mask `R` of readable cells. [`LastObservation`](@ref) forwards to its rule. [`LastActiveRow`](@ref) on a time-varying panel names them with [`last_active_rows`](@ref) on `R`, so an asset whose last active cell holds a placeholder is read at its last readable row. Every other case is [`collapse_rows`](@ref).

# Related

  - [`collapse_rows`](@ref)
  - [`feature_window_mask`](@ref)
"""
function readable_window_rows(alg, pnl::AssetPanel, ::Any)
    return collapse_rows(alg, pnl)
end
function readable_window_rows(alg::LastObservation, pnl::AssetPanel, R)
    return readable_window_rows(alg.alg, pnl, R)
end
function readable_window_rows(::LastActiveRow, ::AssetPanel, R::AbstractMatrix{Bool})
    return last_active_rows(R)
end
"""
    last_active_rows(A)

Name the rows from the earliest last `true` row of an asset in the mask `A`, `observations × assets`, to the last row: `r0:nobs`, where `r0` is `nobs` when no asset has a `true` row. These rows hold the last `true` row of every asset that has one. `nothing` names every row, `Colon()`.

# Related

  - [`collapse_rows`](@ref)
  - [`readable_window_rows`](@ref)
"""
function last_active_rows(::Nothing)
    return Colon()
end
function last_active_rows(A::AbstractMatrix{Bool})
    nobs = size(A, 1)
    r0 = minimum(i -> something(findlast(view(A, :, i)), nobs), axes(A, 2))
    return r0:nobs
end
"""
    static_window(amsk, Z::MatNum, dims::Integer)

Check the one-row mask of a static Feature Matrix against the matrix. A static matrix has one row of cells, so its mask is `1 × assets`, and `nothing` marks every asset readable.

# Validation

  - `size(amsk) == (1, assets)`, where `dims` names the asset axis of `Z`. Raises a `DimensionMismatch`.

# Related

  - [`distance`](@ref)
  - [`window_activity`](@ref)
"""
function static_window(::Nothing, ::MatNum, ::Integer)
    return nothing
end
function static_window(amsk::AbstractMatrix{Bool}, Z::MatNum, dims::Integer)
    ax = (1, size(Z, dims))
    @argcheck(size(amsk) == ax,
              DimensionMismatch("the mask of a static Feature Matrix has one row and one column per asset, so size(amsk) == $(ax) must hold. Got\nsize(amsk) => $(size(amsk))"))
    return amsk
end
"""
    empty_pair_rule(alg::AbstractFeatureCollapseAlgorithm)

Give the rule for a pair with no shared active row that a collapse carries: the `pair` field of [`AggregateDistances`](@ref) and [`StackObservations`](@ref), and `nothing` for a collapse that reads no pair.

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`drop_empty_pairs!`](@ref)
"""
function empty_pair_rule(::AbstractFeatureCollapseAlgorithm)
    return nothing
end
function empty_pair_rule(alg::Union{<:AggregateDistances, <:StackObservations})
    return alg.pair
end
"""
    drop_empty_pairs!(keep::BitVector, pair, A::AbstractMatrix{Bool}) -> BitVector

Drop, under [`DropFewerRows`](@ref), one asset of each pair with no shared active row, until no pair of the kept assets is empty.

Each pass finds the first empty pair `(i, j)` of the kept assets, with `i < j`, and drops the asset with fewer active rows in the window. On a tie it drops `j`, the asset that comes later in the universe. Every other rule keeps `keep` unchanged: [`RefusePair`](@ref) refuses in the kernel, and [`FeatureFallback`](@ref) measures the pair there.

# Arguments

  - `keep`: The assets kept so far, changed in place.
  - `pair`: The rule for an empty pair, or `nothing` for a collapse that reads no pair.
  - `A`: The active mask of the window, `observations × assets`.

# Returns

  - `keep::BitVector`: The kept assets.

# Related

  - [`DropFewerRows`](@ref)
  - [`feature_readable_mask`](@ref)
"""
function drop_empty_pairs!(keep::BitVector, ::Any, ::AbstractMatrix{Bool})
    return keep
end
function drop_empty_pairs!(keep::BitVector, ::DropFewerRows, A::AbstractMatrix{Bool})
    n = vec(sum(A; dims = 1))
    p = first_empty_pair(keep, A)
    while !isnothing(p)
        i, j = p
        keep[n[j] <= n[i] ? j : i] = false
        p = first_empty_pair(keep, A)
    end
    return keep
end
"""
    first_empty_pair(keep::BitVector, A::AbstractMatrix{Bool})

Find the first pair `(i, j)` of kept assets, with `i < j`, that shares no active row of `A`, in the order of `j` and then of `i`, or `nothing` when no pair is empty.

# Related

  - [`drop_empty_pairs!`](@ref)
"""
function first_empty_pair(keep::BitVector, A::AbstractMatrix{Bool})
    k = findall(keep)
    for j in k, i in k
        if i < j && !any(t -> A[t, i] && A[t, j], axes(A, 1))
            return (i, j)
        end
    end
    return nothing
end

public feature_readable, empty_pair_distance!, drop_empty_pairs!
