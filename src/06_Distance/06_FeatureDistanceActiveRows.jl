"""
    active_window(amsk, Z::Arr3Num, dims::Integer)

Check the active mask of a window against the window, and give it only when a cell of it is inactive.

The active mask marks, for each observation of the window and each asset, whether the asset is in the universe. A window with no inactive cell reads as `nothing`, so the collapse takes the path that reads every row, and its result is the one it gives with no mask.

# Algorithm

 1. If `amsk` is `nothing`, return `nothing`.
 2. Otherwise, check that `amsk` is `observations × assets` for the layout that `dims` declares.
 3. Return `nothing` when every entry of `amsk` is `true`, and `amsk` otherwise.

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
    blank_unreadable(Z::Arr3Num, A::AbstractMatrix{Bool}, dims::Integer)

Give a copy of a window of time-varying features with a zero at each cell that the mask `A` marks unreadable.

A collapse reads each asset, or each pair of assets, only at its own readable rows, so the value of an unreadable cell must reach neither the distance nor a refusal. [`distance`](@ref) checks this copy with [`assert_feature_matrix`](@ref) and collapses it. A `NaN`, or a value outside the domain of the metric, at an unreadable cell then refuses nothing. No result of a collapse reads it either. [`StackObservations`](@ref) measures the whole stack, and then replaces the distance of each pair that holds an unreadable cell. [`AggregateDistances`](@ref) measures every asset at each row, and keeps only the pairs whose two assets are active at the row. The zero is the value that [`feature_window`](@ref) writes at a placeholder, and that the panel collapse of a meta-optimiser writes at an inactive cell. The copy keeps the element type of `Z`.

# Algorithm

 1. Lay the mask `A` out against `Z`, giving `M`. At `dims = 1`, `M` is `A`. At `dims = 2`, `M` is `A` reshaped to `observations × 1 × assets`.
 2. Broadcast `ifelse` over `M` and `Z`. Keep each cell of `Z` where `M` is `true`, and write `zero(eltype(Z))` elsewhere.

# Arguments

  - $(arg_dict[:Z])
  - `A`: The mask of readable cells of the window, `observations × assets`.
  - $(arg_dict[:dims])

# Returns

  - `Zr::Array`: The window, the same size and element type as `Z`, with a zero at each unreadable cell.

# Related

  - [`distance`](@ref)
  - [`active_window`](@ref)
  - [`active_feature_distance`](@ref)
  - [`feature_window`](@ref)
"""
function blank_unreadable(Z::Arr3Num, A::AbstractMatrix{Bool}, dims::Integer)
    M = dims == 1 ? A : reshape(A, size(A, 1), 1, size(A, 2))
    return ifelse.(M, Z, zero(eltype(Z)))
end
"""
    feature_readable(alg, A::AbstractMatrix{Bool}) -> BitVector

Mark the assets that a collapse can read in a window, from the active mask of the window.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`LastObservation`](@ref) forwards to its rule.
 2. Under [`LastRow`](@ref), an asset is readable when it is active at the last row.
 3. Under every other rule and collapse, an asset is readable when it is active at one row of the window at least.

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
function feature_readable(::Union{LastActiveRow, AbstractFeatureCollapseAlgorithm},
                          A::AbstractMatrix{Bool})
    return BitVector(vec(any(A; dims = 1)))
end
"""
    feature_asset_labels(nx, i::AbstractVector{<:Integer}) -> String
    feature_asset_labels(nx, E::AbstractVector{<:Tuple}) -> String

Name the assets at the positions `i`, or the pairs of positions `E`, for a refusal. The text quotes the names when `nx` holds them, and the positions otherwise.

# Algorithm

The method that Julia selects is the algorithm.

 1. If `nx` is `nothing`, quote `i`, or the pairs of `E` joined by commas.
 2. Otherwise, quote `nx[i]`, or the pair of names of each pair of `E` joined by commas.

# Arguments

  - `nx`: The asset names, or `nothing`.
  - `i`: The positions of the assets.
  - `E`: The pairs of positions, `(i, j)`.

# Returns

  - `s::String`: The phrase that names the assets, for the message of a refusal.

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

# Algorithm

 1. If `A` is `nothing`, every cell is readable, so return `nothing`.
 2. Otherwise, find the assets that [`feature_readable`](@ref) does not mark, giving `i`.
 3. Refuse when `i` is not empty. The message names each asset of `i` with [`feature_asset_labels`](@ref), and gives the cause and the remedy from [`unreadable_message`](@ref).

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

# Algorithm

The method that Julia selects is the algorithm.

 1. Under [`LastObservation`](@ref) with [`LastRow`](@ref), the cause is the last row. The remedies are a view of the returns data and [`LastActiveRow`](@ref).
 2. Under every other collapse, the cause is a window with no readable row. The remedy is a view of the returns data.

# Arguments

  - `alg`: The collapse algorithm.

# Returns

  - `s::String`: The cause and the remedy.

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

# Algorithm

 1. At `dims = 1`, return `Z`.
 2. At `dims = 2`, return a `PermutedDimsArray` of `Z` that swaps its two last axes. It copies nothing.

# Arguments

  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - `Zd::AbstractArray{<:Number, 3}`: The window, `observations × assets × features`.

# Related

  - [`active_feature_distance`](@ref)
"""
function asset_major(Z::Arr3Num, dims::Integer)
    return dims == 1 ? Z : PermutedDimsArray(Z, (1, 3, 2))
end
"""
    active_collapse(alg::AbstractCollapseAlgorithm, Zd::Arr3Num, w, A, nx)

Collapse each asset's features over its own active rows, with the weights of those rows.

The caller resolves the weights once against the whole window, with [`collapse_weights`](@ref). Each asset keeps the entries of its active rows, and the collapse divides by their sum, so an inactive cell has no weight.

# Mathematical definition

```math
\\begin{align}
\\bar{z}_{i,\\,k} &= \\dfrac{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i} z_{t,\\,i,\\,k}}{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i}}\\,,
\\end{align}
```

Where:

  - $(math_dict[:zbar_ik_feature])
  - $(math_dict[:z_tik_feature])
  - $(math_dict[:a_ti_window])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

[`MedianCollapse`](@ref) takes the same restricted weights.

# Algorithm

 1. Find the active rows of each asset, giving `rows`.
 2. If `w` is not `nothing`, check that the weights of the active rows of each asset sum to more than zero.
 3. Collapse each asset over its active rows with [`collapse_features`](@ref), under the weights of those rows from [`active_weights`](@ref), giving `zs`.
 4. Stack `zs` by asset, `assets × features`.

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

Restrict the resolved observation weights of a window to the rows `r` at which an asset is active. [`nothing_scalar_array_getindex`](@ref) keeps the kind of the weights. No weights stay `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. If `w` is `nothing`, return `nothing`.
 2. Otherwise, return the entries of `w` at `r` with [`nothing_scalar_array_getindex`](@ref).

# Arguments

  - `w`: The resolved observation weights of the window, or `nothing`.
  - `r`: The active rows of one asset.

# Returns

  - `wr::Option{<:VecNum}`: The weights of the rows `r`, or `nothing`.

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

# Algorithm

 1. Measure `a` against `b` with `metric`, giving `d`.
 2. If `d` is not `NaN`, or neither vector is all zero, return `d`.
 3. Otherwise, return zero when both vectors are all zero, and one when only one of them is.

# Arguments

  - `metric`: The distance metric.
  - `a`, `b`: The feature vectors of the two assets.

# Returns

  - `d::Number`: The distance of the pair.

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

Restrict a metric to the coordinates `idx` of a stack. A metric whose parameters hold one entry per coordinate keeps the entries of those coordinates. The function returns every other metric unchanged.

A metric that another package defines, with parameters that hold one entry per coordinate, needs a method of this function too. Without it, the metric measures the shared coordinates with all its parameters, and it raises its own error before [`stack_rescale`](@ref) can refuse.

# Algorithm

The method that Julia selects is the algorithm.

 1. For a weighted metric of Distances.jl, `Distances.WeightedHamming` included, keep `weights[idx]`.
 2. For `Distances.PeriodicEuclidean`, keep `periods[idx]`.
 3. For `Distances.Mahalanobis` and `Distances.SqMahalanobis`, keep the principal submatrix `qmat[idx, idx]`, which is the quadratic form on the shared coordinates.
 4. For every other metric, return `metric`.

# Arguments

  - `metric`: The distance metric.
  - `idx`: The coordinates of the stack that a pair shares.

# Returns

  - `metric::Distances.SemiMetric`: The metric on the coordinates `idx`.

# Related

  - [`stack_rescale`](@ref)
  - [`StackObservations`](@ref)
"""
function stack_metric(metric::Distances.SemiMetric, ::AbstractVector)
    return metric
end
function stack_metric(metric::Distances.WeightedHamming, idx::AbstractVector)
    return Distances.WeightedHamming(metric.weights[idx])
end
function stack_metric(metric::Distances.PeriodicEuclidean, idx::AbstractVector)
    return Distances.PeriodicEuclidean(metric.periods[idx])
end
function stack_metric(metric::Distances.Mahalanobis, idx::AbstractVector)
    return Distances.Mahalanobis(metric.qmat[idx, idx])
end
function stack_metric(metric::Distances.SqMahalanobis, idx::AbstractVector)
    return Distances.SqMahalanobis(metric.qmat[idx, idx])
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

[`StackObservations`](@ref) measures a pair on the rows at which both assets are active. A metric that sums over the coordinates then sums over fewer of them, and gives a smaller distance than the whole window would. The rescale makes the sum an estimate of the sum over the whole window.

# Mathematical definition

| Metric                                                                                                                                          | Rescale                                            |
|:----------------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------- |
| `Distances.Cityblock`, `Distances.TotalVariation`, `Distances.SqEuclidean`, `Distances.ChiSqDist`                                               | ``d \\cdot L / n``                                 |
| `Distances.Euclidean`                                                                                                                           | ``d \\cdot \\sqrt{L / n}``                         |
| `Distances.Minkowski`                                                                                                                           | ``d \\cdot (L / n)^{1/p}``                         |
| A weighted form                                                                                                                                 | the ratio of the weight sums in place of ``L / n`` |
| [`AngularDist`](@ref), `Distances.CosineDist`, `Distances.CorrDist`, `Distances.Jaccard`, `Distances.BrayCurtis`, and the three mean deviations | none                                               |

Where ``L`` is `len` and ``n`` is the length of `idx`. The coordinates of one row carry the features of one observation, so ``L / n`` is ``T`` over the count of shared rows. A weighted form takes ``W / W_{n}``, where ``W`` is the sum of its weights and ``W_{n}`` is the sum of its weights at `idx`.

# Algorithm

The method that Julia selects is the algorithm.

 1. For a metric of the table, return `d` times the factor of its row.
 2. For every other metric, refuse, and name the method to add.

# Arguments

  - `metric`: The distance metric.
  - `d`: The distance of the pair on the coordinates `idx`.
  - `idx`: The coordinates of the stack that the pair shares.
  - `len`: The number of coordinates of the stack.

# Validation

  - `metric` is a metric of the table. Raises an `ArgumentError` that names the method `PortfolioOptimisers.stack_rescale` to add for the type of `metric`.

# Returns

  - `d`: The rescaled distance.

# Related

  - [`StackObservations`](@ref)
  - [`stack_metric`](@ref)
"""
function stack_rescale(metric::Distances.SemiMetric, ::Number, ::AbstractVector, ::Integer)
    return throw(ArgumentError("StackObservations measures a pair of assets on the rows at which both are active, and rescales the distance of a pair that shares fewer rows than the window to the whole window. The metric $(nameof(typeof(metric))) has no rescale. Add a method `PortfolioOptimisers.stack_rescale(metric::$(nameof(typeof(metric))), d, idx, len)` that returns `d`, measured on the coordinates `idx` of the stack, rescaled to its `len` coordinates. Or use AggregateDistances, which needs no rescale."))
end
function stack_rescale(::Union{AngularDist, Distances.CosineDist, Distances.CorrDist,
                               Distances.Jaccard, Distances.BrayCurtis,
                               Distances.MeanAbsDeviation, Distances.MeanSqDeviation,
                               Distances.RMSDeviation}, d::Number, ::AbstractVector,
                       ::Integer)
    return d
end
function stack_rescale(::Union{Distances.Cityblock, Distances.TotalVariation,
                               Distances.SqEuclidean, Distances.ChiSqDist}, d::Number,
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
function stack_rescale(metric::Union{Distances.WeightedCityblock,
                                     Distances.WeightedSqEuclidean}, d::Number,
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

 1. Under [`RefusePair`](@ref), refuse. Name each pair and the two other rules.
 2. Under [`DropFewerRows`](@ref), refuse, because the rule drops an asset at the entry of a fit, and a direct call has none. Name each pair and the remedy.
 3. Under [`FeatureFallback`](@ref), resolve the weights of the rule with [`collapse_weights`](@ref). Collapse each asset over its own active rows with [`active_collapse`](@ref), under the collapse and the weights of the rule, giving `Zf`.
 4. Then lay out the two rows of `Zf` of each pair as the collapse of `de` reads one asset, with [`fallback_features`](@ref). Write their [`pair_distance`](@ref) to `D[i, j]` and `D[j, i]`.

# Arguments

  - `D`: The distance matrix, `assets × assets`, written in place at each pair of `E`.
  - `pair`: The rule for an empty pair.
  - `E`: The empty pairs, `(i, j)` with `i < j`.
  - `de`: The feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

# Validation

  - `pair` is a [`FeatureFallback`](@ref). [`RefusePair`](@ref) and [`DropFewerRows`](@ref) raise an `ArgumentError` that names each pair of `E`.

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

Lay out the collapsed features of an asset as the collapse reads one asset. [`AggregateDistances`](@ref) reads the features of one row, `z` itself. [`StackObservations`](@ref) reads the stack of every row, so `z` fills each of the `T` rows.

# Algorithm

The method that Julia selects is the algorithm.

 1. Under [`AggregateDistances`](@ref), return `z`.
 2. Under [`StackObservations`](@ref), repeat each entry of `z` `T` times, `repeat(z; inner = T)`. Feature `k` then fills the coordinates `t + (k - 1) T` of the stack, as [`stack_observations`](@ref) lays them out.

# Arguments

  - `alg`: The collapse algorithm of the estimator.
  - `z`: The collapsed features of one asset.
  - `T`: The number of observations of the window.

# Returns

  - `zf::AbstractVector`: The features of the asset, as the collapse reads them.

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

Give the empty pairs `E` to [`empty_pair_distance!`](@ref), or return `D` unchanged when no pair is empty.

# Algorithm

 1. If `E` is empty, return `D`.
 2. Otherwise, return [`empty_pair_distance!`](@ref) of `D` under the rule `pair`.

# Arguments

  - `D`: The distance matrix, `assets × assets`.
  - `pair`: The rule for an empty pair.
  - `E`: The empty pairs, `(i, j)` with `i < j`.
  - `de`: The feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

# Returns

  - $(ret_dict[:Ddist])

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

`win` holds the window `Z`, its layout `dims`, its active mask `A` and the asset names `nx`. `Z` holds a zero at each cell that `A` marks `false`, from [`blank_unreadable`](@ref), so no step of a collapse reads the stored value of such a cell. [`distance`](@ref) calls this function only when a cell of the window is inactive, and only after [`assert_feature_readable`](@ref) checked that the collapse can read each asset. A window with no inactive cell takes the path of [`feature_distance`](@ref), so its result does not change.

Under [`AggregateDistances`](@ref) and [`StackObservations`](@ref), an entry can be a weighted mean or a rescaled distance. So [`float_if_integer`](@ref) turns an integer result type of the metric, such as that of `Distances.Cityblock` on a window of integers, into a floating point type. The matrix of a `Rational` window stays exact.

# Mathematical definition

Under [`LastObservation`](@ref) with [`LastActiveRow`](@ref), the collapse reads each asset at its last active row.

```math
\\begin{align}
\\tau_{i} &= \\max\\left\\{t : a_{t,\\,i} = 1\\right\\}\\,,\\\\
D_{i,\\,j} &= m\\left(\\boldsymbol{z}_{\\tau_{i},\\,i},\\, \\boldsymbol{z}_{\\tau_{j},\\,j}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:D_ij_dist])
  - $(math_dict[:a_ti_window])
  - ``\\tau_{i}``: Last active row of asset ``i`` in the window.
  - ``\\boldsymbol{z}_{t,\\,i}``: Feature vector of asset ``i`` at observation ``t``.
  - ``m``: Distance metric, `de.metric`.

Under [`AggregateFeatures`](@ref), ``D_{i,\\,j}`` is the metric of the collapsed vectors that [`active_collapse`](@ref) defines. [`AggregateDistances`](@ref) and [`StackObservations`](@ref) each state their own form for a window with an inactive cell.

# Algorithm

The type of `de.alg` selects one of four methods.

 1. Under [`LastObservation`](@ref) with [`LastRow`](@ref), or when every asset is active at the last row, measure the last row.
 2. Under [`LastObservation`](@ref) otherwise, read the features of each asset at its last active row, giving `zs`, and measure the stack of `zs`.
 3. Under [`AggregateFeatures`](@ref), resolve the weights with [`collapse_weights`](@ref). Collapse each asset over its own active rows with [`active_collapse`](@ref), giving `Zc`, and measure `Zc`.
 4. Under [`AggregateDistances`](@ref), resolve the weights, and measure each row into `Dt`. Add each entry of `Dt` to `D`, and the weight of the row to the weight total `S`, only for a pair whose two assets are active at the row, with [`accumulate_active_pairs!`](@ref).
 5. Then check that the weight total of each asset, on the diagonal of `S`, is not zero. Collect each pair with a zero total, giving `E`. Divide `D` by `S`, and give `E` to [`resolve_empty_pairs!`](@ref).
 6. Under [`StackObservations`](@ref), measure the whole stack, giving `D`. Measure each pair that shares fewer rows than the window again, with [`shared_stack_distance`](@ref). Collect each pair with no shared row, giving `E`, and give `E` to [`resolve_empty_pairs!`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

# Validation

  - Under [`AggregateDistances`](@ref), the weights of the active rows of each asset do not sum to zero. Raises an `ArgumentError` that names each other asset.
  - A pair with no shared active row takes the rule `pair` of the collapse. See [`empty_pair_distance!`](@ref).

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
    #! The entries are weighted means, so an integer distance or weight needs a float type.
    Td = Distances.result_type(metric, eltype(Z), eltype(Z))
    T = float_if_integer(isnothing(w) ? Td : promote_type(Td, eltype(w)))
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
    #! The diagonal total is the weight of the active rows of an asset. A zero total leaves
    #! `0 / 0` on the diagonal, and no pair rule reaches the diagonal.
    z = findall(i -> iszero(S[i, i]), axes(S, 1))
    @argcheck(isempty(z),
              ArgumentError("FeatureDistance reads each asset at its active rows, and the observation weights of those rows sum to zero for $(feature_asset_labels(win.nx, z)), so no distance of the asset has a value to read. Give a positive weight to one active row of each asset."))
    E = [(i, j) for j in axes(S, 2) for i in 1:(j - 1) if iszero(S[i, j])]
    D ./= S
    return resolve_empty_pairs!(D, de.alg.pair, E, de, win)
end
"""
    accumulate_active_pairs!(D::MatNum, S::MatNum, Dt::MatNum, wt::Number, m::AbstractVector{Bool})

Add one row of [`AggregateDistances`](@ref) to its accumulators, for each pair whose two assets are both active at the row, as `m` marks them.

# Algorithm

 1. For each pair `(i, j)` with `m[i]` and `m[j]`, add `wt * Dt[i, j]` to `D[i, j]`, and add `wt` to the weight total `S[i, j]`.

# Arguments

  - `D`: The sum of the weighted distances, `assets × assets`, changed in place.
  - `S`: The weight total of each pair, `assets × assets`, changed in place.
  - `Dt`: The distance matrix of the row.
  - `wt`: The weight of the row.
  - `m`: The active mask of the row, one entry per asset.

# Returns

  - `nothing`.

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
    #! A rescale and a fallback can give a fraction, so an integer distance needs a float type.
    Ds = feature_distance(de.metric, Zs, 1)
    D = convert(Matrix{float_if_integer(eltype(Ds))}, Ds)
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

 1. Name the coordinates of the shared rows, `t + (k - 1) T` for each shared row `t` and each feature `k`, giving `idx`.
 2. Measure the two assets on `idx` with [`pair_distance`](@ref), under the metric that [`stack_metric`](@ref) restricts to `idx`, giving `d`.
 3. Rescale `d` to every coordinate of the stack with [`stack_rescale`](@ref).

# Arguments

  - `metric`: The distance metric.
  - `Zs`: The stack, `assets × (observations · features)`.
  - `(i, j)`: The pair of assets.
  - `rows`: The rows at which both assets are active.
  - `T`: The number of observations of the window.

# Returns

  - `d::Number`: The rescaled distance of the pair.

# Related

  - [`StackObservations`](@ref)
  - [`active_feature_distance`](@ref)
"""
function shared_stack_distance(metric::Distances.SemiMetric, Zs::MatNum,
                               (i, j)::Tuple{Integer, Integer}, rows::AbstractVector,
                               T::Integer)
    idx = [t + (k - 1) * T for k in 1:div(size(Zs, 2), T) for t in rows]
    d = pair_distance(stack_metric(metric, idx), view(Zs, i, idx), view(Zs, j, idx))
    return stack_rescale(metric, d, idx, size(Zs, 2))
end
"""
    feature_readable_mask(x, imsk::Option{<:BitVector}, rd) -> Option{BitVector}

Compose the Investable Mask of a prior with the assets that each [`FeatureDistance`](@ref) inside `x` can read, at the entry of a fit.

A fit reduces its prior, its optimiser and its returns data to the Investable Mask once, at its entry, with [`investable_reduction`](@ref). An asset that the prior admits can still have no value that a [`FeatureDistance`](@ref) of the fit can read in the window of its Asset Panel. A cell is readable where the asset is active and every value column of the Feature Selector holds data, observed or filled. [`LastRow`](@ref) reads nothing for an asset whose last cell is not readable, and every collapse reads nothing for an asset with no readable row. A static panel has one row, so an asset with a placeholder has none. This function drops such an asset from the mask before the reduction. The asset then departs as a non-investable asset. The reduction announces it, the fit gives it a zero weight, and the Non-Investable Axis lists it. Under [`DropFewerRows`](@ref), the function also drops the asset with fewer active rows from each pair with no shared active row, until no pair is empty.

A default prior keeps only the Coverage Universe, the assets that are active at every row of its window. So under a default prior, this function drops only an asset with a placeholder.

Every fit that clusters or builds a phylogeny calls it: [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref), [`SchurComplementHierarchicalRiskParity`](@ref), [`NestedClustered`](@ref), and a [`JuMPOptimiser`](@ref) through its phylogeny and centrality estimators. [`Stacking`](@ref) and [`SubsetResampling`](@ref) cluster only inside their inner optimisers, and each of those calls it at its own entry.

# Algorithm

The method that Julia selects is the algorithm.

 1. For a `Tuple` or a vector, compose the mask of each entry in turn.
 2. For an estimator that holds a distance estimator, such as a [`ClustersEstimator`](@ref), forward to the estimator it holds.
 3. For a [`FeatureDistance`](@ref) with no producer, on returns data that holds a panel, take the mask of readable cells in the rows that the collapse reads, with [`entry_activity`](@ref). Compose it with `imsk` through [`compose_readable_mask`](@ref).
 4. In every other case, a producer and returns data with no panel included, return `imsk` unchanged.

A panel whose every cell is readable takes step 3, and it gives a mask equal to `imsk`.

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
function feature_readable_mask(xs::Union{Tuple, AbstractVector}, imsk::Option{<:BitVector},
                               rd)
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

Compose the Investable Mask `imsk` with the assets that the collapse `alg` can read in a window whose active mask is `A`. This is step 3 of [`feature_readable_mask`](@ref). A window with no active mask, `A === nothing`, keeps `imsk`.

# Algorithm

 1. If `A` is `nothing`, return `imsk`.
 2. Mark the assets that `alg` can read with [`feature_readable`](@ref), giving `keep`. If `imsk` is not `nothing`, keep only the assets that `imsk` keeps too.
 3. Drop assets from `keep` with [`drop_empty_pairs!`](@ref), under the rule of [`empty_pair_rule`](@ref) for `alg`.
 4. Check that `keep` keeps one asset at least.
 5. Return `nothing` when `imsk` is `nothing` and `keep` keeps every asset, and `keep` otherwise.

# Arguments

  - `alg`: The collapse algorithm.
  - `A`: The mask of readable cells of the window, `observations × assets`, or `nothing`.
  - `imsk`: The Investable Mask of the prior, or `nothing` when every asset is investable.

# Validation

  - At least one asset stays. Raises an [`IsEmptyError`](@ref).

# Returns

  - `imsk::Option{BitVector}`: The composed mask, or `nothing` when every asset stays.

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

# Algorithm

The method that Julia selects is the algorithm.

 1. If `pnl` is `nothing`, return `nothing`.
 2. Otherwise, resolve `de.sel` against `pnl` with [`select_fields`](@ref). Return the mask that [`feature_window_mask`](@ref) gives for those columns.

# Arguments

  - `de`: Feature distance estimator.
  - `pnl`: The Asset Panel of the returns data, or `nothing`.

# Returns

  - `A::Option{<:AbstractMatrix{Bool}}`: The mask of readable cells in the rows that the collapse reads, or `nothing`.

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

A cell is readable where the asset is active and every value column of `cols` holds data. A cell that holds the placeholder of a blank that no fill reached holds no data. So a collapse reads it as it reads an inactive cell, and reads each asset and each pair at its own readable rows. The rows come from this mask, so [`LastActiveRow`](@ref) reads the last readable row of each asset.

# Algorithm

 1. Join the active mask of the panel with the cells of `cols` that hold data, from [`feature_data_cells`](@ref), through [`readable_cells`](@ref). This gives `R`.
 2. Name the rows that the collapse reads on `R` with [`readable_window_rows`](@ref), giving `rows`.
 3. Cut `R` to `rows` with [`window_activity`](@ref), giving `A`.

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

Join the active mask `amsk` of a panel with the mask `o` of its cells that hold data. A cell is readable where both are `true`. Either mask can be `nothing`, which marks every cell `true`.

# Algorithm

The method that Julia selects is the algorithm.

 1. If `o` is `nothing`, return `amsk`.
 2. If `amsk` is `nothing`, return `o`.
 3. Otherwise, return `amsk .& o`.

# Arguments

  - `amsk`: The active mask of the panel, or `nothing`.
  - `o`: The mask of the cells that hold data, or `nothing`.

# Returns

  - `R`: The mask of readable cells, or `nothing` when every cell is readable.

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

Name the rows that a collapse reads on the mask `R` of readable cells. [`LastActiveRow`](@ref) reads the rows of `R`, so it reads an asset whose last active cell holds a placeholder at its last readable row.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`LastObservation`](@ref) forwards to its rule.
 2. Under [`LastActiveRow`](@ref), with a matrix `R` of a time-varying panel, name the rows with [`last_active_rows`](@ref) on `R`.
 3. In every other case, a static panel and `R === nothing` included, name the rows with [`collapse_rows`](@ref).

# Arguments

  - `alg`: The collapse algorithm, or the rule of a [`LastObservation`](@ref).
  - `pnl`: The Asset Panel.
  - `R`: The mask of readable cells, `observations × assets` on a time-varying panel, one entry per asset on a static panel, or `nothing`.

# Returns

  - `rows`: A range of observations, or `Colon()`. It is the value of the `rows` keyword of [`feature_matrix`](@ref).

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

Name the rows from the earliest last `true` row of an asset in the mask `A`, `observations × assets`, to the last row. These rows hold the last `true` row of every asset that has one.

# Algorithm

The method that Julia selects is the algorithm.

 1. If `A` is `nothing`, every row is active, so return `Colon()`.
 2. Otherwise, find the last `true` row of each asset, or `nobs` for an asset with none. The earliest of these rows is `r0`.
 3. Return `r0:nobs`.

# Arguments

  - `A`: The mask, `observations × assets`, or `nothing`.

# Returns

  - `rows::Union{UnitRange{Int}, Colon}`: The rows `r0:nobs`, or `Colon()`.

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

# Algorithm

 1. If `amsk` is `nothing`, return `nothing`.
 2. Otherwise, check the size of `amsk`, and return `amsk`.

# Arguments

  - `amsk`: The mask of the static Feature Matrix, `1 × assets`, or `nothing`.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Validation

  - `size(amsk) == (1, assets)`, where `dims` names the asset axis of `Z`. Raises a `DimensionMismatch`.

# Returns

  - `A::Option{<:AbstractMatrix{Bool}}`: `amsk`.

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

Give the rule that a collapse holds for a pair with no shared active row.

# Algorithm

The method that Julia selects is the algorithm.

 1. Under [`AggregateDistances`](@ref) and [`StackObservations`](@ref), return the `pair` field.
 2. Under every other collapse, which reads no pair, return `nothing`.

# Arguments

  - `alg`: The collapse algorithm.

# Returns

  - `pair::Option{<:AbstractEmptyPairAlgorithm}`: The rule for an empty pair, or `nothing`.

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`drop_empty_pairs!`](@ref)
"""
function empty_pair_rule(::AbstractFeatureCollapseAlgorithm)
    return nothing
end
function empty_pair_rule(alg::Union{AggregateDistances, StackObservations})
    return alg.pair
end
"""
    drop_empty_pairs!(keep::BitVector, pair, A::AbstractMatrix{Bool}) -> BitVector

Drop, under [`DropFewerRows`](@ref), one asset of each pair with no shared active row, until no pair of the kept assets is empty.

Each pass drops the asset with fewer active rows in the window from the first empty pair. On a tie it drops `j`, the asset that comes later in the universe. Every other rule keeps `keep` unchanged. [`RefusePair`](@ref) refuses in the kernel, and [`FeatureFallback`](@ref) measures the pair there.

# Algorithm

The method that Julia selects is the algorithm.

 1. Under every rule but [`DropFewerRows`](@ref), return `keep`.
 2. Under [`DropFewerRows`](@ref), count the active rows of each asset in `A`, giving `n`.
 3. Find the first empty pair `(i, j)` of the kept assets with [`first_empty_pair`](@ref), giving `p`.
 4. While `p` is not `nothing`, drop `j` from `keep` when `n[j] <= n[i]`, and `i` otherwise. Then find the next empty pair.
 5. Return `keep`.

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

Find the first pair `(i, j)` of kept assets, with `i < j`, that shares no active row of `A`, or `nothing` when no pair is empty.

# Algorithm

 1. Find the kept assets, giving `k`.
 2. For each `j` of `k`, and then each `i` of `k` with `i < j`, return `(i, j)` when no row of `A` marks both assets active.
 3. Return `nothing`.

# Arguments

  - `keep`: The kept assets.
  - `A`: The active mask of the window, `observations × assets`.

# Returns

  - `p::Option{Tuple{Int, Int}}`: The first empty pair, or `nothing`.

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
