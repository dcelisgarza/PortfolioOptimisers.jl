"""
    attribution_key_align(blk::AttributionBlock, T::Integer, key::Nothing)
    attribution_key_align(blk::AttributionBlock, T::Integer, key::AttributionRowKey)

Return the lag-aligned history of a factor model block over the observations of a series that may name its rows.

The series of a cross-validation names its rows, and its folds usually cover fewer rows than a prior fitted on the whole sample. A `key` of `nothing` names no row, so the function aligns the block at the tail with [`attribution_align`](@ref). A `key` with the fields `idx` and `ts` names the position and the timestamp of each observation of the series. [`attribution_key_pair`](@ref) picks the key that the block and the series both carry. The function then aligns the whole block, so each block row keeps the exposures of the row `lag` observations before it. Last, it cuts the aligned block with [`attribution_window`](@ref) to the rows that [`attribution_match_rows`](@ref) matches, and sets its `rows` to the matched observations of the series through [`rebuild_struct`](@ref). The first observation of a fold that starts after the first block row therefore keeps its exposures, which a cut of the block before the alignment would lose. When the two carry no common key, the function aligns at the tail.

# Arguments

  - `blk`: The block as bare arrays, from [`attribution_block_arrays`](@ref) or [`attribution_array_block`](@ref).
  - `T`: The number of observations the caller's return series carries.
  - `key`: The position `idx` and the timestamp `ts` of each observation of the series, each `nothing` when the series does not carry it, or `nothing`.

# Validation

  - With a common key, the key of the series has `T` entries and the key of the block has one entry for each observation of the block, else a `DimensionMismatch` is raised, and the rules of [`attribution_match_rows`](@ref) hold.
  - The rules of [`attribution_align`](@ref).

# Returns

  - `al::AlignedAttributionBlock`: The aligned history, as [`attribution_align`](@ref) states it. Its `rows` are the observations of the series that the block decomposes, and they are consecutive.

# Related

  - [`attribution_align`](@ref)
  - [`attribution_key_pair`](@ref)
  - [`attribution_match_rows`](@ref)
  - [`factor_attribution`](@ref)
"""
function attribution_key_align(blk::AttributionBlock, T::Integer, ::Nothing)
    return attribution_align(blk, T)
end
function attribution_key_align(blk::AttributionBlock, T::Integer, key::AttributionRowKey)
    kp = attribution_key_pair(blk.key, key)
    if isnothing(kp)
        return attribution_align(blk, T)
    end
    @argcheck(length(kp.series) == T,
              DimensionMismatch("the row key of the return series ($(length(kp.series)) entries) must have one entry for each observation of the series ($T)"))
    @argcheck(length(kp.block) == size(blk.f, 1),
              DimensionMismatch("the row key of the factor model block ($(length(kp.block)) entries) must have one entry for each observation of the block ($(size(blk.f, 1)))"))
    al = attribution_align(blk, size(blk.f, 1))
    rows, arows = attribution_match_rows(kp.block, kp.series, blk.lag)
    return rebuild_struct(attribution_window(al, arows), (; rows = rows))
end
"""
$(DocStringExtensions.TYPEDEF)

One kind of row key that a factor model block and a return series share: the key of the block and the key of the series.

This type is an implementation detail and is not intended for direct use. [`attribution_key_pair`](@ref) makes it, and [`attribution_key_align`](@ref) reads it. The two keys have one type, positions or timestamps, so a record names them.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`attribution_key_pair`](@ref)
  - [`attribution_match_rows`](@ref)
"""
@concrete struct RowKeyPair
    """
    `block`: The key of each block row.
    """
    block
    """
    `series`: The key of each observation of the series, of the same kind as `block`.
    """
    series
end
"""
    attribution_key_pair(bk::AttributionRowKey, key::AttributionRowKey)
    attribution_key_pair(b::AbstractVector, s::AbstractVector)
    attribution_key_pair(b::Option{<:AbstractVector}, s::Option{<:AbstractVector})

Return the row keys of a factor model block and of a return series that the two share, or `nothing`.

A timestamp names an observation in every returns data that holds it, so the function takes the timestamps when both carry them. A position names an observation in one returns data alone, so the function takes the positions only when one of the two carries no timestamps. The positions of the block are then positions in the returns data that the prior read, and those of the series are positions in the returns data that the cross-validation split. The two agree only when the prior read the returns data that the cross-validation split, and the function cannot check that.

# Arguments

  - `bk`: The row key of the block, from [`attribution_row_key`](@ref).
  - `key`: The position `idx` and the timestamp `ts` of each observation of the series, each `nothing` when the series does not carry it.
  - `b`, `s`: One kind of key of the block and of the series, each `nothing` when its side does not carry it. The two methods over them pair the keys when both are present, and answer `nothing` otherwise.

# Returns

  - `kp::Option{<:RowKeyPair}`: The key of the block under `block` and that of the series under `series`, or `nothing` when the two share no key.

# Related

  - [`attribution_align`](@ref)
  - [`attribution_match_rows`](@ref)
"""
function attribution_key_pair(bk::AttributionRowKey, key::AttributionRowKey)
    kp = attribution_key_pair(bk.ts, key.ts)
    return isnothing(kp) ? attribution_key_pair(bk.idx, key.idx) : kp
end
function attribution_key_pair(b::AbstractVector, s::AbstractVector)
    return RowKeyPair(b, s)
end
function attribution_key_pair(::Option{<:AbstractVector}, ::Option{<:AbstractVector})
    return nothing
end
"""
    attribution_match_rows(bk::AbstractVector, sk::AbstractVector, lag::Integer)

Return the rows of a return series that a factor model block decomposes, and the aligned block row of each.

Block row `t` of a fit with exposure lag `lag` decomposes its returns with the exposures of block row `t - lag`, so the first `lag` block rows decompose nothing. Aligned row `a` is block row `a + lag`. The function finds the aligned row whose key equals the key of each observation of the series. The observations before the first match and after the last match fall outside the block, as the observations of a fold before the first fitted row do, and the attribution leaves them out, when their keys lie outside the range of the keys of the aligned rows. Every other observation must match, because a key inside that range with no row is a gap of the block, for example an observation that the fit dropped for a Shared Series gap. A gap at the head of the series is refused as a gap between two matches is.

# Arguments

  - `bk`: The key of each block row, positions or timestamps.
  - `sk`: The key of each observation of the series, of the same kind as `bk`.
  - `lag`: The exposure lag of the block.

# Validation

  - At least one observation of the series matches an aligned row. Raises a `DimensionMismatch`.
  - Every observation between the first match and the last match, and every observation whose key lies inside the range of the keys of the aligned rows, matches an aligned row. Raises an `ArgumentError` that names the first unmatched observations.
  - The matched aligned rows increase strictly, so the series repeats no observation and keeps the order of the block. Raises an `ArgumentError`.

# Returns

The tuple `(rows, arows)`:

  - `rows::UnitRange{Int}`: The observations of the series that the block decomposes.
  - `arows::Vector{Int}`: The aligned block row of each of them.

# Related

  - [`attribution_align`](@ref)
  - [`attribution_key_pair`](@ref)
"""
function attribution_match_rows(bk::AbstractVector, sk::AbstractVector, lag::Integer)
    Tb = length(bk)
    pos = Dict(zip(view(bk, (lag + 1):Tb), 1:(Tb - lag)))
    m = [get(pos, k, 0) for k in sk]
    kept = findall(!iszero, m)
    @argcheck(!isempty(kept),
              DimensionMismatch("no observation of the return series is a row that the factor model block decomposes. The block decomposes $(Tb - lag) rows, those after its exposure lag of $lag. Fit the prior on returns data that covers the rows of the series."))
    rows = first(kept):last(kept)
    # An unmatched observation before the first match is outside the block only when its key
    # comes before the first aligned row. A key inside the range of the aligned rows is a gap
    # of the block, wherever the observation stands in the series.
    lo, hi = extrema(view(bk, (lag + 1):Tb))
    gaps = findall(k -> iszero(m[k]) && (k in rows || lo <= sk[k] <= hi), eachindex(m))
    @argcheck(isempty(gaps),
              ArgumentError("observations $(first(gaps, 5)) of the return series lie inside the rows of the factor model block, and the block holds no row for them. Fit the prior on the returns data that the cross-validation split, or drop those observations."))
    arows = m[rows]
    @argcheck(all(>(0), diff(arows)),
              ArgumentError("the return series repeats a row of the factor model block, or does not keep its order. A realised attribution reads the observations in the order of time, so its folds must not overlap and must come in the order of time."))
    return rows, arows
end
"""
    attribution_series_key(pred::MultiPeriodPredictionResult)

Return the row key of the series that a cross-validation produced.

The series is the folds stacked in order, as [`attribution_prediction_history`](@ref) stacks them, so its key is the keys of the folds stacked in the same order. A fold records its positions in `idx`, and the timestamps of its rows in `rd.ts`. The series carries positions only when every fold carries them, and timestamps only when every fold carries them, as `pred.mrd.ts` holds them.

# Arguments

  - `pred`: A multi-period prediction result.

# Returns

  - `key::AttributionRowKey`: The position `idx` of each observation of the series in the returns data that the cross-validation split, or `nothing`, and the timestamp `ts` of each observation of the series, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_prediction_history`](@ref)
  - [`PredictionResult`](@ref)
"""
function attribution_series_key(pred::MultiPeriodPredictionResult)
    idx = if any(p -> isnothing(p.idx), pred.pred)
        nothing
    else
        reduce(vcat, p.idx for p in pred.pred)
    end
    return AttributionRowKey(idx, pred.mrd.ts)
end
