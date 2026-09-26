"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator that drops the assets and the observations of a price window whose share of missing prices exceeds a threshold.

The fit decides the asset universe. An asset survives when its share of missing observations over the training window does not exceed `col_thr`. The apply step replays that universe by name on a later window, in the fitted order, and refuses a window that lacks a fitted asset. The train weights and the test returns therefore name the same assets in the same order. The row filter reads each window alone. The apply step drops an observation when its share of missing assets, over the fitted universe, exceeds `row_thr`.

This estimator is the only missing-data filter of the library. [`prices_to_returns`](@ref) removes no observation and no asset, because a deletion is a **Universe Policy**. A policy needs a fit on a training window and a replay by name on later windows, and a stateless conversion has no fit to replay. Only the asset series `X` decides what survives. The factor, benchmark and implied volatility series, and the Asset Panel, follow the surviving assets and observations, because the carrier states one clock.

A share is the count of missing entries divided by the count of entries, computed in the type of the threshold. `29 / 100` is `0.29` in `Float64` and `29f0 / 100f0` is `0.29f0` in `Float32`, so a threshold written as a decimal keeps an asset or an observation whose share equals it. A `Rational` threshold compares the exact share.

# Algorithm

## Fit

 1. Count the missing observations of each asset column with [`is_missing_value`](@ref).
 2. Keep the assets whose share of missing observations does not exceed `col_thr`. Raise an `IsEmptyError` when no asset survives.
 3. Return a [`MissingDataFilterResult`](@ref) that holds the names of the surviving assets, in the column order of the training window, and `row_thr`.

## Apply

 1. Find the column of each fitted asset in the window, in the fitted order. Raise an `ArgumentError` that names the first absent asset.
 2. Count the missing assets of each observation over those columns. Keep the observations whose share of missing assets does not exceed `row_thr`. Raise an `IsEmptyError` when no observation survives.
 3. Return [`port_opt_view`](@ref) of the carrier at the kept timestamps and the fitted columns. It reads the factor, benchmark and implied volatility series at the kept timestamps, subsets a per-asset benchmark, the implied volatilities and a vector `ivpa` to the fitted columns, and views the Asset Panel and the Listing Span at the kept rows and columns.

The two thresholds count opposite axes. `col_thr` counts the missing observations of an asset and drops assets. `row_thr` counts the missing assets of an observation and drops observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
        cache::Option{<:AbstractPartialFitState} = nothing,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields. Both thresholds admit zero. `col_thr = 0` keeps the assets with no missing observation, and `row_thr = 0` keeps the observations with no missing asset. Both thresholds admit one, which drops nothing on its axis.

## Validation

  - `0 <= col_thr <= 1`, else `DomainError`.
  - `0 <= row_thr <= 1`, else `DomainError`.

# Online form

The column filter changes only the universe. A stepped filter counts the missing observations of every block in `cache` through [`partial_fit_transform`](@ref) and passes the rows through. [`fit_preprocessing`](@ref) with no data reads the [`MissingDataFilterResult`](@ref) of the whole history out of that count, and a [`Pipeline`](@ref) applies that universe as a view at its read-out. The row filter has an online form only at `row_thr = 1`, where it drops nothing. At a lower threshold, the filter can drop a row at a later step that it kept at an earlier step, when an asset leaves the universe. [`supports_partial_fit`](@ref) answers `false` for it, and a Pipeline refuses it by name at warm-up.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 NaN; 102.0 NaN; 104.0 105.0],
                     [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> res = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr);

julia> res.nx
1-element Vector{Symbol}:
 :A
```

# Related

  - [`MissingDataFilterResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`PriceGapFill`](@ref)
  - [`PricesResult`](@ref)
"""
@concrete struct MissingDataFilter <: AbstractPricesPreprocessingEstimator
    """
    Largest share `[0, 1]` of missing observations that an asset can have over the training window and stay in the universe. `0` keeps the assets with a price at every observation of the training window.
    """
    col_thr
    """
    Largest share `[0, 1]` of missing assets that an observation can have and stay in the window. `0` keeps the observations at which every asset of the fitted universe has a price.
    """
    row_thr
    """
    $(field_dict[:pfcache])
    """
    cache
    function MissingDataFilter(col_thr::Number, row_thr::Number,
                               cache::Option{<:AbstractPartialFitState})
        assert_closed_unit_interval(col_thr, :col_thr)
        assert_closed_unit_interval(row_thr, :row_thr)
        return new{typeof(col_thr), typeof(row_thr), typeof(cache)}(col_thr, row_thr, cache)
    end
end
function MissingDataFilter(; col_thr::Number = 1.0, row_thr::Number = 1.0,
                           cache::Option{<:AbstractPartialFitState} = nothing)::MissingDataFilter
    return MissingDataFilter(col_thr, row_thr, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`MissingDataFilter`](@ref).

It holds the asset universe that the training window selected and the row threshold that the apply step reads on a later window. [`fit_preprocessing`](@ref) returns it and [`apply_preprocessing`](@ref) reads it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MissingDataFilter`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct MissingDataFilterResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets that survived the training window, in its column order. This is the fitted universe.
    """
    nx
    """
    Largest share `[0, 1]` of missing assets that an observation can have and stay in the window.
    """
    row_thr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `k` missing entries out of `n` are a share that does not exceed `thr`.

The share is `k / n` computed in the type of `thr`, so a decimal threshold keeps the count that it names. `29 / 100` is `0.29` in `Float64`, but `100 * 0.29` is `28.999999999999996`, so a test of `k <= n * thr` drops 29 missing entries of 100 at `thr = 0.29`. A `Float32` threshold divides in `Float32`, and a `Rational` threshold divides exactly.

# Arguments

  - `k`: Count of missing entries.
  - `n`: Count of entries, positive.
  - `thr`: Largest share that passes.

# Returns

  - `flag::Bool`: `true` when the share does not exceed `thr`.

# Related

  - [`MissingDataFilter`](@ref)
  - [`is_missing_value`](@ref)
"""
function share_at_most(k::Integer, n::Integer, thr::Real)::Bool
    return k * one(thr) / n <= thr
end
function fit_preprocessing(mdf::MissingDataFilter,
                           pr::PricesResult)::MissingDataFilterResult
    vals = values(pr.X)
    miss = vec(count(is_missing_value, vals; dims = 1))
    keep = share_at_most.(miss, size(vals, 1), mdf.col_thr)
    @argcheck(any(keep),
              IsEmptyError("MissingDataFilter with col_thr = $(mdf.col_thr) drops every asset in the training window"))
    return MissingDataFilterResult(TimeSeries.colnames(pr.X)[keep], mdf.row_thr)
end
function apply_preprocessing(res::MissingDataFilterResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    cols = Vector{Int}(undef, length(res.nx))
    for (k, name) in pairs(res.nx)
        j = findfirst(==(name), names)
        @argcheck(!isnothing(j),
                  ArgumentError(unknown_variable_msg(name, string.(names), :nx;
                                                     consequence = "the window must contain the whole fitted universe")))
        cols[k] = j
    end
    miss = vec(count(is_missing_value, view(values(pr.X), :, cols); dims = 2))
    rows = findall(share_at_most.(miss, length(cols), res.row_thr))
    @argcheck(!isempty(rows),
              IsEmptyError("MissingDataFilter with row_thr = $(res.row_thr) drops every observation of the window"))
    #! The carrier states one clock, so every series it holds is read at the surviving
    #! timestamps and the fitted columns, by the view that owns that rule.
    return port_opt_view(pr, TimeSeries.timestamp(pr.X)[rows], cols)
end
export MissingDataFilter, MissingDataFilterResult
