"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator dropping assets and observations with excessive missing data from price-level data.

The *asset universe is fitted state*: the training window decides which assets survive (per-column missing fraction at most `col_thr`), and applying the fitted result to an unseen window subsets it to that same universe — so train weights and test returns always refer to the same assets. Observation (row) filtering is window-local: rows whose missing fraction across the surviving assets exceeds `row_thr` are dropped from whichever window is being transformed.

This estimator is the library's **only** missing-data filter: [`prices_to_returns`](@ref) removes no observation and no asset, because deleting either is a **Universe Policy** and a policy is fitted on a training window and replayed by name, which a stateless conversion cannot do (ADR 0133). Only the asset series `X` (and the matching implied volatility columns, and the feature matrix, whose axes are parallel to `X`) participate; factor and benchmark series pass through unchanged.

# Algorithm

## Fit

 1. Count the missing observations of each asset column with [`is_missing_value`](@ref), and divide each count by the observation total, giving `frac`.
 2. Keep the assets whose fraction does not exceed `col_thr`, and check that one asset at least survives.
 3. Return a [`MissingDataFilterResult`](@ref) holding the surviving asset names and `row_thr`.

## Apply

 1. Find the columns of the window whose names are in the fitted universe, and check that one at least is present.
 2. Count the missing assets of each row over those columns alone, and keep the rows whose count does not exceed `row_thr` of the column total.
 3. Rebuild `X` from the kept rows and the kept columns.
 4. Subselect the implied volatilities on the kept columns, and `ivpa` with them when it is a vector. The implied volatility series keeps every row, because its own clock is not the one that was filtered.
 5. View the Asset Panel with [`panel_carrier_view`](@ref) at the kept rows and the kept columns, handing it the asset names so that a tensor Panel Field whose labels *are* the asset names ([`features_are_assets`](@ref)) is cut on its label axis too.
 6. Rebuild the [`PricesResult`](@ref). The factor series `F` and the benchmark series `B` pass through untouched.

The two thresholds count opposite axes: `col_thr` counts the missing rows of a column and drops columns, and `row_thr` counts the missing columns of a row and drops rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
        cache::Option{<:AbstractPartialFitState} = nothing,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields.

# Online form

The column filter is universe-only, so a stepped filter counts the gaps of every block in `cache` through [`partial_fit_transform`](@ref), passes the rows through, and reads the [`MissingDataFilterResult`](@ref) of the whole history out through [`fit_preprocessing`](@ref) with no data; a [`Pipeline`](@ref) applies that universe as a view at its read-out. The row filter is row-local only at `row_thr = 1`, where it drops nothing: a lower threshold drops a row kept earlier when a column leaves the universe, so [`supports_partial_fit`](@ref) answers `false` for it and a Pipeline refuses it at warm-up by name.

Both thresholds admit zero, which is the tightest policy the estimator can state: `col_thr = 0` keeps the assets with no gap at all, and `row_thr = 0` keeps the observations with no gap at all.

## Validation

  - `0 <= col_thr <= 1`.
  - `0 <= row_thr <= 1`.

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
    Maximum allowed fraction `[0, 1]` of missing observations per asset column; assets above it are dropped from the universe at fit time. `0` tolerates no gap at all, and keeps the assets priced at every observation of the training window.
    """
    col_thr
    """
    Maximum allowed fraction `[0, 1]` of missing assets per observation row; rows above it are dropped from the window being transformed. `0` tolerates no gap at all, and keeps the observations at which every surviving asset is priced.
    """
    row_thr
    """
    $(field_dict[:pfcache])
    """
    cache
    function MissingDataFilter(col_thr::Number, row_thr::Number,
                               cache::Option{<:AbstractPartialFitState})
        @argcheck(zero(col_thr) <= col_thr <= one(col_thr), DomainError)
        @argcheck(zero(row_thr) <= row_thr <= one(row_thr), DomainError)
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

Carries the asset universe selected on the training window plus the row threshold needed to transform further windows. Produced by [`fit_preprocessing`](@ref), consumed by [`apply_preprocessing`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MissingDataFilter`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct MissingDataFilterResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets that survived the training window (the fitted universe).
    """
    nx
    """
    Maximum allowed fraction `[0, 1]` of missing assets per observation row.
    """
    row_thr
end
function fit_preprocessing(mdf::MissingDataFilter,
                           pr::PricesResult)::MissingDataFilterResult
    vals = values(pr.X)
    frac = vec(count(is_missing_value, vals; dims = 1)) / size(vals, 1)
    keep = frac .<= mdf.col_thr
    @argcheck(any(keep),
              IsEmptyError("MissingDataFilter with col_thr = $(mdf.col_thr) drops every asset in the training window"))
    return MissingDataFilterResult(TimeSeries.colnames(pr.X)[keep], mdf.row_thr)
end
function apply_preprocessing(res::MissingDataFilterResult, pr::PricesResult)::PricesResult
    cols = findall(in(res.nx), TimeSeries.colnames(pr.X))
    @argcheck(!isempty(cols),
              IsEmptyError("none of the fitted universe assets $(res.nx) are present in the data window"))
    vals = values(pr.X)[:, cols]
    rows = findall(vec(count(is_missing_value, vals; dims = 2)) .<=
                   length(cols) * res.row_thr)
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X)[rows], vals[rows, :],
                             TimeSeries.colnames(pr.X)[cols])
    iv, ivpa = if isnothing(pr.iv)
        nothing, pr.ivpa
    else
        ivv = values(pr.iv)[:, cols]
        ivm = TimeSeries.TimeArray(TimeSeries.timestamp(pr.iv), ivv,
                                   TimeSeries.colnames(pr.iv)[cols])
        ivm, isa(pr.ivpa, VecNum) ? pr.ivpa[cols] : pr.ivpa
    end
    pnl = panel_carrier_view(pr.pnl, rows, cols, string.(TimeSeries.colnames(pr.X)))
    #! The span is a fact about the instruments, so the surviving window's span is the
    #! carrier's viewed at the rows and columns that survived, never one re-derived.
    span = span_carrier_view(pr.span, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X),
                             cols)
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = iv, ivpa = ivpa, pnl = pnl,
                        span = span)
end
export MissingDataFilter, MissingDataFilterResult
