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
 5. Read `sq` from [`features_are_assets`](@ref), and view `nz` at the kept columns when `sq` is `true`.
 6. View the Asset Panel with [`panel_carrier_view`](@ref) at the kept rows and the kept columns, handing it the asset names.
 7. Rebuild the [`PricesResult`](@ref). The factor series `F` and the benchmark series `B` pass through untouched.

The two thresholds count opposite axes: `col_thr` counts the missing rows of a column and drops columns, and `row_thr` counts the missing columns of a row and drops rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields.

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
  - [`Imputer`](@ref)
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
    function MissingDataFilter(col_thr::Number, row_thr::Number)
        @argcheck(zero(col_thr) <= col_thr <= one(col_thr), DomainError)
        @argcheck(zero(row_thr) <= row_thr <= one(row_thr), DomainError)
        return new{typeof(col_thr), typeof(row_thr)}(col_thr, row_thr)
    end
end
function MissingDataFilter(; col_thr::Number = 1.0,
                           row_thr::Number = 1.0)::MissingDataFilter
    return MissingDataFilter(col_thr, row_thr)
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
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window.

The *imputation parameters are fitted state*: each asset's fill value is computed from the training window's observed (non-missing) prices with the configured [`Num_VecToScaM`](@ref), and applying the fitted result to an unseen window fills that window's missing observations with the *training* values — never with statistics of the window being transformed, which is exactly the leakage a fit/apply contract exists to prevent.

Assets with no observed values in the training window get no fill value and are left untouched at apply time; combine with [`MissingDataFilter`](@ref) to drop them instead.

# Algorithm

## Fit

 1. For each asset column, collect the observed prices. An entry [`is_missing_value`](@ref) accepts is left out.
 2. Skip an asset whose column holds no observed price. It gets no fill value, and no entry in the result.
 3. Reduce the observed prices of the column to one value with `stat`, giving that asset's fill value.
 4. Return an [`ImputerResult`](@ref) holding the fitted asset names and their fill values, aligned.

## Apply

 1. Copy the price values of the window, so the input is not mutated.
 2. For each fitted asset name, find its column in the window. Skip a name the window does not carry.
 3. Replace every missing entry of that column with that asset's fitted fill value.
 4. Rebuild `X` from the filled values, keeping the timestamps and the column names, then rebuild the [`PricesResult`](@ref). Every other field passes through untouched.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Imputer(;
        stat::Num_VecToScaM = MedianValue(),
    ) -> Imputer

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 1.0; NaN 3.0; 104.0 5.0],
                     [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> res = fit_preprocessing(Imputer(), pr);

julia> pv = apply_preprocessing(res, pr);

julia> values(pv.X)[2, 1]
102.0
```

# Related

  - [`ImputerResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`Num_VecToScaM`](@ref)
"""
@concrete struct Imputer <: AbstractPricesPreprocessingEstimator
    """
    Reducer computing an asset's fill value from its observed training prices ([`Num_VecToScaM`](@ref)).
    """
    stat
    function Imputer(stat::Num_VecToScaM)
        return new{typeof(stat)}(stat)
    end
end
function Imputer(; stat::Num_VecToScaM = MedianValue())::Imputer
    return Imputer(stat)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of an [`Imputer`](@ref).

Carries the per-asset fill values computed on the training window. Produced by [`fit_preprocessing`](@ref), consumed by [`apply_preprocessing`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Imputer`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct ImputerResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets with a fitted fill value.
    """
    nx
    """
    Fill values, aligned with `nx`.
    """
    v
end
function fit_preprocessing(imp::Imputer, pr::PricesResult)::ImputerResult
    names = TimeSeries.colnames(pr.X)
    vals = values(pr.X)
    keep = Vector{Int}(undef, 0)
    v = Vector{Any}(undef, 0)
    for i in axes(vals, 2)
        obs = identity.([x for x in view(vals, :, i) if !is_missing_value(x)])
        if isempty(obs)
            continue
        end
        push!(keep, i)
        push!(v, vec_to_real_measure(imp.stat, obs))
    end
    return ImputerResult(names[keep], identity.(v))
end
function apply_preprocessing(res::ImputerResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(pr.X))
    for (name, fill_val) in zip(res.nx, res.v)
        j = findfirst(==(name), names)
        if isnothing(j)
            continue
        end
        for i in axes(vals, 1)
            if is_missing_value(vals[i, j])
                vals[i, j] = fill_val
            end
        end
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, names)
    #! A fill states a price, not a listing, so the span passes through untouched: the
    #! clock and the asset axis are the ones it was derived on.
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                        pnl = pr.pnl, span = pr.span)
end
export MissingDataFilter, MissingDataFilterResult, Imputer, ImputerResult
