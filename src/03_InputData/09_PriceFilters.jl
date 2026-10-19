"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules that decide how the fit of a [`MissingDataFilter`](@ref) reads the column shares.

All concrete subtypes should subtype `AbstractMissingDataFilterAlgorithm`. A member of the family goes in the `alg` field of a [`MissingDataFilter`](@ref). The family exists because the two thresholds read each other. `col_thr` drops assets, and the rows that `row_thr` keeps depend on the assets that stay. When the row filter drops an observation, the share of missing observations of each kept asset changes. One rule reads the column shares one time, over the whole training window. The other rule reads them again over the kept observations until no asset drops.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractMissingDataFilterAlgorithm` and implement the following method:

## fit_universe

  - `fit_universe(alg::AbstractMissingDataFilterAlgorithm, mdf::MissingDataFilter, pr::PricesResult) -> Vector{Int}`: The column positions in `pr.X` of the assets that the fit keeps, in the column order of `pr.X`.

### Arguments

  - `alg`: The concrete subtype instance.
  - `mdf`: The filter, which holds `col_thr` and `row_thr`.
  - `pr`: The training window.

### Returns

  - `cols::Vector{Int}`: The positions of the kept assets. One position at least.

### Examples

```jldoctest
julia> struct KeepFirst <: PortfolioOptimisers.AbstractMissingDataFilterAlgorithm end

julia> PortfolioOptimisers.fit_universe(::KeepFirst, ::MissingDataFilter, ::PricesResult) = [1];

julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 2), [100.0 50.0; 101.0 51.0],
                     [\"A\", \"B\"]);

julia> fit_preprocessing(MissingDataFilter(; alg = KeepFirst()), PricesResult(; X = X)).nx
1-element Vector{Symbol}:
 :A
```

# Related

  - [`SinglePassFilter`](@ref)
  - [`FixedPointFilter`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`fit_universe`](@ref)
"""
abstract type AbstractMissingDataFilterAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Reads the column shares of a [`MissingDataFilter`](@ref) one time, over every observation of the training window.

The fit keeps an asset when its share of missing observations over the whole training window does not exceed `col_thr`. It does not read the shares again after the row filter. So over the observations that the apply step keeps, a kept asset can miss a larger share than `col_thr`. [`FixedPointFilter`](@ref) removes that gap. This rule is the fit of the releases before it, and it gives the same universe as [`FixedPointFilter`](@ref) when `row_thr = 1`.

# Related

  - [`AbstractMissingDataFilterAlgorithm`](@ref)
  - [`FixedPointFilter`](@ref)
  - [`MissingDataFilter`](@ref)
"""
struct SinglePassFilter <: AbstractMissingDataFilterAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Reads the column shares of a [`MissingDataFilter`](@ref) again over the kept observations, until no asset drops.

This is the default rule. At the fixed point, each kept asset misses at most a share `col_thr` of the observations that the apply step keeps on the training window. Each kept observation misses at most a share `row_thr` of the kept assets. The rule finds the kept observations again from the whole training window on each pass, as the apply step finds them. So an observation that a dropped asset removed comes back when the asset goes.

The universe only gets smaller, and a pass that drops no asset stops the loop. So the loop stops after at most one pass for each asset. At `row_thr = 1` the row filter drops nothing, and the first pass is the fixed point.

# Algorithm

 1. Keep the assets whose share of missing observations over the whole training window does not exceed `col_thr`, with [`column_pass`](@ref).
 2. Find the observations that the row filter keeps over those assets, with [`kept_rows`](@ref).
 3. Keep the assets whose share of missing observations over the kept observations does not exceed `col_thr`.
 4. Stop when step 3 drops no asset. Otherwise go back to step 2 with the assets of step 3.

# Related

  - [`AbstractMissingDataFilterAlgorithm`](@ref)
  - [`SinglePassFilter`](@ref)
  - [`MissingDataFilter`](@ref)
"""
struct FixedPointFilter <: AbstractMissingDataFilterAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator that drops the assets and the observations of a price window whose share of missing prices exceeds a threshold.

The fit decides the asset universe. Under the default `alg = FixedPointFilter()`, an asset survives when its share of missing observations does not exceed `col_thr` over the observations that the apply step keeps on the training window. Under `alg = SinglePassFilter()`, the share is read over every observation of the training window. The apply step replays that universe by name on a later window, in the fitted order, and refuses a window that lacks a fitted asset. The train weights and the test returns therefore name the same assets in the same order. The row filter reads each window alone. The apply step drops an observation when its share of missing assets, over the fitted universe, exceeds `row_thr`.

This estimator is the only missing-data filter of the library. [`prices_to_returns`](@ref) removes no observation and no asset, because a deletion is a **Universe Policy**. A policy needs a fit on a training window and a replay by name on later windows, and a stateless conversion has no fit to replay. The factor, benchmark and implied volatility series, and the Asset Panel, follow the surviving assets and observations, because the [`PricesResult`](@ref) states one clock.

The filter counts a gap in every series that a fit reads, as the fit reads it. A gap in the series of one asset is a missing observation of that asset: a gap in `X`, in its implied volatility, or in its own column of a per-asset benchmark. A gap in a series that every asset shares, a one-column benchmark or a factor, is a missing observation of every asset at that observation, because a fit drops that observation, see [`shared_series_view`](@ref). [`asset_gap_cells`](@ref) and [`shared_gap_rows`](@ref) find the two kinds. A series that holds no entry at a timestamp of `X` has a gap there.

A share is the count of missing entries divided by the count of entries, computed in the type of the threshold. `29 / 100` is `0.29` in `Float64` and `29f0 / 100f0` is `0.29f0` in `Float32`, so a threshold written as a decimal keeps an asset or an observation whose share equals it. A `Rational` threshold compares the exact share.

# Algorithm

## Fit

 1. Find the surviving assets with [`fit_universe`](@ref) under `alg`. It counts the missing observations of each asset column, the cells of [`asset_gap_cells`](@ref), and keeps the assets whose share does not exceed `col_thr`. [`FixedPointFilter`](@ref) reads the shares again over the observations that the row filter keeps, until no asset drops. Raise an `IsEmptyError` when no asset survives, or when the row filter of a [`FixedPointFilter`](@ref) keeps no observation.
 2. Return a [`MissingDataFilterResult`](@ref) that holds the names of the surviving assets, in the column order of the training window, and `row_thr`.

## Apply

 1. Find the column of each fitted asset in the window, in the fitted order. Raise an `ArgumentError` that names the first absent asset.
 2. Count the missing assets of each observation over those columns, the cells of [`asset_gap_cells`](@ref). An observation of [`shared_gap_rows`](@ref) misses every asset. Keep the observations whose share of missing assets does not exceed `row_thr`. Raise an `IsEmptyError` when no observation survives.
 3. Return [`port_opt_view`](@ref) of the `PricesResult` at the kept timestamps and the fitted columns. It reads the factor, benchmark and implied volatility series at the kept timestamps, subsets a per-asset benchmark, the implied volatilities and a vector `ivpa` to the fitted columns, and views the Asset Panel and the Listing Span at the kept rows and columns.

The two thresholds count opposite axes. `col_thr` counts the missing observations of an asset and drops assets. `row_thr` counts the missing assets of an observation and drops observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
        alg::AbstractMissingDataFilterAlgorithm = FixedPointFilter(),
        cache::Option{<:AbstractPartialFitState} = nothing,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields. Both thresholds admit zero. `col_thr = 0` keeps the assets with no missing observation, and `row_thr = 0` keeps the observations with no missing asset. Both thresholds admit one, which drops nothing on its axis.

## Validation

  - `0 <= col_thr <= 1`, else `DomainError`.
  - `0 <= row_thr <= 1`, else `DomainError`.

# Online form

The column filter changes only the universe. A stepped filter counts the missing observations of every block in `cache` through [`partial_fit_transform`](@ref) and passes the rows through. [`fit_preprocessing`](@ref) with no data reads the [`MissingDataFilterResult`](@ref) of the whole history out of that count, and a [`Pipeline`](@ref) applies that universe as a view in `fit(pipe)` with no data. The row filter has an online form only at `row_thr = 1`, where it drops nothing. At a lower threshold, the filter can drop a row at a later step that it kept at an earlier step, when an asset leaves the universe. [`supports_partial_fit`](@ref) answers `false` for it, and a Pipeline refuses it at warm-up with a message that names the step. With no data, the filter reads the column shares one time, over every folded observation. At `row_thr = 1` that is the fixed point, so `alg` does not change the online form.

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

`C` misses 4 of 10 observations, which `col_thr = 0.4` admits. The row filter drops the first three observations, and over the seven that stay `C` misses 4. The default rule drops `C`. The single pass keeps it.

```jldoctest
julia> M = [NaN NaN 70; NaN NaN 71; NaN NaN 72; 100 50 NaN; 101 51 NaN; 102 52 NaN; 103 53 NaN;
            104 54 77; 105 55 78; 106 56 79];

julia> pr = PricesResult(;
                         X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 10), M,
                                       [\"A\", \"B\", \"C\"]));

julia> fit_preprocessing(MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3), pr).nx
2-element Vector{Symbol}:
 :A
 :B

julia> fit_preprocessing(MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3,
                                           alg = SinglePassFilter()), pr).nx
3-element Vector{Symbol}:
 :A
 :B
 :C
```

# Related

  - [`AbstractMissingDataFilterAlgorithm`](@ref)
  - [`FixedPointFilter`](@ref)
  - [`SinglePassFilter`](@ref)
  - [`MissingDataFilterResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`PriceGapFill`](@ref)
  - [`PricesResult`](@ref)
"""
@concrete struct MissingDataFilter <: AbstractPricesPreprocessingEstimator
    """
    Largest share `[0, 1]` of missing observations that an asset can have over the training window and stay in the universe. `alg` decides which observations of the training window the share counts. A gap in its implied volatility or in its own column of a per-asset benchmark is a missing observation. `0` keeps the assets with a price at every observation of the training window.
    """
    col_thr
    """
    Largest share `[0, 1]` of missing assets that an observation can have and stay in the window. A gap in a one-column benchmark or in a factor misses every asset. `0` keeps the observations at which every asset of the fitted universe has a price.
    """
    row_thr
    """
    Rule that decides how the fit reads the column shares, a [`FixedPointFilter`](@ref) or a [`SinglePassFilter`](@ref).
    """
    alg
    """
    $(field_dict[:pfcache])
    """
    cache
    function MissingDataFilter(col_thr::Number, row_thr::Number,
                               alg::AbstractMissingDataFilterAlgorithm,
                               cache::Option{<:AbstractPartialFitState})
        assert_closed_unit_interval(col_thr, :col_thr)
        assert_closed_unit_interval(row_thr, :row_thr)
        return new{typeof(col_thr), typeof(row_thr), typeof(alg), typeof(cache)}(col_thr,
                                                                                 row_thr,
                                                                                 alg, cache)
    end
end
function MissingDataFilter(; col_thr::Number = 1.0, row_thr::Number = 1.0,
                           alg::AbstractMissingDataFilterAlgorithm = FixedPointFilter(),
                           cache::Option{<:AbstractPartialFitState} = nothing)::MissingDataFilter
    return MissingDataFilter(col_thr, row_thr, alg, cache)
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
"""
    column_pass(cells::AbstractMatrix{Bool}, rows::Union{Colon, VecInt}, cols::VecInt,
                col_thr::Number) -> Vector{Int}

Keep the assets of `cols` whose share of missing observations over the observations `rows` does not exceed `col_thr`.

# Arguments

  - `cells`: The missing observations of each asset, from [`asset_gap_cells`](@ref).
  - `rows`: The observations that the share counts, or `Colon()` for every observation.
  - `cols`: The positions of the assets to test.
  - `col_thr`: Largest share that passes.

# Validation

  - One asset at least survives. An `IsEmptyError` is thrown otherwise.

# Returns

  - `kept::Vector{Int}`: The positions in `cols` of the kept assets, in the order of `cols`.

# Related

  - [`fit_universe`](@ref)
  - [`share_at_most`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function column_pass(cells::AbstractMatrix{Bool}, rows::Union{Colon, VecInt}, cols::VecInt,
                     col_thr::Number)::Vector{Int}
    sub = view(cells, rows, cols)
    kept = cols[share_at_most.(vec(count(sub; dims = 1)), size(sub, 1), col_thr)]
    @argcheck(!isempty(kept),
              IsEmptyError("MissingDataFilter with col_thr = $(col_thr) drops every asset in the training window"))
    return kept
end
"""
    kept_rows(cells::AbstractMatrix{Bool}, shared::AbstractVector{Bool}, cols::VecInt,
              row_thr::Number, nx::AbstractVector) -> Vector{Int}

Find the observations whose share of missing assets over the assets `cols` does not exceed `row_thr`.

An observation of `shared` has a gap in a series that every asset shares, so it misses every asset. The fit of a [`FixedPointFilter`](@ref) and the apply step of a [`MissingDataFilter`](@ref) both call this function, so the two find the same observations.

# Arguments

  - `cells`: The missing observations of each asset, from [`asset_gap_cells`](@ref).
  - `shared`: The observations with a gap in a shared series, from [`shared_gap_rows`](@ref).
  - `cols`: The positions of the assets that the share counts.
  - `row_thr`: Largest share that passes.
  - `nx`: The names of every asset of the window, which the error message reads.

# Validation

  - One observation at least survives. An `IsEmptyError` from [`empty_window_msg`](@ref) is thrown otherwise.

# Returns

  - `rows::Vector{Int}`: The positions of the kept observations, in order.

# Related

  - [`column_pass`](@ref)
  - [`share_at_most`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function kept_rows(cells::AbstractMatrix{Bool}, shared::AbstractVector{Bool}, cols::VecInt,
                   row_thr::Number, nx::AbstractVector)::Vector{Int}
    sub = view(cells, :, cols)
    miss = vec(count(sub; dims = 2))
    miss[shared] .= length(cols)
    rows = findall(share_at_most.(miss, length(cols), row_thr))
    @argcheck(!isempty(rows),
              IsEmptyError(empty_window_msg(sub, shared, nx[cols], row_thr)))
    return rows
end
"""
    empty_window_msg(sub::AbstractMatrix{Bool}, shared::AbstractVector{Bool},
                     nx::AbstractVector, row_thr::Number) -> String

Write the message of a row filter that keeps no observation, and name the cause when the window shows it.

A row filter keeps no observation for one of two causes that no threshold of the filter can repair. A shared series with a gap at every observation misses every asset at every observation, and no filter drops a series. An asset with no observation misses every observation, and `col_thr = 1` keeps it. The message names the first cause that the window shows.

# Arguments

  - `sub`: The missing observations of the kept assets.
  - `shared`: The observations with a gap in a shared series.
  - `nx`: The names of the kept assets.
  - `row_thr`: The threshold of the row filter.

# Returns

  - `msg::String`: The message.

# Related

  - [`kept_rows`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function empty_window_msg(sub::AbstractMatrix{Bool}, shared::AbstractVector{Bool},
                          nx::AbstractVector, row_thr::Number)::String
    msg = "MissingDataFilter with row_thr = $(row_thr) drops every observation of the window"
    empty = nx[vec(all(sub; dims = 1))]
    if !isempty(shared) && all(shared)
        msg *= ". A factor or a one-column benchmark has a gap at every observation, and no filter drops a series. Fill the series, or remove it from the input"
    elseif !isempty(empty)
        msg *= ". The assets $(join(empty, ", ")) have no observation in the window, so every observation misses them. A fit with col_thr below 1 drops an asset with no observation"
    end
    return msg * "."
end
"""
    fit_universe(alg::SinglePassFilter, mdf::MissingDataFilter, pr::PricesResult) -> Vector{Int}
    fit_universe(alg::FixedPointFilter, mdf::MissingDataFilter, pr::PricesResult) -> Vector{Int}

Find the column positions in `pr.X` of the assets that the fit of `mdf` keeps, under the rule `alg`.

[`SinglePassFilter`](@ref) keeps the assets whose share of missing observations over the whole training window does not exceed `col_thr`. [`FixedPointFilter`](@ref) then reads the shares again over the observations that the row filter keeps, and loops until no asset drops. Its `# Algorithm` section states the loop.

# Arguments

  - `alg`: The rule.
  - `mdf`: The filter, which holds `col_thr` and `row_thr`.
  - `pr`: The training window.

# Validation

  - One asset at least survives each column pass. An `IsEmptyError` is thrown otherwise, from [`column_pass`](@ref).
  - Under [`FixedPointFilter`](@ref), the row filter keeps one observation at least. An `IsEmptyError` is thrown otherwise, from [`kept_rows`](@ref).

# Returns

  - `cols::Vector{Int}`: The positions of the kept assets, in the column order of `pr.X`.

# Related

  - [`AbstractMissingDataFilterAlgorithm`](@ref)
  - [`column_pass`](@ref)
  - [`kept_rows`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function fit_universe(::SinglePassFilter, mdf::MissingDataFilter,
                      pr::PricesResult)::Vector{Int}
    cells = asset_gap_cells(pr)
    return column_pass(cells, :, axes(cells, 2), mdf.col_thr)
end
function fit_universe(::FixedPointFilter, mdf::MissingDataFilter,
                      pr::PricesResult)::Vector{Int}
    cells = asset_gap_cells(pr)
    shared = shared_gap_rows(pr)
    nx = TimeSeries.colnames(pr.X)
    cols = column_pass(cells, :, axes(cells, 2), mdf.col_thr)
    while true
        #! The rows come from the whole window on each pass, as the apply step finds them,
        #! so the fixed point holds on the rows that the apply step keeps.
        rows = kept_rows(cells, shared, cols, mdf.row_thr, nx)
        kept = column_pass(cells, rows, cols, mdf.col_thr)
        if length(kept) == length(cols)
            return cols
        end
        cols = kept
    end
end
function fit_preprocessing(mdf::MissingDataFilter,
                           pr::PricesResult)::MissingDataFilterResult
    return MissingDataFilterResult(TimeSeries.colnames(pr.X)[fit_universe(mdf.alg, mdf, pr)],
                                   mdf.row_thr)
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
    rows = kept_rows(asset_gap_cells(pr), shared_gap_rows(pr), cols, res.row_thr, names)
    #! The PricesResult states one clock, so every series it holds is read at the surviving
    #! timestamps and the fitted columns, by the view that owns that rule.
    return port_opt_view(pr, TimeSeries.timestamp(pr.X)[rows], cols)
end
"""
    clock_gap_cells(A::TimeSeries.TimeArray, ts::AbstractVector) -> BitMatrix

Mark the gaps of a series on the clock `ts`, one row for each timestamp of `ts` and one column for each column of `A`.

A cell is a gap when `A` holds no entry at the timestamp, or when its entry is missing by [`is_missing_value`](@ref). The factor, benchmark and implied volatility series of a [`PricesResult`](@ref) keep their own timestamps, so the function pairs them with the asset clock by timestamp and not by position.

# Arguments

  - `A`: The series.
  - `ts`: The clock, the timestamps of the asset prices.

# Returns

  - `m::BitMatrix`: `true` at each gap.

# Related

  - [`asset_gap_cells`](@ref)
  - [`shared_gap_rows`](@ref)
  - [`optional_gap_cells`](@ref)
"""
function clock_gap_cells(A::TimeSeries.TimeArray, ts::AbstractVector)::BitMatrix
    tA = TimeSeries.timestamp(A)
    pos = Dict(zip(tA, eachindex(tA)))
    idx = get.(Ref(pos), ts, 0)
    found = .!iszero.(idx)
    vals = values(A)
    m = trues(length(ts), size(vals, 2))
    m[found, :] .= is_missing_value.(view(vals, idx[found], :))
    return m
end
"""
    optional_gap_cells(A::Nothing, ts::AbstractVector, n::Integer) -> BitMatrix
    optional_gap_cells(A::TimeSeries.TimeArray, ts::AbstractVector, n::Integer) -> BitMatrix

Mark the gaps of an optional series on the clock `ts`. An absent series has no gap, so it gives `n` columns of `false`.

# Related

  - [`clock_gap_cells`](@ref)
  - [`asset_gap_cells`](@ref)
"""
function optional_gap_cells(::Nothing, ts::AbstractVector, n::Integer)::BitMatrix
    return falses(length(ts), n)
end
function optional_gap_cells(A::TimeSeries.TimeArray, ts::AbstractVector,
                            ::Integer)::BitMatrix
    return clock_gap_cells(A, ts)
end
"""
    benchmark_columns(B::Nothing, shared::Bool) -> Nothing
    benchmark_columns(B::TimeSeries.TimeArray, shared::Bool) -> Option{TimeSeries.TimeArray}

Return the benchmark when it is of the kind that `shared` names, and `nothing` otherwise. A one-column benchmark is shared by every asset, and a benchmark of one column per asset is per asset.

# Related

  - [`asset_gap_cells`](@ref)
  - [`shared_gap_rows`](@ref)
"""
function benchmark_columns(::Nothing, ::Bool)
    return nothing
end
function benchmark_columns(B::TimeSeries.TimeArray, shared::Bool)
    return ifelse(isone(size(values(B), 2)) == shared, B, nothing)
end
"""
    asset_gap_cells(pr::PricesResult) -> BitMatrix

Mark the missing observations of each asset of `pr`, `observations × assets` on the clock of the asset prices.

An asset misses an observation when its price is missing, when its implied volatility is missing, or when its own column of a per-asset benchmark is missing. A one-column benchmark is shared by every asset, so [`shared_gap_rows`](@ref) reads it instead.

# Arguments

  - `pr`: The prices.

# Returns

  - `m::BitMatrix`: `true` at each missing observation of an asset.

# Related

  - [`MissingDataFilter`](@ref)
  - [`clock_gap_cells`](@ref)
"""
function asset_gap_cells(pr::PricesResult)::BitMatrix
    ts = TimeSeries.timestamp(pr.X)
    vals = values(pr.X)
    n = size(vals, 2)
    return BitMatrix(is_missing_value.(vals)) .| optional_gap_cells(pr.iv, ts, n) .|
           optional_gap_cells(benchmark_columns(pr.B, false), ts, n)
end
"""
    shared_gap_rows(pr::PricesResult) -> BitVector

Mark the observations of `pr` at which a series that every asset shares has a gap: a one-column benchmark, or any factor.

A fit that reads such a series drops the observation, see [`shared_series_view`](@ref), so [`MissingDataFilter`](@ref) counts it as an observation that misses every asset.

# Arguments

  - `pr`: The prices.

# Returns

  - `r::BitVector`: `true` at each observation with a gap in a shared series.

# Related

  - [`MissingDataFilter`](@ref)
  - [`clock_gap_cells`](@ref)
"""
function shared_gap_rows(pr::PricesResult)::BitVector
    ts = TimeSeries.timestamp(pr.X)
    return vec(any(optional_gap_cells(benchmark_columns(pr.B, true), ts, 1); dims = 2)) .|
           vec(any(optional_gap_cells(pr.F, ts, 1); dims = 2))
end

export MissingDataFilter, MissingDataFilterResult, SinglePassFilter, FixedPointFilter
public AbstractMissingDataFilterAlgorithm, fit_universe
