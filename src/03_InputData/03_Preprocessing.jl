"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all returns result types.

All concrete and/or types representing the result of returns calculations should be subtypes of `AbstractReturnsResult`.

## The asset-selector contract

[`select_assets`](@ref) and [`fit_preprocessing`](@ref) dispatch on this supertype, so any subtype reaching an [`AbstractAssetSelector`](@ref) must carry `nx` and an `observations × assets` matrix `X`, plus a [`port_opt_view`](@ref) that replays a selected universe. [`ClusterGroups`](@ref) widens that to `{nx, X, Z}`: it reads the feature matrix `Z` straight off the carrier, because preselection runs before any prior exists and no other source is reachable.

Widening the contract rather than the `Pr_RR` bridge is deliberate — that alias's concreteness is load-bearing at nine routing sites. The cost is that the contract is implicit: it is satisfied by [`ReturnsResult`](@ref) and enforced by nothing. [`PredictionReturnsResult`](@ref) subtypes this supertype, but its `X` is a *portfolio* return vector rather than an asset matrix — the asset axis is already collapsed away — so it satisfies neither the old contract nor the widened one, and every entry point refuses it loudly rather than measuring the wrong axis.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`select_assets`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all price-level data result types.

All concrete types representing price-level data should be subtypes of `AbstractPricesResult`. Defined alongside [`AbstractReturnsResult`](@ref) so cross-validation splitting, preprocessing, and prediction can dispatch on either data level.

# Related

  - [`AbstractResult`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that asset or factor names and their corresponding returns matrix are provided and consistent.

# Arguments

  - `names`: Asset or factor names.
  - `mat`: Returns matrix.
  - `names_sym`: Symbolic name for the names argument displayed in error messages.
  - `mat_sym`: Symbolic name for the matrix argument displayed in error messages.

# Validation

  - `allunique(names)`, whenever `names` is not `nothing`.

  - If either `names` or `mat` is not `nothing`:

      + `!isnothing(names)` and `!isnothing(mat)`.
      + `!isempty(names)` and `!isempty(mat)`.
      + `length(names) == size(mat, 2)`.

# Returns

  - `nothing`.

# Related

  - [`ReturnsResult`](@ref)
"""
function check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
                                        names_sym::Symbol, mat_sym::Symbol)
    if !isnothing(names)
        @argcheck(allunique(names),
                  ArgumentError("$names_sym names must be unique. Got\nallunique($names_sym) => $(allunique(names))"))
    end
    if !(isnothing(names) && isnothing(mat))
        @argcheck(!isnothing(names),
                  IsNothingError("$names_sym cannot be nothing if $mat_sym is not `nothing`. Got\n!isnothing($names_sym) => $(!isnothing(names))\n!isnothing($mat_sym) => $(!isnothing(mat))"))
        @argcheck(!isnothing(mat),
                  IsNothingError("$mat_sym cannot be nothing if $names_sym is not `nothing`. Got\n!isnothing($names_sym) => $(!isnothing(names))\n!isnothing($mat_sym) => $(!isnothing(mat))"))
        @argcheck(!isempty(names), IsEmptyError("$names_sym cannot be empty."))
        @argcheck(!isempty(mat), IsEmptyError("$mat_sym cannot be empty."))
        @argcheck(length(names) == size(mat, 2),
                  DimensionMismatch("length($names_sym) == size($mat_sym, 2) must hold. Got\nlength($names_sym) => $(length(names))\nsize($mat_sym, 2) => $(size(mat, 2))"))
    end
    return nothing
end
"""
    feature_row_indices(pnl::Nothing, ts_new, ts_old) -> Colon
    feature_row_indices(pnl::AssetPanel, ts_new, ts_old) -> Union{Colon, VecInt}

Recover the positional row indices of a time-varying [`AssetPanel`](@ref) from a timestamp window.

A Panel Field holds a plain array, so its observation axis is parallel to the carrier's clock positionally rather than aligned by timestamp. Whenever a routine selects rows of `X` by timestamp, the surviving timestamps are matched back into the original clock to recover the rows the panel must keep. A surviving timestamp absent from that clock throws: it means the row bookkeeping has been broken (a synthesised timestamp, or an outer join that introduced a row `X` never had), and slicing the panel positionally from there would silently pair each asset with another period's values.

Two sites use it. At **price level** the clock is `TimeSeries.timestamp(X)` and the selection is a timestamp window. At the **cross-validation assembly seam** the clock is `ReturnsResult.ts` and the selection is a fold: [`fold_row_indices`](@ref) recovers a fold's rows from the timestamps its view of the returns already carries, which is why `ts` must be unique — it *keys* the observation axis rather than merely labelling it.

The static and absent shapes have no observation axis, so they return `Colon` and cost nothing.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`, or static: return `Colon()`. Neither has an observation axis, so there is no row to recover and the timestamps are not read.
 2. `pnl` is time-varying: match `ts_new` into `ts_old` with [`matched_row_indices`](@ref), which throws when the selection kept no timestamp, or when a surviving timestamp is absent from the original clock.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `ts_new`: Timestamps surviving the selection.
  - `ts_old`: Timestamps of the clock the panel's observation axis is parallel to.

# Validation

  - `ts_new` is not `nothing` when the panel is time-varying.
  - Every entry of `ts_new` appears in `ts_old`.

# Returns

  - `Colon` for a static or absent panel; otherwise the row indices, as a `Vector{Int}`.

# Related

  - [`AssetPanel`](@ref)
  - [`matched_row_indices`](@ref)
  - [`PricesResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`fold_row_indices`](@ref)
"""
function feature_row_indices(::Nothing, ::Any, ::Any)
    return Colon()
end
function feature_row_indices(pnl::AssetPanel, ts_new, ts_old)
    return panel_is_static(pnl) ? Colon() : matched_row_indices(ts_new, ts_old)
end
"""
    matched_row_indices(ts_new::Nothing, ts_old) -> Union{}
    matched_row_indices(ts_new, ts_old) -> Vector{Int}

Match the surviving timestamps back into the original clock, and return the rows they hold.

The body [`feature_row_indices`](@ref) shares between the time-varying feature matrix and the time-varying [`AssetPanel`](@ref): both hold their observation axis parallel to the carrier's clock positionally, so both recover their rows the same way.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ts_new` is `nothing`: throw. The selection kept no timestamp, so the rows to keep cannot be named.
 2. Otherwise match `ts_new` into `ts_old` with `indexin`, check that every surviving timestamp was found, and return the positions as a `Vector{Int}`.

# Arguments

  - `ts_new`: Timestamps that survived the selection.
  - `ts_old`: Timestamps of the clock the observation axis is parallel to.

# Validation

  - `ts_new` is not `nothing`. Raises an `ArgumentError`.
  - Every entry of `ts_new` appears in `ts_old`. Raises an `ArgumentError`.

# Returns

  - `rows::Vector{Int}`: The position each surviving timestamp holds in the original clock.

# Related

  - [`feature_row_indices`](@ref)
  - [`AssetPanel`](@ref)
  - [`prices_to_returns`](@ref)
"""
function matched_row_indices(::Nothing, ::Any)
    return throw(ArgumentError("a time-varying feature axis has its observation axis parallel to the price timestamps, but no timestamps survived the conversion, so the rows to keep cannot be recovered. Pass a static Asset Panel instead."))
end
function matched_row_indices(ts_new, ts_old)
    rows = indexin(ts_new, ts_old)
    missed = findfirst(isnothing, rows)
    @argcheck(isnothing(missed),
              ArgumentError("a time-varying feature axis has its observation axis parallel to the price timestamps, but the timestamp $(ts_new[missed]) selected here is absent from them, so the row it corresponds to cannot be recovered. This happens when the surviving timestamps are not a subset of the original clock — a `collapse_args` timestamp function that synthesises timestamps, or an outer join that introduced rows the asset prices never had. Pass a static Asset Panel, or align the feature axis to the price clock first."))
    return Vector{Int}(rows)
end
"""
    panel_feature_names(pnl::Nothing) -> nothing
    panel_feature_names(pnl::AssetPanel) -> Vector{String}

Name the columns an [`AssetPanel`](@ref) derives, without building the Feature Matrix.

A consumer that needs the column names alone reads them here, and the values are not stacked to answer it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): walk its Panel Fields, appending each one's value column names and then its observed-mask column names.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.

# Returns

  - `nz::Option{Vector{String}}`: One name per derived column, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_feature_matrix`](@ref)
"""
function panel_feature_names(::Nothing)
    return nothing
end
function panel_feature_names(pnl::AssetPanel)
    nz = String[]
    for f in pnl.pf
        append!(nz, panel_field_labels(f))
        if !isnothing(f.omsk)
            append!(nz, panel_field_observed_labels(f))
        end
    end
    return nz
end
"""
    panel_carrier_view(pnl::Nothing, i, j, nx) -> nothing
    panel_carrier_view(pnl::AssetPanel, i, j, nx) -> AssetPanel

View a carrier's [`AssetPanel`](@ref), or return `nothing` when the carrier holds none.

The one-line wrapper every carrier view goes through, so the `nothing` case is written once rather than at each of the six sites that slice a panel.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): return [`port_opt_view`](@ref) of it.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `i`: Observation index.
  - `j`: Asset index.
  - `sq`: Whether the Panel Fields are the assets.

# Returns

  - An Asset Panel over the selected observations and assets, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
  - [`PricesResult`](@ref)
"""
function panel_carrier_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function panel_carrier_view(pnl::AssetPanel, i, j, nx::Option{<:VecStr})
    return port_opt_view(pnl, i, j, nx)
end
"""
$(DocStringExtensions.TYPEDEF)

A container for aligned, time-indexed price-level data.

`PricesResult` is the prices-level mirror of [`ReturnsResult`](@ref): it bundles asset prices with optional factor, benchmark, and implied volatility series, all as `TimeSeries.TimeArray`s. It is the input to price-level preprocessing estimators and prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline cross-validation via [`port_opt_view`](@ref).

The asset price series `X` is the master clock: [`port_opt_view`](@ref) selects observation windows on `X` and aligns the other series to the selected timestamps.

The feature matrix `Z` is the exception to that alignment. It is a plain array, not a `TimeArray` — `TimeSeries.jl` has no 3-dimensional `TimeArray`, and the static shape has no clock at all — so it cannot be aligned by timestamp, only indexed positionally. Its axes are therefore held *parallel* to `X`: the asset axis to `TimeSeries.colnames(X)`, and, for the time-varying shape, the observation axis to `TimeSeries.timestamp(X)` row for row. Every routine that drops an asset or an observation from `X` must drop it from `Z` in the same step, which is what [`port_opt_view`](@ref), [`MissingDataFilter`](@ref) and [`prices_to_returns`](@ref) do.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesResult(;
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing,
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
        span::Option{<:AbstractMatrix{Bool}} = nothing,
    ) -> PricesResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(X)`.
  - If `F` is not `nothing`: `!isempty(F)`.
  - If `B` is not `nothing`: `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`: `!isempty(iv)`, `all(x -> x >= 0, values(iv))`, `all(x -> isfinite(x), values(iv))`, and `size(values(iv), 2) == size(values(X), 2)`.
  - If `ivpa` is not `nothing`: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(values(X), 2)`.
  - `pnl`'s asset axis is `size(values(X), 2)`, and its observation axis is `size(values(X), 1)` when it is time-varying. See [`check_asset_panel`](@ref).
  - If `span` is not `nothing`: `size(span) == size(values(X))`. Raises a `DimensionMismatch`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> size(values(pr.X))
(3, 2)
```

# Related

  - [`AbstractPricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`Num_VecNum`](@ref)
  - [`MatNum_Arr3Num`](@ref)
  - [`check_asset_panel`](@ref)
"""
@concrete struct PricesResult <: AbstractPricesResult
    """
    Asset price data (observations × assets). The master clock for timestamp-window slicing.
    """
    X
    """
    Optional factor price data (observations × factors).
    """
    F
    """
    Optional benchmark price data (observations × 1) or (observations × assets).
    """
    B
    """
    Optional implied volatility data (observations × assets).
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    Optional [`AssetPanel`](@ref): the Panel Fields of the universe, and its two universe masks. Not a `TimeArray`: its axes are held positionally parallel to `X`.
    """
    pnl
    """
    Optional **Listing Span**: which assets are listed at each observation of the price clock, `observations × assets`, held positionally parallel to `X`. A [`PortfolioOptimisers.ListingSpan`](@ref) when [`PriceIngestion`](@ref) derived it by the Span Rule, and any other `AbstractMatrix{Bool}` when a caller declared their own listing calendar. `nothing` says the carrier was not built by the ingestion layer.
    """
    span
    function PricesResult(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray},
                          B::Option{<:TimeSeries.TimeArray},
                          iv::Option{<:TimeSeries.TimeArray}, ivpa::Option{<:Num_VecNum},
                          pnl::Option{<:AssetPanel}, span::Option{<:AbstractMatrix{Bool}})
        @argcheck(!isempty(X), IsEmptyError)
        if !isnothing(F)
            @argcheck(!isempty(F), IsEmptyError)
        end
        if !isnothing(B)
            @argcheck(!isempty(B), IsEmptyError)
            @argcheck(size(values(B), 2) in (1, size(values(X), 2)), DimensionMismatch)
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(values(iv), :iv)
            @argcheck(size(values(iv), 2) == size(values(X), 2), DimensionMismatch)
        end
        if !isnothing(ivpa)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(values(X), 2), DimensionMismatch)
            end
        end
        check_asset_panel(pnl, size(values(X), 2), size(values(X), 1), "size(values(X), 2)")
        if !isnothing(span)
            @argcheck(size(span) == size(values(X)),
                      DimensionMismatch("a Listing Span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices; got size(span) = $(size(span)) and size(values(X)) = $(size(values(X)))"))
        end
        return new{typeof(X), typeof(F), typeof(B), typeof(iv), typeof(ivpa), typeof(pnl),
                   typeof(span)}(X, F, B, iv, ivpa, pnl, span)
    end
end
function PricesResult(; X::TimeSeries.TimeArray,
                      F::Option{<:TimeSeries.TimeArray} = nothing,
                      B::Option{<:TimeSeries.TimeArray} = nothing,
                      iv::Option{<:TimeSeries.TimeArray} = nothing,
                      ivpa::Option{<:Num_VecNum} = nothing,
                      pnl::Option{<:AssetPanel} = nothing,
                      span::Option{<:AbstractMatrix{Bool}} = nothing)::PricesResult
    return PricesResult(X, F, B, iv, ivpa, pnl, span)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `PricesResult` for the observation window `i` and the assets `j` of the asset price series `X`.

The asset price series is the master clock: `i` selects rows of `X`, and the factor, benchmark, and implied volatility series are aligned to the selected timestamps (rows whose timestamps are absent from a series are dropped from that series). `j` selects asset columns and defaults to `:`, so a call giving only `i` is an observation window over the whole universe.

# Algorithm

The method that Julia selects is the algorithm. The timestamp methods do the work, and the integer method routes into them.

 1. `i` and `j` are both `Colon`: return `pr` itself. No view is built.

 2. `i` is a vector of timestamps and `j` is a `Colon`: index `X`, `F`, `B` and `iv` by the timestamps `i`. Recover the rows of a time-varying Asset Panel from the surviving timestamps with [`feature_row_indices`](@ref), and view the panel on that observation axis with [`panel_carrier_view`](@ref). A static panel has no observation axis and ignores the row index. View the Listing Span on the same surviving timestamps with [`span_carrier_view`](@ref). Carry `ivpa` through untouched, because the asset index does not reach it. Rebuild the [`PricesResult`](@ref).

 3. `i` is a vector of timestamps and `j` is a vector of asset indices:

     1. Index `X` by the timestamps `i`, then keep the asset columns `j`.
     2. Index `F` by the timestamps `i` alone. `j` is an asset index, and the factors are a separate axis, so every factor column is kept.
     3. Index `B` by the timestamps `i`. Keep its columns `j` when `B` holds one column per asset, and keep its single column otherwise. The test is `B`'s own width, because a shared benchmark has one column to give whatever `j` asks for.
     4. Index `iv` by the timestamps `i` and the asset columns `j`, and view `ivpa` at `j`.
     5. Read `sq` from [`features_are_assets`](@ref) on `nz` and the asset names of `X`. When `sq` is `true`, view `nz` at `j` as well.
     6. Recover the rows of a time-varying Asset Panel with [`feature_row_indices`](@ref), and view the panel at those rows and the assets `j` with [`panel_carrier_view`](@ref), handing it the asset names so that a square tensor Panel Field is cut on its label axis too.
     7. View the Listing Span at the surviving timestamps and the assets `j` with [`span_carrier_view`](@ref).
     8. Rebuild the [`PricesResult`](@ref).

 4. `i` and `j` are integer indices, ranges or `Colon`s: read the timestamps `TimeSeries.timestamp(pr.X)[i]`, and call step 2 or step 3 with them. This is the method a caller reaches with `port_opt_view(pr, 2:3)`.

# Arguments

  - `pr`: A `PricesResult` object.
  - `i`: Observation window into the rows of `pr.X`. Either integer indices (`AbstractVector{<:Integer}`, `AbstractRange`, or `Colon`) or a vector of timestamps (`AbstractVector{<:Dates.AbstractTime}`).
  - `j`: Asset window into the columns of `pr.X`. Integer indices, an `AbstractRange`, or `Colon` for the whole universe. A `Colon` leaves `X`, `B`, `iv` and `ivpa` alone, which is why `ivpa` passes through untouched on the observation-only arity and is viewed at `j` on the other.

# Returns

  - `new_pr::PricesResult`: A new `PricesResult` containing only the data for the selected window.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> pv = PortfolioOptimisers.port_opt_view(pr, 2:3);

julia> first(timestamp(pv.X))
2020-01-02

julia> size(values(pv.X))
(2, 2)
```

# Related

  - [`PricesResult`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::PricesResult, ::Colon, ::Colon)
    return pr
end
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},
                       ::Colon = :)
    X = pr.X[i]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    B = isnothing(pr.B) ? nothing : pr.B[i]
    iv = isnothing(pr.iv) ? nothing : pr.iv[i]
    rows = feature_row_indices(pr.pnl, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X))
    pnl = panel_carrier_view(pr.pnl, rows, :, nothing)
    span = span_carrier_view(pr.span, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X),
                             :)
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = pr.ivpa, pnl = pnl,
                        span = span)
end
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},
                       j::AbstractVector)
    X = pr.X[i][TimeSeries.colnames(pr.X)[j]]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    #! A benchmark is either one column per asset or a single shared column
    #! (the PricesResult constructor admits no other width). Only the first is
    #! indexed by the asset index; slicing the second by `j` reads past its one
    #! column. The test is B's own width, never `length(j)`.
    B = if isnothing(pr.B)
        nothing
    elseif length(TimeSeries.colnames(pr.B)) == size(values(pr.X), 2)
        pr.B[i][TimeSeries.colnames(pr.B)[j]]
    else
        pr.B[i]
    end
    iv = isnothing(pr.iv) ? nothing : pr.iv[i][TimeSeries.colnames(pr.iv)[j]]
    ivpa = nothing_scalar_array_view(pr.ivpa, j)
    rows = feature_row_indices(pr.pnl, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X))
    pnl = panel_carrier_view(pr.pnl, rows, j, string.(TimeSeries.colnames(pr.X)))
    span = span_carrier_view(pr.span, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X),
                             j)
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = ivpa, pnl = pnl, span = span)
end
function port_opt_view(pr::PricesResult,
                       i::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :,
                       j::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :)
    return port_opt_view(pr, TimeSeries.timestamp(pr.X)[i], j)
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the results of asset and factor returns calculations.

`ReturnsResult` is the standard result type returned by returns-processing routines, such as [`prices_to_returns`](@ref).

It supports both asset and factor returns, as well as optional time series and implied volatility information, and is designed for downstream compatibility with optimisation and analysis routines.

It also carries the optional feature matrix `Z` that [`FeatureDistance`](@ref) turns into a distance. `Z` is *data*, not configuration, which is why it is held here rather than on the estimator: the clustering stack is asset-subset-blind by construction, so an estimator-held feature matrix would survive a nested-clustered subproblem or a cross-validation fold unsliced, with its asset axis silently pointing at the full universe. `ReturnsResult` implements [`port_opt_view`](@ref), so a carried `Z` is subselected in step with `X`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsResult(;
        nx::Option{<:VecStr} = nothing,
        X::Option{<:MatNum} = nothing,
        nf::Option{<:VecStr} = nothing,
        F::Option{<:MatNum} = nothing,
        nb::Option{<:VecStr} = nothing,
        B::Option{<:VecNum_MatNum} = nothing,
        ts::Option{<:VecDate} = nothing,
        iv::Option{<:MatNum} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
    ) -> ReturnsResult

Keywords correspond to the struct's fields.

## Validation

  - If `nx` or `X` is not `nothing`, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is not `nothing`, `!isempty(nf)`, `!isempty(F)`, `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `nb` or `B` is not `nothing` and `B` is a matrix: `!isempty(nb)`, `!isempty(B)`, and `length(nb) == size(B, 2)`.
  - If `nb` or `B` is not `nothing` and `B` is a vector: `length(nb) == 1`.
  - If `X` and `B` are not `nothing`: if `B` is a vector, `size(X, 1) == size(B, 1)`; if `B` is a matrix, `size(X) == size(B)`.
  - If `ts` is not `nothing`, `!isempty(ts)`, `allunique(ts)`, and `length(ts) == size(X, 1)`. Uniqueness is required because `ts` *keys* the observation axis rather than merely labelling it: [`feature_row_indices`](@ref) recovers a subset's rows by matching its surviving timestamps back into this clock, and a repeated timestamp would resolve to the first occurrence and pair an asset with another period's features.
  - If `ts` and `B` are not `nothing`: `length(ts) == size(B, 1)`.
  - If `iv` is not `nothing`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `all(x -> isfinite(x), iv)`, and `size(iv) == size(X)`.
  - `ivpa` is validated in that same branch, so it is checked only when `iv` is given: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`, and, if a vector, `length(ivpa) == size(iv, 2)`. The bound is strict — a zero adjustment is rejected. An `ivpa` passed without an `iv` reaches no check, because it has no implied volatility to adjust.
  - `pnl`'s asset axis is `length(nx)`, and its observation axis is `size(X, 1)` when it is time-varying. See [`check_asset_panel`](@ref).

# Examples

```jldoctest
julia> ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`AssetPanel`](@ref)
  - [`asset_panel`](@ref)
  - [`check_asset_panel`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
  - [`VecDate`](@ref)
  - [`Num_VecNum`](@ref)
"""
@concrete struct ReturnsResult <: AbstractReturnsResult
    """
    Names or identifiers of asset columns (assets × 1).
    """
    nx
    """
    Asset returns matrix (observations × assets).
    """
    X
    """
    Names or identifiers of factor columns (factors × 1).
    """
    nf
    """
    Factor returns matrix (observations × factors).
    """
    F
    """
    Names or identifiers of benchmark columns (observations × 1) or (observations × assets).
    """
    nb
    """
    Benchmark prices (observations × 1) or (observations × assets).
    """
    B
    """
    Optional timestamps for each observation (observations × 1).
    """
    ts
    """
    Implied volatilities matrix (observations × assets).
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    Optional [`AssetPanel`](@ref): the Panel Fields of the universe, and its two universe masks.
    """
    pnl
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, nb::Option{<:VecStr},
                           B::Option{<:VecNum_MatNum}, ts::Option{<:VecDate},
                           iv::Option{<:MatNum}, ivpa::Option{<:Num_VecNum},
                           pnl::Option{<:AssetPanel})
        check_names_and_returns_matrix(nx, X, :nx, :X)
        check_names_and_returns_matrix(nf, F, :nf, :F)
        if isa(B, VecNum) && !isnothing(nb)
            @argcheck(length(nb) == 1,
                      DimensionMismatch("a single-column benchmark (B) admits exactly one benchmark name (nb), got length(nb) = $(length(nb))"))
        elseif isa(B, MatNum)
            check_names_and_returns_matrix(nb, B, :nb, :B)
        end
        if !isnothing(X) && !isnothing(F)
            @argcheck(size(X, 1) == size(F, 1),
                      DimensionMismatch("asset returns (X) and factor returns (F) must share the same number of observations (rows), got size(X, 1) = $(size(X, 1)) and size(F, 1) = $(size(F, 1))"))
        end
        if !isnothing(X) && !isnothing(B)
            if isa(B, VecNum)
                @argcheck(size(X, 1) == size(B, 1),
                          DimensionMismatch("benchmark returns (B) must match asset returns (X) in number of observations (rows), got size(X, 1) = $(size(X, 1)) and size(B, 1) = $(size(B, 1))"))
            else
                @argcheck(size(X) == size(B),
                          DimensionMismatch("benchmark returns (B) must match asset returns (X) in size, got size(X) = $(size(X)) and size(B) = $(size(B))"))
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            # `ts` is an *index* into the observation axis, not merely a label on it: a
            # subset's surviving timestamps are matched back into it to recover the rows a
            # time-varying feature matrix must keep (see `feature_row_indices`). A repeated
            # timestamp makes that recovery pick the first occurrence and silently pair an
            # asset with another period's features, so the axis must be uniquely keyed.
            @argcheck(allunique(ts),
                      ArgumentError("timestamps (ts) must be unique — they key the observation axis, and a repeated timestamp makes a row unrecoverable by time. Got $(length(ts) - length(unique(ts))) duplicate(s), the first being $(ts[findfirst(i -> ts[i] in view(ts, 1:(i - 1)), eachindex(ts))])"))
            if !isnothing(X)
                @argcheck(length(ts) == size(X, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per asset-returns (X) observation (row), got length(ts) = $(length(ts)) and size(X, 1) = $(size(X, 1))"))
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per factor-returns (F) observation (row), got length(ts) = $(length(ts)) and size(F, 1) = $(size(F, 1))"))
            end
            if !isnothing(B)
                @argcheck(length(ts) == size(B, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per benchmark-returns (B) observation (row), got length(ts) = $(length(ts)) and size(B, 1) = $(size(B, 1))"))
            end
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X),
                      DimensionMismatch("implied volatilities (iv) must match asset returns (X) in size, got size(iv) = $(size(iv)) and size(X) = $(size(X))"))
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(iv, 2),
                          DimensionMismatch("the implied-volatility risk-premium adjustment (ivpa), when a vector, must have one entry per asset (implied-volatility column), got length(ivpa) = $(length(ivpa)) and size(iv, 2) = $(size(iv, 2))"))
            end
        end
        check_asset_panel(pnl, isnothing(nx) ? nothing : length(nx),
                          isnothing(X) ? nothing : size(X, 1), "length(nx)")
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ts), typeof(iv), typeof(ivpa), typeof(pnl)}(nx, X, nf, F, nb, B,
                                                                      ts, iv, ivpa, pnl)
    end
end
function ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                       nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                       nb::Option{<:VecStr} = nothing, B::Option{<:VecNum_MatNum} = nothing,
                       ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                       ivpa::Option{<:Num_VecNum} = nothing,
                       pnl::Option{<:AssetPanel} = nothing)::ReturnsResult
    return ReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa, pnl)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `ReturnsResult` object for the assets at indices `i`.

This is the [`port_opt_view`](@ref) method for [`ReturnsResult`](@ref) — the View of the library's central data structure, restricting it to a subset of assets.

!!! warning

    This two-argument method indexes **assets**, matching the rest of the `port_opt_view` family. The four-argument method `port_opt_view(rd, i, j, k)` indexes **observations** first and assets second. The two arities therefore give `i` different meanings; see [`port_opt_view(rd::ReturnsResult, i, j, k)`](@ref).

# Algorithm

 1. View the asset names `nx` at `i` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, :, i)`. Axis 2 is the assets, and every observation is kept.
 3. When `B` is a matrix, it holds one column per asset: view `nb` at `i`, and view `B` as `view(rd.B, :, i)`. Otherwise — a single shared benchmark, or none at all — `nb` and `B` both pass through untouched.
 4. View the implied volatilities as `view(rd.iv, :, i)`, and the adjustment `ivpa` at `i`.
 5. Read `sq` from [`features_are_assets`](@ref) on `nz` and `nx`. When `sq` is `true`, view `nz` at `i` as well, because the feature axis is the asset axis.
 6. View the Asset Panel with [`panel_carrier_view`](@ref) at `i` on the asset axis, handing it the asset names. The observation index is a `Colon`, so a time-varying panel keeps every observation.
 7. View the [`AssetPanel`](@ref) `pnl` at `i` on the asset axis, which slices every Panel Field's values and both universe masks. A Panel Field's label axis is not touched: it addresses the features, which an asset view does not reach.
 8. Rebuild the [`ReturnsResult`](@ref). The factor names `nf`, the factor returns `F` and the timestamps `ts` pass through untouched, because none of the three has an asset axis.

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Indices of the assets to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified index.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> PortfolioOptimisers.port_opt_view(rd, 2:2)
ReturnsResult
    nx ┼ SubArray{String, 1, Vector{String}, Tuple{UnitRange{Int64}}, true}: ["B"]
     X ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

* * *

    port_opt_view(
        rd::ReturnsResult,
        i,
        j,
        k = :
    ) -> ReturnsResult

Return a view of the `ReturnsResult` object for assets at indices `j`, observations at indices `i`, and factors at indices `k`.

!!! warning

    Unlike every other [`port_opt_view`](@ref) method — including [`port_opt_view(rd::ReturnsResult, i)`](@ref) — the first index of this method selects **observations**, not assets. Assets are the *second* index. Cross-validation splits observations and assets together, which is why this arity exists at all.

# Algorithm

 1. View the asset names `nx` at `j` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, i, j)`. Axis 1 is the observations, and axis 2 is the assets.
 3. View the factor names `nf` at `k`, unless `k` is a `Colon`, in which case `nf` passes through. View the factor returns as `view(rd.F, i, k)`.
 4. When `B` is a matrix, it holds one column per asset: view `nb` at `j`, and view `B` as `view(rd.B, i, j)`. When `B` is a vector, it is a single shared benchmark: view it as `view(rd.B, i)`, and carry `nb` through.
 5. View the timestamps `ts` at `i`, the implied volatilities as `view(rd.iv, i, j)`, and the adjustment `ivpa` at `j`.
 6. Read `sq` from [`features_are_assets`](@ref) on `nz` and `nx`. When `sq` is `true`, view `nz` at `j` as well.
 7. View the Asset Panel with [`panel_carrier_view`](@ref) at the observations `i` and the assets `j`, handing it the asset names. A static panel has no observation axis and ignores `i`, which is the same asymmetry `ivpa` has on the asset axis.
 8. View the [`AssetPanel`](@ref) `pnl` at the observations `i` and the assets `j`, which slices both axes of every Panel Field and of both universe masks.
 9. Rebuild the [`ReturnsResult`](@ref).

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified indices.

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4; 0.5 0.6], nf = [\"F1\"],
                          F = [1.0; 2.0; 3.0;;])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 3×2 Matrix{Float64}
    nf ┼ Vector{String}: ["F1"]
     F ┼ 3×1 Matrix{Float64}
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> PortfolioOptimisers.port_opt_view(rd, 1:2, 2:2)
ReturnsResult
    nx ┼ SubArray{String, 1, Vector{String}, Tuple{UnitRange{Int64}}, true}: ["B"]
     X ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    nf ┼ Vector{String}: ["F1"]
     F ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

* * *

    port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)

Erroring tripwire for [`AbstractReturnsResult`](@ref) subtypes that do not implement [`port_opt_view`](@ref).

Without it, the universal leaf fallback `port_opt_view(x, i, args...)` would hand back the returns result *unsubselected*, and a meta-optimiser or cross-validation fold would silently train on the full universe. Returns data is never a leaf value, so an unhandled subtype is a missing method, not a pass-through.

Subtypes carrying an [`AssetPanel`](@ref) owe it the same treatment as `X`: subselect its asset axis on every arity, its observation axis on the arities that take one, and — when a tensor Panel Field's labels *are* the assets ([`features_are_assets`](@ref)) — that field's label axis as well. A panel that survives a fold unsliced is the same silent-wrongness as an unsliced returns matrix, one level down: the distance it produces is finite, plausible, and computed over the wrong universe. [`port_opt_view`](@ref) implements the rule; the [`ReturnsResult`](@ref) methods are the reference.

# Algorithm

 1. Throw an `ArgumentError` naming the concrete type and the number of index arguments the call gave. The method reads neither the indices nor the fields of `rd`.

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractReturnsResult`](@ref)

* * *

    port_opt_view(rd::ReturnsResult, args...; kwargs...)

Erroring tripwire for [`ReturnsResult`](@ref) calls whose *call shape* no supported arity matches.

`ReturnsResult` does implement [`port_opt_view`](@ref), so the [`AbstractReturnsResult`](@ref) tripwire above would misreport a mistyped call as an unimplemented subtype. This method takes the call instead and names the call shape: the supported arities take one, two, or three positional index arguments and no keyword arguments — in particular `factors` is the third *positional* index, not a keyword.

# Algorithm

 1. Count the positional arguments `args`, and read the names of the keyword arguments `kwargs`.
 2. Throw an `ArgumentError` reporting both counts, and naming the three supported call shapes.

# Related

  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
"""
function port_opt_view(rd::ReturnsResult, i)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, i)
    B = !isa(rd.B, MatNum) ? rd.B : view(rd.B, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    pnl = panel_carrier_view(rd.pnl, :, i, rd.nx)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, nb = nb, B = B, ts = rd.ts,
                         iv = iv, ivpa = ivpa, pnl = pnl)
end
function port_opt_view(rd::ReturnsResult, i, j, k = :)
    nx = nothing_scalar_array_view(rd.nx, j)
    X = isnothing(rd.X) ? rd.X : view(rd.X, i, j)
    nf = isnothing(rd.nf) || isa(k, Colon) ? rd.nf : view(rd.nf, k)
    F = isnothing(rd.F) ? rd.F : view(rd.F, i, k)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, j)
    B = if isnothing(rd.B)
        nothing
    elseif isa(rd.B, VecNum)
        view(rd.B, i)
    else
        view(rd.B, i, j)
    end
    ts = isnothing(rd.ts) ? rd.ts : view(rd.ts, i)
    iv = isnothing(rd.iv) ? rd.iv : view(rd.iv, i, j)
    ivpa = nothing_scalar_array_view(rd.ivpa, j)
    pnl = panel_carrier_view(rd.pnl, i, j, rd.nx)
    return ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, ts = ts, iv = iv,
                         ivpa = ivpa, pnl = pnl)
end
function port_opt_view(rd::ReturnsResult, args...; kwargs...)
    kws = keys(kwargs)
    kwmsg = isempty(kws) ? "" : " and keyword argument(s) " * join(kws, ", ")
    return throw(ArgumentError("port_opt_view(::ReturnsResult, ...) does not accept this call shape; got $(length(args)) positional index argument(s)$(kwmsg). Supported shapes: port_opt_view(rd, assets) to subselect assets; port_opt_view(rd, observations, assets) or port_opt_view(rd, observations, assets, factors) to subselect observations and assets together (note the reversed index order, and that `factors` is the third positional index, not a keyword)."))
end
function port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)
    return throw(ArgumentError("$(typeof(rd)) subtypes AbstractReturnsResult but does not implement port_opt_view for $(length(args)) index argument(s). Extension authors: implement port_opt_view for the subtype; without it a meta-optimiser or cross-validation fold would silently train on the unsubselected universe. See port_opt_view(rd::ReturnsResult, ...) for the reference implementation."))
end
"""
    const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}

Union of the two data levels cross-validation folds can be computed on: returns-level ([`AbstractReturnsResult`](@ref)) and price-level ([`AbstractPricesResult`](@ref)) data.

Fold generation only needs an observation count ([`cv_nobs`](@ref)) and a timestamp vector ([`cv_timestamps`](@ref)), so [`Base.split`](@ref) and [`n_splits`](@ref) accept either level. Price-level splitting is what lets a `Pipeline` be cross-validated on its *input* rows, keeping stateful preprocessing inside the fold.

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`AbstractPricesResult`](@ref)
  - [`cv_nobs`](@ref)
  - [`cv_timestamps`](@ref)
  - [`n_splits`](@ref)
"""
const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one side of a train/test split into a row count.

A size is either an `Integer` count of observations, or an `AbstractFloat` fraction of them in `(0, 1)`. Counts saturate at `N` (asking for more rows than exist takes all of them); the [`safe_index`](@ref) window guards then reject a split that leaves either side empty.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and the two mean different things: a count and a fraction.

 1. `s` is an `Integer`, so it is a count of rows: check that `s > 0`, and return `min(Int(s), N)`. A count larger than the data takes every row.
 2. `s` is an `AbstractFloat`, so it is a fraction of the rows: check that `0 < s < 1`, and return `clamp(floor(Int, s * N), 1, N)`. The fraction rounds **down** to whole rows, and the clamp keeps a small fraction of a short window from resolving to zero rows.

# Arguments

  - `s`: One side of the split, as a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`).
  - `N`: Number of observations available.
  - `name`: Symbolic name of the side, displayed in error messages.

# Validation

  - `s > 0` when `s` is an `Integer`.
  - `0 < s < 1` when `s` is an `AbstractFloat`.

# Returns

  - `n::Int`: The number of rows the side takes, in `1:N`.

# Related

  - [`safe_index`](@ref)
  - [`TrainTestSplit`](@ref)
"""
function split_count(s::Integer, N::Integer, name::Symbol)::Int
    @argcheck(s > zero(s),
              DomainError(s, "the $name of a train/test split must be > 0, got $s"))
    return min(Int(s), N)
end
function split_count(s::AbstractFloat, N::Integer, name::Symbol)::Int
    @argcheck(zero(s) < s < one(s),
              DomainError(s,
                          "the $name of a train/test split must lie in (0, 1) when given as a fraction, got $s"))
    return clamp(floor(Int, s * N), 1, N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the `(train, test)` observation ranges of a holdout split over `N` time-ordered rows.

Training rows come from the head of the data and test rows from the tail, so the test window is always the most recent one. Each size is a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`), resolved by [`split_count`](@ref).

  - **Neither given**: the split falls at `D` (75 % train, 25 % test).
  - **One given**: the other side is its complement, so the two windows partition the data.
  - **Both given**: the head supplies `lo` training rows, the tail supplies `hi` test rows, and any rows between them are **embargoed** — they belong to neither window. This is how a gap between train and test is expressed. The gap is declared by the two sizes and nothing else; a rule that derives one from the label horizon belongs to the purged cross-validators ([`CombinatorialCrossValidation`](@ref)), not here.

# Algorithm

 1. Resolve the two window lengths `N_l` and `N_h`, through the branch that `lo` and `hi` select:

     1. Neither is given: take `n = clamp(floor(Int, D * N), 1, N)`, then `N_l = n` and `N_h = N - n`.
     2. Only `lo` is given: resolve it with [`split_count`](@ref), then `N_l = n` and `N_h = N - n`.
     3. Only `hi` is given: resolve it with [`split_count`](@ref), then `N_l = N - n` and `N_h = n`.
     4. Both are given: resolve each with [`split_count`](@ref) on its own. Neither is the complement of the other, so the rows between the two windows are embargoed.

 2. Check that both windows are non-empty, and that the two do not overlap.

 3. Return the two ranges `1:N_l` and `(N - N_h + 1):N`. The training window is the head of the data, and the test window is the tail, so the embargoed rows sit between them.

# Arguments

  - `lo`: Training rows, as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `hi`.
  - `hi`: Test rows, likewise; `nothing` takes the complement of `lo`.
  - `N`: Number of observations available.
  - `D = 0.75`: Training fraction taken when neither size is given.

# Validation

  - Both windows are non-empty. A split whose sizes saturate the data on one side (`train_size = N`) leaves nothing to test on and throws.
  - The windows do not overlap: `lo + hi <= N`.

# Returns

  - `(train, test)`: The training and test row ranges, as two `UnitRange{Int}`s.

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`split_count`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function safe_index(lo::Option{<:Number}, hi::Option{<:Number}, N::Integer, D = 0.75)
    N_l, N_h = if isnothing(lo) && isnothing(hi)
        n = clamp(floor(Int, D * N), 1, N)
        n, N - n
    elseif isnothing(hi)
        n = split_count(lo, N, :train_size)
        n, N - n
    elseif isnothing(lo)
        n = split_count(hi, N, :test_size)
        N - n, n
    else
        split_count(lo, N, :train_size), split_count(hi, N, :test_size)
    end
    @argcheck(N_l > 0 && N_h > 0,
              ArgumentError("a train/test split of $N observations must leave both windows non-empty, got $N_l training and $N_h test observations"))
    @argcheck(N_l + N_h <= N,
              ArgumentError("the training and test windows of a train/test split must not overlap, but $N_l training and $N_h test observations exceed the $N available; rows between the two windows are embargoed, so their sizes may sum to less than $N but never to more"))
    return 1:N_l, (N - N_h + 1):N
end
"""
    train_test_split(rd::ReturnsResult; train_size, test_size) -> (train, test)
    train_test_split(pr::PricesResult; train_size, test_size) -> (train, test)

Cut price- or returns-level data into a training window (the head) and a held-out test window (the tail).

The free-function form of [`TrainTestSplit`](@ref); the windows are [`port_opt_view`](@ref)s, so no data is copied. See [`safe_index`](@ref) for the sizing rules — complement when one side is given, embargo when both are.

# Algorithm

 1. Read the observation count `N` from the asset data: `size(rd.X, 1)` at returns level, and `size(TimeSeries.values(pr.X), 1)` at price level.
 2. Resolve the two row ranges with [`safe_index`](@ref).
 3. Return a [`port_opt_view`](@ref) of each range. The returns-level method passes a `Colon` asset index after the row range, because that arity indexes observations first and assets second.

# Arguments

  - `rd`/`pr`: The data to split.
  - `train_size`: Training rows as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
  - `test_size`: Test rows, likewise; `nothing` takes the complement of `train_size`. With neither given the split is 75/25.

# Returns

  - `(train, test)`: The two windows, of the same type as the input.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\"], X = reshape(collect(0.1:0.1:1.0), 10, 1));

julia> train, test = train_test_split(rd; test_size = 0.2);

julia> size(train.X, 1), size(test.X, 1)
(8, 2)
```

# Related

  - [`TrainTestSplit`](@ref)
  - [`safe_index`](@ref)
"""
function train_test_split(rd::ReturnsResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(rd.X, 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train, :), port_opt_view(rd, test, :)
end
function train_test_split(rd::PricesResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(TimeSeries.values(rd.X), 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train), port_opt_view(rd, test)
end
"""
    returns_result_picker(rd::ReturnsResult, brt::Bool) -> ReturnsResult

Return a `ReturnsResult` appropriate for benchmark-tracking optimisations.

This helper inspects the `ReturnsResult`'s benchmark field `B` and the boolean flag `brt` (benchmark-tracking). If `brt` is `true` and a benchmark `B` is present it returns a new `ReturnsResult` in which asset returns `X` have the benchmark removed (i.e. `X - B` or broadcast `X .- B` for vector benchmarks). If `brt` is `false` or no benchmark is present, the original `ReturnsResult` is returned unchanged.

# Algorithm

The first step is a method selected on the field type of `B`, so a carrier with no benchmark runs no branch at all.

 1. `rd` carries no benchmark, because its `B` field is `Nothing`: return `rd` itself.
 2. `brt` is `false`: return `rd` itself.
 3. `brt` is `true`: subtract the benchmark from the asset returns, giving `X`. A vector benchmark subtracts by broadcast, `rd.X .- rd.B`, which takes one benchmark value per observation from every asset column. A matrix benchmark subtracts elementwise, `rd.X - rd.B`.
 4. Rebuild the [`ReturnsResult`](@ref) from `X`, and leave `nb` and `B` unset. The benchmark is spent on the subtraction, which is what makes a second call return its argument unchanged. Every other field — `nx`, `nf`, `F`, `ts`, `iv`, `ivpa` and `pnl` — is carried over. The argument itself is never modified.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset, factor and/or benchmark returns.
  - `brt`: Boolean flag indicating whether benchmark-tracking behaviour should be applied. When `true`, asset returns are adjusted by subtracting the benchmark `B` (if present).

# Returns

  - `rd::ReturnsResult`:

      + If `brt` is `true` and a benchmark `B` is present: A new `ReturnsResult` with adjusted asset returns
      + Otherwise: The `rd` is returned unchanged. `nb` and `B` hold `nothing` on an adjusted result, which is what makes the adjustment idempotent.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.10 0.20; 0.30 0.40], nb = [\"BM\"],
                          B = [0.01; 0.02])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ Vector{String}: ["BM"]
     B ┼ Vector{Float64}: [0.01, 0.02]
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd2 = returns_result_picker(rd, false)  # no change when brt is false
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ Vector{String}: ["BM"]
     B ┼ Vector{Float64}: [0.01, 0.02]
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd === rd2
true

julia> rd3 = returns_result_picker(rd, true)
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd.X .- rd.B == rd3.X
true
```

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
"""
function returns_result_picker(rd::ReturnsResult{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                 Nothing}, ::Any)
    return rd
end
function returns_result_picker(rd::ReturnsResult{<:Any, <:MatNum, <:Any, <:Any, <:Any,
                                                 <:VecNum_MatNum}, brt::Bool)
    return if !brt
        rd
    else
        X = isa(rd.B, VecNum) ? rd.X .- rd.B : rd.X - rd.B
        ReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts, iv = rd.iv,
                      ivpa = rd.ivpa, pnl = rd.pnl)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Supertype of the policies that write a return into the cells a price gap left non-finite.

A return is the change between two consecutive observations, so a run of `k` gapped prices leaves `k + 1` non-finite returns and the move across the gap is recorded nowhere. That is the default, and it is what `nothing` means on [`prices_to_returns`](@ref). A caller who wants the move booked states one of these instead.

An algorithm may write **only** a non-finite cell inside the asset's Listing Span that has an earlier observed price in its column; [`gap_return_writable`](@ref) derives that set and [`apply_gap_return`](@ref) restores every other cell. So a cell computed from two observed prices is frozen whatever the algorithm returns, a gap can never spread beyond the cells that read one of its prices, and no return is invented before an asset's first price.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractGapReturnAlgorithm` and implement the following methods:

## `gap_return`

  - `gap_return(alg::AbstractGapReturnAlgorithm, p::AbstractVector, r::AbstractVector, ret_method::Symbol) -> Vector`: One column's returns, with the writable cells resolved.

### Arguments

  - `alg`: The concrete subtype instance.
  - `p`: One column's prices along the observation axis, gaps included.
  - `r`: The returns `TimeSeries.percentchange` computed from `p`, so `length(p) - length(r)` is `0` under `padding` and `1` otherwise.
  - `ret_method`: `:simple` or `:log`. Compute a value with [`gap_return_value`](@ref) rather than re-spelling the two branches.

### Returns

  - `out::Vector`: The same length as `r`. Only the cells [`gap_return_writable`](@ref) admits are read back, so a method may return the frozen cells unchanged and need not defend the invariant itself.

# Related

  - [`CatchUpGapReturn`](@ref)
  - [`gap_return`](@ref)
  - [`gap_return_writable`](@ref)
  - [`apply_gap_return`](@ref)
  - [`prices_to_returns`](@ref)
"""
abstract type AbstractGapReturnAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Books a Held Gap's whole move on the observation that ends it, shortening the gap to `k`.

A suspension of `k` observations leaves `k + 1` non-finite returns by default. This puts ``P_{t+k} / P_{t-1} - 1`` on the observation the asset resumes trading and leaves the `k` observations inside the gap non-finite, so wealth is conserved across the gap and the Held Gap is exactly the run of unpriced observations. An asset's inception is untouched: it has no earlier observed price to anchor on.

The cost is stated by ADR 0131. The estimation mask reads the values it was given and is unaware of which algorithm produced them, so the re-pricing cell is estimable and a `(k + 1)`-period return enters a one-period moment as one draw, at roughly ``\\sqrt{k + 1}`` the scale.

# Constructors

    CatchUpGapReturn() -> CatchUpGapReturn

# Examples

```jldoctest
julia> CatchUpGapReturn()
CatchUpGapReturn()
```

# Related

  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`gap_return`](@ref)
  - [`prices_to_returns`](@ref)
"""
struct CatchUpGapReturn <: AbstractGapReturnAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute one return from a pair of prices that need not be consecutive.

The one place the `ret_method` branches are spelled for the Gap Return family, so a new algorithm states which pair of prices it reads and never which formula turns them into a return. It mirrors `TimeSeries.percentchange`, which computes both branches through logarithms, so a value written here sits on the same arithmetic as the cells around it.

# Arguments

  - `ret_method`: `:simple` or `:log`.
  - `pt`: The later price.
  - `p0`: The earlier price, the return's anchor.

# Returns

  - `r::Number`: ``\\ln P_t - \\ln P_0`` under `:log`, and `expm1` of it otherwise.

# Related

  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`gap_return`](@ref)
"""
function gap_return_value(ret_method::Symbol, pt::Number, p0::Number)
    lr = log(pt) - log(p0)
    return ret_method === :log ? lr : expm1(lr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the cells of one column a Gap Return algorithm is allowed to write.

This is the family's invariant, held once rather than re-argued per algorithm. [`apply_gap_return`](@ref) restores every cell outside the returned set, so no algorithm can rewrite a return computed from two observed prices, manufacture one before an asset's first price, or resurrect a delisting.

The bounds are the Span Rule and its projection, the same ones [`listing_span`](@ref) and [`PortfolioOptimisers.project_span`](@ref) state for a whole panel. They are read here off the one price column the conversion is holding, because the writable set is per column and the table reaching [`prices_to_returns`](@ref)'s conversion step is the filtered one rather than the caller's.

# Algorithm

 1. Read the offset between the two clocks as `length(p) - length(r)`, which is `0` when `padding` kept the first observation and `1` when it did not. Return cell `j` is then the change onto price row `j + off`.
 2. Locate the column's Listing Span on the price clock: the first observed price and the last. A column with no observed price admits nothing.
 3. Admit return cell `j` when its price row lies in `[first + 1, last]` — the span projected onto the returns clock, since a return consumes the earlier price of its pair — and the default rule left the cell non-finite.

# Arguments

  - `p`: One column's prices along the observation axis, gaps included.
  - `r`: The returns `TimeSeries.percentchange` computed from `p`.

# Returns

  - `w::BitVector`: The same length as `r`, true on the cells an algorithm may write.

# Related

  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`apply_gap_return`](@ref)
  - [`listing_span`](@ref)
  - [`PortfolioOptimisers.project_span`](@ref)
  - [`is_missing_value`](@ref)
"""
function gap_return_writable(p::AbstractVector, r::AbstractVector)::BitVector
    w = falses(length(r))
    observed = .!is_missing_value.(p)
    i1 = findfirst(observed)
    if isnothing(i1)
        return w
    end
    i2 = findlast(observed)
    off = length(p) - length(r)
    for j in eachindex(r)
        t = j + off
        w[j] = i1 < t <= i2 && !isfinite(r[j])
    end
    return w
end
"""
    gap_return(alg::CatchUpGapReturn, p::AbstractVector, r::AbstractVector, ret_method::Symbol) -> Vector

Resolve the writable cells of one column's returns.

The method Julia selects is the algorithm. Only [`CatchUpGapReturn`](@ref) ships, and it is the reason the family is an algorithm rather than a flag: a caller who wants a suspension's move spread across its observations is asking a question of the same kind, and it costs one type.

# Algorithm

[`CatchUpGapReturn`](@ref) walks the observation axis carrying the row of the last observed price.

 1. On a gapped price, carry nothing forward and write nothing: the observation is inside the Held Gap and stays non-finite.
 2. On an observed price whose immediate predecessor was observed too, write nothing: `TimeSeries.percentchange` already computed that cell from two consecutive prices, and [`gap_return_writable`](@ref) freezes it in any case.
 3. On an observed price whose immediate predecessor was not, write [`gap_return_value`](@ref) of it against the carried price. This is the observation that ends the gap, and the whole move across the gap lands on it.

A column's first observed price carries nothing, so nothing is written on it, which is what makes an inception and an interior gap one case.

# Arguments

  - `alg`: The Gap Return algorithm.
  - `p`: One column's prices along the observation axis, gaps included.
  - `r`: The returns `TimeSeries.percentchange` computed from `p`.
  - `ret_method`: `:simple` or `:log`.

# Returns

  - `out::Vector`: The same length as `r`, with the writable cells resolved.

# Examples

```jldoctest
julia> PortfolioOptimisers.gap_return(CatchUpGapReturn(), [100.0, NaN, NaN, 110.0],
                                      [NaN, NaN, NaN], :simple)
3-element Vector{Float64}:
 NaN
 NaN
   0.0999999999999999
```

# Related

  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`CatchUpGapReturn`](@ref)
  - [`gap_return_writable`](@ref)
  - [`gap_return_value`](@ref)
  - [`apply_gap_return`](@ref)
"""
function gap_return(::CatchUpGapReturn, p::AbstractVector, r::AbstractVector,
                    ret_method::Symbol)
    out = collect(r)
    off = length(p) - length(r)
    anchor = 0
    for t in eachindex(p)
        if is_missing_value(p[t])
            continue
        end
        if anchor != 0 && anchor != t - 1
            out[t - off] = gap_return_value(ret_method, p[t], p[anchor])
        end
        anchor = t
    end
    return out
end
"""
    apply_gap_return(alg::Nothing, R::DataFrames.DataFrame, P::DataFrames.DataFrame, ret_method::Symbol) -> DataFrames.DataFrame
    apply_gap_return(alg::AbstractGapReturnAlgorithm, R::DataFrames.DataFrame, P::DataFrames.DataFrame, ret_method::Symbol) -> DataFrames.DataFrame

Apply the `gap_return_alg` given to [`prices_to_returns`](@ref) to the converted table.

The seam that keeps the family optional and holds its invariant. `nothing` is the default path, and its method returns the table untouched, so the arithmetic `TimeSeries.percentchange` produced is bit-identical to what it was before the family existed.

The rule is per-column arithmetic on consecutive observations and reads no asset axis, so it applies to every series of the converted table alike — asset, factor and benchmark.

# Algorithm

 1. Walk the series columns of `R`, taking each column's prices from `P` by name.
 2. Derive the writable cells with [`gap_return_writable`](@ref).
 3. Call [`gap_return`](@ref) on the column and copy back **only** the writable cells, so every other cell is frozen whatever the algorithm returned.
 4. Report an `@info` when no column admitted a single cell. A table that holds no gap admits none, which is the ordinary case rather than a mistake, so this is neither a refusal, which would reject a configuration that computes a correct answer, nor a warning, which could not tell that case from one where the caller expected a gap.

# Arguments

  - `alg`: The Gap Return algorithm, or `nothing` for the default rule.
  - `R`: The converted table, `:timestamp` first and one column per series.
  - `P`: The price table reaching the conversion, with the same series columns.
  - `ret_method`: `:simple` or `:log`.

# Returns

  - `R::DataFrames.DataFrame`: The converted table, with the writable cells resolved.

# Related

  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`CatchUpGapReturn`](@ref)
  - [`gap_return`](@ref)
  - [`gap_return_writable`](@ref)
  - [`prices_to_returns`](@ref)
"""
function apply_gap_return(::Nothing, R::DataFrames.DataFrame, ::DataFrames.DataFrame,
                          ::Symbol)
    return R
end
function apply_gap_return(alg::AbstractGapReturnAlgorithm, R::DataFrames.DataFrame,
                          P::DataFrames.DataFrame, ret_method::Symbol)
    wrote = false
    for nm in names(R)[2:end]
        r = R[!, nm]
        p = P[!, nm]
        w = gap_return_writable(p, r)
        if !any(w)
            continue
        end
        wrote = true
        out = gap_return(alg, p, r, ret_method)
        @argcheck(length(out) == length(r), DimensionMismatch)
        r[w] .= out[w]
    end
    if !wrote
        @info("`gap_return_alg` is a $(typeof(alg)) and no cell is writable, so the returns are the ones the default rule computed. A Gap Return writes only a non-finite return inside an asset's Listing Span that has an earlier observed price in its column, and the table reaching the conversion holds no such cell.")
    end
    return R
end
"""
    prices_to_returns(
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing;
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        ret_method::Symbol = :simple, padding::Bool = false,
        gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
        missing_col_percent::Number = 1.0,
        missing_row_percent::Option{<:Number} = 1.0,
        collapse_args::Tuple = (),
        map_func::Option{<:Function} = nothing,
        join_method::Symbol = :outer,
        pnl::Option{<:AssetPanel} = nothing,
        span::Option{<:AbstractMatrix{Bool}} = nothing
    ) -> ReturnsResult

Convert `TimeSeries.TimeArray` price data to returns. Handles factor data, a price gap, and
optional implied volatility information.

An absent price has one spelling, `NaN`, and the conversion carries it into the returns rather
than deleting the observation or the asset that holds one. Filling a gap is
[`PriceGapFill`](@ref)'s and deleting one is [`MissingDataFilter`](@ref)'s, both of them fitted
steps; ADR 0133 owns the rule.

# Mathematical definition

Returns are computed from prices ``P_{t,i}`` as:

```math
\\begin{align}
r_{t,i} &= \\begin{cases}
(P_{t,i} - P_{t-1,i}) / P_{t-1,i} & \\text{simple} \\\\
\\ln(P_{t,i} / P_{t-1,i}) & \\text{log}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``r_{t,i}``: Return of asset ``i`` at time ``t``.
  - ``P_{t,i}``: Price of asset ``i`` at time ``t``.

Both branches need a **positive** price, and a zero price gives ``\\pm\\infty``.

A benchmark ``B`` is converted by the same rule and **carried alongside** the asset returns, never subtracted from them. The subtraction that forms the excess return ``\\tilde{r}_{t,i} = r_{t,i} - b_{t,i}`` is a separate operation, and it is applied only when the optimisation tracks the benchmark.

# Algorithm

 1. Check `X`, `missing_col_percent` and `missing_row_percent`. Read the asset names and the asset timestamps from `X`, and check `pnl` against them with [`check_asset_panel`](@ref).
 2. Merge the factor prices `F` into `X` under `join_method`, and record the factor names.
 3. Merge the benchmark prices `B` into `X` under `join_method`, and record the benchmark names. A benchmark is one shared column, or one column per asset.
 4. Apply `map_func` to every entry, when one is given.
 5. Collapse the time series with `collapse_args`, when they are given. This is the step that changes the frequency.
 6. Convert the table to a `DataFrames.DataFrame`.
 7. Replace every `missing` with `NaN`, so that the two conventions a source spells an absent price with become one and no later step deletes it: an outer join of per-asset series pads with `NaN`, and a wide table built from a tidy one leaves `missing`. This is the only unification, it runs unconditionally, and it is what makes the two ragged-history sources behave alike.
 8. Count the gapped entries of each row of the table with [`is_missing_value`](@ref), which reads both conventions because it stands at the door.
 9. Drop each row whose count of missing columns exceeds `missing_col_percent` of the column total.
10. Count the missing entries of each column over the rows that step 9 kept. Drop each column whose count exceeds `missing_row_percent` of the surviving row total. When `missing_row_percent` is `nothing`, keep instead the columns whose count equals the mode of the counts.
11. Convert the surviving prices to returns with `TimeSeries.percentchange` under `ret_method` and `padding`. This is the step that applies the formula above. It computes both branches through logarithms — the log return is ``\\ln P_{t,i} - \\ln P_{t-1,i}``, and the simple return is `expm1` of it — so the two agree with the closed forms above to floating point rather than to the last bit. When `padding` is `true` the first observation is kept and its return is `NaN`, so the returns keep the length of the price clock. **A gap carried here does not spread.** The formula reads two prices, so a run of `k` gapped prices makes exactly the `k + 1` returns that read one of them non-finite, and every later return of that column is computed from two observed prices and is finite. A gap is confined to its own column for the same reason: no asset's return reads another's price.
12. Resolve the cells the conversion left non-finite with [`apply_gap_return`](@ref), under `gap_return_alg`. `nothing` is the default rule, and its method returns the table untouched, so the arithmetic step 11 produced is bit-identical. An algorithm may write only a non-finite cell inside a column's Listing Span that has an earlier observed price, which is what freezes every return computed from two observed prices, and it reports an `@info` when it finds no such cell.
13. Split the surviving column names into the asset names `nx`, the factor names `nf`, the benchmark names `nb`, and the timestamp column, which gives `ts`.
14. Index the implied volatilities `iv` by `ts`, then check `iv` and `ivpa` against the surviving asset count.
15. Subselect the [`AssetPanel`](@ref). Read the surviving assets' positions `acols` in the original asset names, recover the surviving rows with [`feature_row_indices`](@ref), and view the panel with [`port_opt_view`](@ref), handing it the surviving asset names so that a square tensor Panel Field is cut on its label axis too. An asset dropped by steps 9 and 10 takes its Panel Field values with it, or the panel and the returns would desynchronise silently. A time-varying panel is subselected to the surviving observations as well, matched back into the original price timestamps; a surviving timestamp absent from that clock throws. Under `collapse_args` this gives the aggregated period the values of the row at its representative timestamp, which is last-observation semantics and matches [`LastObservation`](@ref).
16. State the universe. Cut `span` to the price rows and the assets that survived with [`span_carrier_view`](@ref), and hand it and the converted returns to [`returns_universe_masks`](@ref), which projects it onto the returns clock and intersects it with finiteness. A carrier that states no span states no universe, and the conversion emits no panel. [`attach_universe_masks`](@ref) puts the pair onto the Asset Panel, keeping whatever Panel Fields it already carried, and mints one with no field when the carrier held none.
17. Build the asset, factor and benchmark matrices from the surviving columns. A group whose columns all went is `nothing`.
18. Return the [`ReturnsResult`](@ref).

Step 8 counts the missing columns of a row, and step 10 counts the missing rows of a column. The name of each keyword reads as the axis it counts, and the axis it drops is the other one. Both filters read the same table, because step 10 counts only over the rows that step 9 kept.

# Arguments

  - `X`: Asset price data (observations × assets).
  - `F`: Optional Factor price data (observations × factors).
  - `B`: Optional Benchmark price data (observations × assets) or (observations × 1).
  - `iv`: Optional Implied volatility data.
  - `ivpa`: Optional Implied volatility risk premium adjustment.
  - `ret_method`: Return calculation method (`:simple` or `:log`).
  - `padding`: Whether to pad missing values in returns calculation.
  - `gap_return_alg`: What the observations a price gap left non-finite carry. `nothing` is the arithmetic — a return is the change between two consecutive observations, so a run of `k` gapped prices leaves `k + 1` non-finite returns and the move across the gap is recorded nowhere — and [`CatchUpGapReturn`](@ref) books that move on the observation the asset resumes trading instead, shortening the Held Gap to `k`. Any algorithm may write only a non-finite cell inside an asset's Listing Span that has an earlier observed price in its column, so a return computed from two observed prices is frozen whichever one is stated. It has no cell to write over a gap-free table, and reports an `@info` there.
  - `missing_col_percent`: Maximum allowed fraction `(0, 1]` of missing **columns** in an observation row. A row above it is dropped. The name reads as the axis that is counted, not the axis that is dropped.
  - `missing_row_percent`: Maximum allowed fraction `(0, 1]` of missing **rows** in a column, counted over the rows that `missing_col_percent` kept. A column above it is dropped. `nothing` keeps the columns whose missing count equals the mode of the counts instead, which is the shape of a panel whose assets share one history.
  - `collapse_args`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func`: Optional function to apply to the data before returns calculation.
  - `join_method`: How to join asset, factor data and benchmark data (`:outer`, `:inner`, etc.).
  - `pnl`: Optional [`AssetPanel`](@ref), as [`asset_panel`](@ref) returns it.
  - `span`: Optional **Listing Span** on the price clock, as [`PriceIngestion`](@ref) derives it or a caller declares it. Given one, the conversion projects it onto the returns clock with [`universe_masks`](@ref) and hands the returns carrier an [`AssetPanel`](@ref) stating the universe — always, a gapless panel included, so `pnl === nothing` means one thing only: the carrier was not built by the ingestion layer. `nothing` states no universe and emits no panel, whether or not the prices hold a gap; a window-local span cannot answer the question, because a delisting straddling the window end reads there as an asset that was never listed.

# Validation

  - Every price reaching step 11 is positive. `TimeSeries.percentchange` takes a logarithm on both branches, so a negative price raises a `DomainError` from inside it, on the simple branch as well.
  - `!isempty(X)`.
  - `0 < missing_col_percent <= 1`
  - `0 < missing_row_percent <= 1`.
  - If `F` is not `nothing`, `!isempty(F)`.
  - If `B` is not `nothing`, `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`, the timestamps of the merged data matrix must be a subset of `TimeSeries.timestamp(iv)`, then `iv = values(iv)`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `all(x -> isfinite(x), iv)`, and `size(iv) == size(X)`.
  - If `span` is not `nothing`, `size(span) == (size(values(X), 1), size(values(X), 2))`. Raises a `DimensionMismatch`.
  - `ivpa` is validated in that same branch, so it is checked only when `iv` is given: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`, and, if a vector, `length(ivpa) == size(iv, 2)`. The bound is strict — a zero adjustment is rejected.

# Returns

  - `rr::ReturnsResult`: Struct containing asset/factor returns, names, time series, and optional implied volatility data. A converted benchmark is carried in its `B` field.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100 101; 102 103; 104 105],
                     [\"A\", \"B\"])
3×2 TimeSeries.TimeArray{Int64, 2, Dates.Date, Matrix{Int64}} 2020-01-01 to 2020-01-03
┌────────────┬─────┬─────┐
│            │ A   │ B   │
├────────────┼─────┼─────┤
│ 2020-01-01 │ 100 │ 101 │
│ 2020-01-02 │ 102 │ 103 │
│ 2020-01-03 │ 104 │ 105 │
└────────────┴─────┴─────┘

julia> prices_to_returns(X)
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ Vector{Dates.Date}: [Dates.Date("2020-01-02"), Dates.Date("2020-01-03")]
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
  - [`VecDate`](@ref)
  - [`Num_VecNum`](@ref)
  - [`TimeSeries`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type)
  - [`apply_gap_return`](@ref)
  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`CatchUpGapReturn`](@ref)
  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
  - [`returns_universe_masks`](@ref)
  - [`attach_universe_masks`](@ref)
  - [`span_carrier_view`](@ref)
  - [`returns_result_picker`](@ref): subtracts the carried benchmark, and only when the optimisation tracks it.
"""
function prices_to_returns(X::TimeSeries.TimeArray,
                           F::Option{<:TimeSeries.TimeArray} = nothing;
                           B::Option{<:TimeSeries.TimeArray} = nothing,
                           iv::Option{<:TimeSeries.TimeArray} = nothing,
                           ivpa::Option{<:Num_VecNum} = nothing,
                           ret_method::Symbol = :simple, padding::Bool = false,
                           gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
                           missing_col_percent::Number = 1.0,
                           missing_row_percent::Option{<:Number} = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Option{<:Function} = nothing,
                           join_method::Symbol = :outer,
                           pnl::Option{<:AssetPanel} = nothing,
                           span::Option{<:AbstractMatrix{Bool}} = nothing)
    @argcheck(!isempty(X), IsEmptyError)
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError)
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent), DomainError)
    end
    asset_names = string.(TimeSeries.colnames(X))
    asset_ts = TimeSeries.timestamp(X)
    check_asset_panel(pnl, length(asset_names), length(asset_ts),
                      "the number of asset price columns")
    assert_span_shape(span, length(asset_ts), length(asset_names))
    factor_names = String[]
    benchmark_names = String[]
    if !isnothing(F)
        @argcheck(!isempty(F), IsEmptyError)
        factor_names = string.(TimeSeries.colnames(F))
        X = TimeSeries.merge(X, F; method = join_method)
    end
    if !isnothing(B)
        @argcheck(!isempty(B), IsEmptyError)
        benchmark_names = string.(TimeSeries.colnames(B))
        @argcheck(length(benchmark_names) in (1, length(asset_names)), DimensionMismatch)
        X = TimeSeries.merge(X, B; method = join_method)
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = TimeSeries.collapse(X, collapse_args...)
    end
    X = DataFrames.DataFrame(X)

    # Absence has one spelling and it is `NaN`, because the returns level must carry it in a
    # `Matrix{Float64}`. A source spells it either way -- an outer join of ragged histories
    # pads with `NaN`, a wide table built from a tidy one leaves `missing` -- so this is the
    # one unification, and after it nothing below asks about spelling again.
    DataFrames.transform!(X,
                          2:DataFrames.DataAPI.ncol(X) .=>
                              DataFrames.ByRow((x) -> ifelse(ismissing(x), NaN, x));
                          renamecols = false)
    missing_mtx = is_missing_value.(Matrix(X[!, 2:end]))
    missings_cols = vec(count(missing_mtx; dims = 2))
    keep_rows = missings_cols .<= (DataFrames.DataAPI.ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, :]
    # Both filters read the same table: count the missing entries of a column over the rows
    # that survived the row filter, never over the rows that filter already dropped.
    missings_rows = vec(count(view(missing_mtx, keep_rows, :); dims = 1))
    keep_cols = if !isnothing(missing_row_percent)
        missings_rows .<= DataFrames.DataAPI.nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, [true; keep_cols]]
    P = X
    X = TimeSeries.percentchange(TimeSeries.TimeArray(X; timestamp = :timestamp),
                                 ret_method; padding = padding)
    X = DataFrames.DataFrame(X)
    X = apply_gap_return(gap_return_alg, X, P, ret_method)
    col_names = names(X)
    nx = intersect(col_names, asset_names)
    nf = intersect(col_names, factor_names)
    nb = intersect(col_names, benchmark_names)
    oc = setdiff(col_names, union(nx, nf, nb))
    N = length(nx)
    ts = isempty(oc) ? nothing : vec(Matrix(X[!, oc]))
    if !isnothing(ts) && !isnothing(iv)
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)),
                  ArgumentError("ts must be a subset of the timestamps in iv"))
        iv = iv[ts]
    end
    if !isnothing(iv)
        iv = values(iv)
        assert_nonempty_nonneg_finite_val(iv, :iv)
        assert_nonempty_gt0_finite_val(ivpa, :ivpa)
        @argcheck(size(iv) == (DataFrames.DataAPI.nrow(X), N), DimensionMismatch)
        if isa(ivpa, VecNum)
            @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
        end
    end
    if !isnothing(pnl)
        @argcheck(!isempty(nx),
                  IsEmptyError("every asset was dropped during the conversion, so the Asset Panel (pnl) has no asset axis left to bind to"))
        acols = Vector{Int}(indexin(nx, asset_names))
        rows = feature_row_indices(pnl, ts, asset_ts)
        pnl = port_opt_view(pnl, rows, acols, asset_names)
    end
    if !isempty(nx)
        #! The span is on the price clock and the masks are on the returns clock, so the
        #! span is cut to the price rows and assets that survived and universe_masks does
        #! the crossing. Both padding conventions reach it, and it reads which from the
        #! two row counts.
        cols = Vector{Int}(indexin(nx, asset_names))
        amsk, emsk = returns_universe_masks(span_carrier_view(span, P[!, :timestamp],
                                                              asset_ts, cols),
                                            Matrix(X[!, nx]))
        pnl = attach_universe_masks(pnl, amsk, emsk)
    end
    if isempty(nf)
        nf = nothing
        F = nothing
    else
        F = Matrix(X[!, nf])
    end
    if isempty(nb)
        nb = nothing
        B = nothing
    else
        B = length(nb) == 1 ? X[!, nb[1]] : Matrix(X[!, nb])
    end
    if isempty(nx)
        nx = nothing
        X = nothing
    else
        X = Matrix(X[!, nx])
    end
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, iv = iv,
                         ivpa = ivpa, pnl = pnl)
end
"""
    find_complete_indices(X::AbstractMatrix; dims::Int = 1) -> VecInt

Return the indices of columns (or rows) in matrix `X` that do not contain any missing or NaN values.

This function scans the specified dimension of the input matrix and returns the indices of columns (or rows) that are complete, i.e., contain no `missing` or `NaN` values.

Internal machinery — the caller-facing form is [`CompleteAssetSelector`](@ref), which wraps the `dims = 1` (complete-column) mode as a fit/apply estimator. The `dims = 2` (complete-row) mode has no estimator form: dropping observations is a price-level concern ([`MissingDataFilter`](@ref)).

# Algorithm

 1. Orient `X` with `dims_oriented`, so that the axis to test is axis 2 in both modes. `dims = 2` transposes the matrix, and `dims = 1` leaves it alone.
 2. Read the column count `N` of the oriented matrix.
 3. For each column of the oriented matrix, test whether it holds a `missing` entry or a `NaN` entry. Collect the positions of the columns that do, giving `to_remove`. One entry is enough to remove the whole column.
 4. Return `setdiff(1:N, to_remove)`, the positions of the complete columns, in ascending order.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - `dims in (1, 2)`.

# Returns

  - `res::VecInt`: Indices of columns (or rows) in `X` that are complete.

# Examples

```jldoctest
julia> X = [1.0 2.0 NaN; 4.0 missing 6.0];

julia> PortfolioOptimisers.find_complete_indices(X)
1-element Vector{Int64}:
 1

julia> PortfolioOptimisers.find_complete_indices(X; dims = 2)
Int64[]
```

# Related

  - [`CompleteAssetSelector`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`prices_to_returns`](@ref)
"""
function find_complete_indices(X::AbstractMatrix; dims::Int = 1)
    X = dims_oriented(dims, X)
    N = size(X, 2)
    to_remove = Vector{Int}(undef, 0)
    for i in axes(X, 2)
        if any(ismissing, X[:, i]) || any(isnan, X[:, i])
            push!(to_remove, i)
        end
    end
    return setdiff(1:N, to_remove)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing estimator types.

Preprocessing estimators transform price or returns data (prices-to-returns conversion, missing-data filtering, imputation) under a fit/apply contract. Fitting one on training data with [`fit_preprocessing`](@ref) produces a result carrying any fitted state — imputation parameters, thresholds, and the selected asset universe — which [`apply_preprocessing`](@ref) then replays on unseen data so train and test windows are transformed consistently. Stateless preprocessing estimators carry no state, and applying them is equivalent to running them.

They are ordinary estimators: they know nothing about pipelines. A `Pipeline` drives them through the same fit/apply verbs any other caller would use.

All concrete preprocessing estimators should subtype one of the two data-level subtypes:

  - [`AbstractPricesPreprocessingEstimator`](@ref): consumes and produces price-level data ([`PricesResult`](@ref)).
  - [`AbstractReturnsPreprocessingEstimator`](@ref): consumes and produces returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
abstract type AbstractPreprocessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce price-level data.

Concrete subtypes transform a [`PricesResult`](@ref) into another [`PricesResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce returns-level data.

Concrete subtypes transform a [`ReturnsResult`](@ref) into another [`ReturnsResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing result types.

Preprocessing results are produced by [`fit_preprocessing`](@ref) on training data. They carry the fitted state needed to apply the same transformation to unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators produce results that carry only their configuration.

All concrete preprocessing results should subtype one of the two data-level subtypes, [`AbstractPricesPreprocessingResult`](@ref) or [`AbstractReturnsPreprocessingResult`](@ref), so a caller can replay each fitted transformation at the data level it applies to.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
abstract type AbstractPreprocessingResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to price-level data ([`PricesResult`](@ref)).

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
"""
abstract type AbstractPricesPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractReturnsPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` counts as a missing observation in price-level data.

Price-level data stores absent observations either as `missing` or as `NaN` (the two conventions [`prices_to_returns`](@ref) already unifies).

# Algorithm

 1. Return `true` when `x` is `missing`.
 2. Return `true` when `x` is a `Number` and `isnan(x)` holds. The type test guards the call, because `isnan` is not defined for every value a price table can carry.
 3. Return `false` otherwise.

# Arguments

  - `x`: The value to test.

# Returns

  - `flag::Bool`: `true` when `x` is `missing` or a `NaN` number.

# Related

  - [`MissingDataFilter`](@ref)
  - [`Imputer`](@ref)
"""
function is_missing_value(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
"""
    fit_preprocessing(est::AbstractPreprocessingEstimator, data) -> fitted

Fit a preprocessing estimator on a data window and return the fitted object consumed by [`apply_preprocessing`](@ref).

The fitted object carries whatever state the transformation needs to be replayed consistently on unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators return themselves.

# Interfaces

Concrete preprocessing estimators must implement:

  - `fit_preprocessing(est::MyPreprocessing, data) -> fitted`: Compute the fitted state from the training window.
  - `apply_preprocessing(fitted, data) -> data′`: Transform a data window with the fitted state.

# Arguments

  - `est`: The preprocessing estimator.
  - `data`: The training data window ([`PricesResult`](@ref) or [`ReturnsResult`](@ref) depending on the estimator's level).

# Returns

  - `fitted`: The fitted object, typically an [`AbstractPreprocessingResult`](@ref) or the estimator itself when stateless.

# Related

  - [`apply_preprocessing`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
"""
function fit_preprocessing(est::AbstractPreprocessingEstimator, data)
    return throw(ArgumentError("$(typeof(est)) subtypes AbstractPreprocessingEstimator but does not implement fit_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′."))
end
"""
    apply_preprocessing(fitted, data) -> data′

Transform a data window with a fitted preprocessing object.

Applying the fitted object produced by [`fit_preprocessing`](@ref) on the training window to an unseen (test) window replays the *same* transformation — the same asset universe, the same imputation parameters — so train and test data stay consistent and no information flows from test to train.

# Arguments

  - `fitted`: The fitted object returned by [`fit_preprocessing`](@ref) (an [`AbstractPreprocessingResult`](@ref), or a stateless estimator).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window.

# Related

  - [`fit_preprocessing`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
function apply_preprocessing(fitted::Union{<:AbstractPreprocessingEstimator,
                                           <:AbstractPreprocessingResult}, data)
    return throw(ArgumentError("$(typeof(fitted)) subtypes the preprocessing interface but does not implement apply_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′; a stateless estimator returns itself from fit_preprocessing and does the work here."))
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator reserving the tail of the observations as a held-out test window.

The estimator form of [`train_test_split`](@ref), and the way the holdout protocol enters a [`Pipeline`](@ref): as the **first** step, it hands the training window to every step downstream and stashes the test window in its fitted [`TrainTestSplitResult`](@ref). `fit_predict(pipe, data)` then evaluates the fitted workflow on that held-out window in one line.

It is the one preprocessing estimator that is not pinned to a data level: it splits whichever level the pipeline input provides, price or returns, since a holdout is a statement about *rows*, not about columns or units.

Replaying a fitted split on an unseen window is a **pass-through** — the fitted rows are training-window state, and applying them to new data would be meaningless — so `predict(res, future_data)` keeps working on genuinely new observations.

!!! warning

    A pipeline containing a `TrainTestSplit` may not also be cross-validated: the split and the cross-validator are two evaluation protocols, and cross-validation already defines its own train/test windows. [`search_cross_validation`](@ref) rejects such a pipeline rather than silently shaving a second holdout off every fold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrainTestSplit(;
        train_size::Option{<:Number} = nothing,
        test_size::Option{<:Number} = nothing,
    ) -> TrainTestSplit

Keywords correspond to the struct's fields. Sizes follow [`safe_index`](@ref): a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`); one side given makes the other its complement; both given embargoes the rows between them; neither given splits 75/25.

# Examples

```jldoctest
julia> pipe = Pipeline(;
                       steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                                EmpiricalPrior(), EqualWeighted()));

julia> pipe.names
("split", "returns", "prior", "opt")
```

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`Pipeline`](@ref)
"""
@concrete struct TrainTestSplit <: AbstractPreprocessingEstimator
    """
    Training observations as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
    """
    train_size
    """
    Test observations, likewise; `nothing` takes the complement of `train_size`.
    """
    test_size
    function TrainTestSplit(train_size::Option{<:Number}, test_size::Option{<:Number})
        return new{typeof(train_size), typeof(test_size)}(train_size, test_size)
    end
end
function TrainTestSplit(; train_size::Option{<:Number} = nothing,
                        test_size::Option{<:Number} = nothing)::TrainTestSplit
    return TrainTestSplit(train_size, test_size)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`TrainTestSplit`](@ref), carrying both windows of the holdout.

The `test` window is the payoff: it is the data the fitted pipeline has never seen, and what `fit_predict(pipe, data)` predicts on. The `train` window is kept alongside it so the raw data the workflow was fitted on is retrievable from the result rather than having to be re-derived.

Both are [`port_opt_view`](@ref)s of the input at whichever level the split ran (price or returns).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TrainTestSplit`](@ref)
  - [`PipelineResult`](@ref)
"""
@concrete struct TrainTestSplitResult <: AbstractResult
    """
    The training window: the head of the observations, and the data every downstream step is fitted on.
    """
    train
    """
    The held-out test window: the tail of the observations, which no fitted step has seen.
    """
    test
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a [`TrainTestSplit`](@ref) by cutting the data into its two windows.

Unlike the other preprocessing estimators, the fitted result is *not* replayed on unseen data: a holdout's rows are a fact about the fitting window alone, so [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) passes the window through unchanged.

# Algorithm

 1. [`fit_preprocessing`](@ref) calls [`train_test_split`](@ref) on the data, and returns the [`TrainTestSplitResult`](@ref) that holds both windows.
 2. [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) returns its data argument unchanged.
 3. [`apply_preprocessing`](@ref) on a [`TrainTestSplit`](@ref) returns its data argument unchanged, so an unfitted step is a pass-through as well.

# Related

  - [`TrainTestSplit`](@ref)
  - [`train_test_split`](@ref)
"""
function fit_preprocessing(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    return train_test_split(tts, data)
end
function apply_preprocessing(::TrainTestSplitResult, data::Prices_RR)
    return data
end
function apply_preprocessing(::TrainTestSplit, data::Prices_RR)
    return data
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Split `data` under a [`TrainTestSplit`](@ref), returning both windows as a [`TrainTestSplitResult`](@ref).

The estimator-form counterpart of the keyword form: `train_test_split(rd; test_size = 0.2)` hands back a bare `(train, test)` tuple, while this hands back the same fitted result a pipeline's split step produces, so a holdout configured once can be reused verbatim inside and outside a [`Pipeline`](@ref).

# Algorithm

 1. Call the keyword form of [`train_test_split`](@ref) with `tts.train_size` and `tts.test_size`, giving the two windows.
 2. Wrap the pair in a [`TrainTestSplitResult`](@ref), in the order `(train, test)`.

# Related

  - [`TrainTestSplit`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function train_test_split(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    train, test = train_test_split(data; train_size = tts.train_size,
                                   test_size = tts.test_size)
    return TrainTestSplitResult(train, test)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for returns-level preprocessing estimators that restrict the asset universe.

An asset selector answers one question on the training window — *which asset columns survive?* — and that answer is its fitted state. [`apply_preprocessing`](@ref) replays the fitted universe on unseen windows, so a selector is safe inside cross-validation: the selection is made on train data alone and never re-decided on test data.

Concrete subtypes implement a single method, [`select_assets`](@ref); the family shares one [`fit_preprocessing`](@ref) and one [`apply_preprocessing`](@ref).

The funnel reduces the training window to its **Coverage Universe** before it calls [`select_assets`](@ref), so a selector ranks among the assets that are live throughout that window and never among an asset that is not yet listed, is delisted, or is missing a quote. A selector therefore needs no finiteness guard of its own, and [`CompleteAssetSelector`](@ref) is the identity on the reduced window.

Selectors restrict *columns only*. Observation filtering is a price-level concern ([`MissingDataFilter`](@ref)), because a fitted transformation cannot decide which rows of an unseen window to drop without breaking the weights/returns alignment `assert_universe_aligned` enforces.

See `docs/adr/0029-asset-selection-is-returns-preprocessing.md` for the design rationale.

# Related

  - [`select_assets`](@ref)
  - [`AssetSelectorResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractAssetSelector <: AbstractReturnsPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of any [`AbstractAssetSelector`](@ref).

Carries the asset universe selected on the training window. One result type serves the whole family: every selector differs in *how* it chooses the universe, never in what it stores.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`select_assets`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
@concrete struct AssetSelectorResult <: AbstractReturnsPreprocessingResult
    """
    Names of the assets that survived the training window, in their original column order (the fitted universe).
    """
    nx
end
"""
    select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult) -> BitVector

Return the keep-mask over the asset columns of `rd`.

This is the single method a concrete [`AbstractAssetSelector`](@ref) must implement. It is called by [`fit_preprocessing`](@ref) on the **Coverage Universe of the training window** only; the resulting universe is then replayed on every later window by [`apply_preprocessing`](@ref).

`rd` is the *reduced* carrier, so every column it carries is finite at every row and active at every row of the panel. A selector ranks among live assets alone, and it needs no finiteness guard: a non-finite score computed from a live column is a defect of the measure, which is why [`asset_scores`](@ref) keeps its refusal.

`rd` is read for `nx` and an `observations × assets` `X`; the funnel itself reads `rd.pnl`, and [`ClusterGroups`](@ref) reads it too, so the implicit contract of the family is `{nx, X, pnl}` (see [`AbstractReturnsResult`](@ref)). A selector is fitted from returns data alone and never sees a prior result, so it reads the data carrier and nothing else.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Returns

  - `keep::BitVector`: `true` for each asset column to retain, `length(keep) == size(rd.X, 2)`, over the reduced window.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`coverage_reduction(rd::AbstractReturnsResult)`](@ref)
"""
function select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult)
    return throw(ArgumentError("$(typeof(sel)) subtypes AbstractAssetSelector but does not implement select_assets. Extension authors: every AbstractAssetSelector must define select_assets(sel, rd) returning a keep-mask over the asset columns of rd."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit any [`AbstractAssetSelector`](@ref) by recording the asset universe [`select_assets`](@ref) keeps.

This is the one funnel of the family, and it is where the Coverage Universe is applied. The window is reduced first, so **every** selector ranks among the assets that are live throughout the training window: a score, a redundancy and a rule all read live columns alone, and a new selector cannot forget the rule. An all-dead window throws an `IsEmptyError` where the mask is derived, so the refusal is [`coverage_mask`](@ref)'s.

# Algorithm

 1. Reduce the training window to its Coverage Universe with [`coverage_reduction(rd::AbstractReturnsResult)`](@ref).
 2. Call [`select_assets`](@ref) on the reduced window, giving the keep-mask `keep`.
 3. Check that `keep` holds one entry per asset column of the reduced window.
 4. Check that `keep` keeps at least one asset.
 5. Return an [`AssetSelectorResult`](@ref) holding the names of the kept assets, in their original column order.

The result records **names**, so the expansion is free: [`apply_preprocessing`](@ref) finds each name in the window it replays on, and a name the reduction dropped is simply absent from the fitted universe.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Validation

  - The carrier must hold an `observations × assets` returns matrix; one that collapsed the asset axis matches no method of the reduction and is named by a `MethodError`.
  - At least one asset must be in the Coverage Universe of the training window.
  - `select_assets` must return a mask whose length matches the number of asset columns of the reduced window.
  - The selection must keep at least one asset; a selector that empties the universe throws rather than passing a zero-asset problem downstream (the [`MissingDataFilter`](@ref) precedent).

# Returns

  - `res::AssetSelectorResult`: The fitted asset universe.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`AssetSelectorResult`](@ref)
  - [`apply_preprocessing`](@ref)
"""
function fit_preprocessing(sel::AbstractAssetSelector,
                           rd::AbstractReturnsResult)::AssetSelectorResult
    _, rdc = coverage_reduction(rd)
    keep = select_assets(sel, rdc)
    @argcheck(length(keep) == size(rdc.X, 2),
              DimensionMismatch("select_assets for a $(typeof(sel)) returned a mask of length $(length(keep)) for the $(size(rdc.X, 2)) asset columns of the Coverage Universe of the training window"))
    @argcheck(any(keep),
              IsEmptyError("a $(typeof(sel)) selects no assets from the Coverage Universe of the training window; loosen its configuration"))
    return AssetSelectorResult(collect(rdc.nx[keep]))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replay a fitted asset universe on a data window.

The surviving columns are emitted in *fitted* order, not in the window's own column order, because the terminal weights are indexed by the training universe and `assert_universe_aligned` compares the two name vectors elementwise.

# Algorithm

 1. For each fitted asset name, in fitted order, find the column of the window that carries it, and record that position in `idx`.
 2. Check that the name is present. A name the window does not carry throws.
 3. Return a [`port_opt_view`](@ref) of the window at `idx`. The positions are in fitted order, so the view reorders the window's columns when the two orders differ.

# Arguments

  - `res`: The fitted asset universe.
  - `rd`: The data window to transform.

# Validation

  - Every fitted asset name must be present in the window; a missing one throws rather than silently shrinking the universe.

# Returns

  - `rd′::AbstractReturnsResult`: The window restricted to the fitted universe, in fitted order.

# Related

  - [`AssetSelectorResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function apply_preprocessing(res::AssetSelectorResult, rd::AbstractReturnsResult)
    idx = Vector{Int}(undef, length(res.nx))
    for (k, name) in pairs(res.nx)
        j = findfirst(==(name), rd.nx)
        @argcheck(!isnothing(j),
                  ArgumentError("the fitted asset \"$name\" is absent from the data window, whose assets are $(collect(rd.nx)); the window must contain the whole fitted universe $(res.nx)"))
        idx[k] = j
    end
    return port_opt_view(rd, idx)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator converting price-level data into returns-level data.

`PricesToReturns` is the estimator form of [`prices_to_returns`](@ref): it consumes a [`PricesResult`](@ref) and produces a [`ReturnsResult`](@ref). It is stateless — applying it to any window simply runs the conversion — so its fitted object is the estimator itself.

Missing-data filtering is deliberately *not* part of this estimator (the corresponding [`prices_to_returns`](@ref) keywords are held at their permissive defaults); use [`MissingDataFilter`](@ref) and [`PriceGapFill`](@ref) as separate, independently tunable steps. Deleting an observation or an asset is a **Universe Policy**, and a policy is fitted on a training window and replayed by name; this step is stateless, so it holds none.

The step is stateless, and it does not need to be stateful to fix an asset universe: the carrier states one. A [`PricesResult`](@ref) that [`price_ingestion`](@ref) built carries a **Listing Span**, and this step projects it onto the returns clock and hands the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) whose two masks say which assets are in the universe and which of them can be estimated at each observation. The asset axis is fixed before the split, so every window of every fold carries every asset and a window can no longer silently lose a column.

!!! warning

    A carrier the ingestion layer did not build states no universe, and the conversion does not guess one from the window: a window-local span reads a delisting straddling the window end as an asset that was never listed. Its gaps are still carried and still handled — with no panel the Coverage Universe reads finiteness alone — but the fold is left to infer the universe it would otherwise have been told. Build the carrier with [`price_ingestion`](@ref), or declare a listing calendar as its `span`.

# Algorithm

The estimator is stateless, so both verbs are thin.

 1. [`fit_preprocessing`](@ref) returns the estimator itself. There is no state to fit.
 2. [`apply_preprocessing`](@ref) calls [`prices_to_returns`](@ref) with the six fields as keywords, and with `X`, `F`, `B`, `iv`, `ivpa`, `pnl` and `span` read off the [`PricesResult`](@ref). It returns the [`ReturnsResult`](@ref).

The two threshold keywords of [`prices_to_returns`](@ref) are not fields of this estimator, so they hold their permissive defaults and every row and column reaches the conversion. `gap_return_alg` *is* a field, because it decides what the observations a gap left non-finite carry, which is the arithmetic of a return rather than a policy about the universe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturns(;
        ret_method::Symbol = :simple,
        padding::Bool = false,
        gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
        collapse_args::Tuple = (),
        map_func::Option{<:Function} = nothing,
        join_method::Symbol = :outer,
        strict::Bool = false,
    ) -> PricesToReturns

Keywords correspond to the struct's fields.

## Validation

  - `ret_method in (:simple, :log)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> rr = apply_preprocessing(PricesToReturns(), pr);

julia> size(rr.X)
(2, 2)

julia> rr.nx
2-element Vector{String}:
 "A"
 "B"
```

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`prices_to_returns`](@ref)
  - [`AbstractGapReturnAlgorithm`](@ref)
  - [`CatchUpGapReturn`](@ref)
  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct PricesToReturns <: AbstractPreprocessingEstimator
    """
    Return calculation method (`:simple` or `:log`).
    """
    ret_method
    """
    Whether to pad missing values in the returns calculation.
    """
    padding
    """
    What the observations a price gap left non-finite carry. `nothing` is the arithmetic, and [`CatchUpGapReturn`](@ref) books the move across the gap on the observation that ends it. See [`AbstractGapReturnAlgorithm`](@ref).
    """
    gap_return_alg
    """
    Arguments for collapsing the time series (e.g. to lower frequency).
    """
    collapse_args
    """
    Optional function applied to the data before the returns calculation.
    """
    map_func
    """
    How asset, factor, and benchmark data are joined (`:outer`, `:inner`, etc.).
    """
    join_method
    function PricesToReturns(ret_method::Symbol, padding::Bool,
                             gap_return_alg::Option{<:AbstractGapReturnAlgorithm},
                             collapse_args::Tuple, map_func::Option{<:Function},
                             join_method::Symbol)
        @argcheck(ret_method in (:simple, :log),
                  ArgumentError("ret_method must be :simple or :log, got :$ret_method"))
        return new{typeof(ret_method), typeof(padding), typeof(gap_return_alg),
                   typeof(collapse_args), typeof(map_func), typeof(join_method)}(ret_method,
                                                                                 padding,
                                                                                 gap_return_alg,
                                                                                 collapse_args,
                                                                                 map_func,
                                                                                 join_method)
    end
end
function PricesToReturns(; ret_method::Symbol = :simple, padding::Bool = false,
                         gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
                         collapse_args::Tuple = (), map_func::Option{<:Function} = nothing,
                         join_method::Symbol = :outer)::PricesToReturns
    return PricesToReturns(ret_method, padding, gap_return_alg, collapse_args, map_func,
                           join_method)
end
function prices_to_returns(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(pr.X, pr.F; B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                             ret_method = ptr.ret_method, padding = ptr.padding,
                             gap_return_alg = ptr.gap_return_alg,
                             collapse_args = ptr.collapse_args, map_func = ptr.map_func,
                             join_method = ptr.join_method, pnl = pr.pnl, span = pr.span)
end
function fit_preprocessing(ptr::PricesToReturns, ::PricesResult)
    return ptr
end
function apply_preprocessing(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(ptr, pr)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator dropping assets and observations with excessive missing data from price-level data.

The *asset universe is fitted state*: the training window decides which assets survive (per-column missing fraction at most `col_thr`), and applying the fitted result to an unseen window subsets it to that same universe — so train weights and test returns always refer to the same assets. Observation (row) filtering is window-local: rows whose missing fraction across the surviving assets exceeds `row_thr` are dropped from whichever window is being transformed.

This estimator supersedes the `missing_col_percent`/`missing_row_percent` keywords of [`prices_to_returns`](@ref), making the thresholds fitted state and independently tunable. Only the asset series `X` (and the matching implied volatility columns, and the feature matrix, whose axes are parallel to `X`) participate; factor and benchmark series pass through unchanged.

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

## Validation

  - `0 < col_thr <= 1`.
  - `0 < row_thr <= 1`.

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
    Maximum allowed fraction `(0, 1]` of missing observations per asset column; assets above it are dropped from the universe at fit time.
    """
    col_thr
    """
    Maximum allowed fraction `(0, 1]` of missing assets per observation row; rows above it are dropped from the window being transformed.
    """
    row_thr
    function MissingDataFilter(col_thr::Number, row_thr::Number)
        @argcheck(zero(col_thr) < col_thr <= one(col_thr), DomainError)
        @argcheck(zero(row_thr) < row_thr <= one(row_thr), DomainError)
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
    Maximum allowed fraction `(0, 1]` of missing assets per observation row.
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
"""
    asset_panel(ape::Nothing, pr, rd::ReturnsResult, X) -> AssetPanel
    asset_panel(ape::Nothing, pr::ReturnsResult, rd::Nothing, X) -> AssetPanel
    asset_panel(ape::Nothing, pr, rd::Nothing, X) -> Union{}

Resolve the [`AssetPanel`](@ref) a [`FeatureDistance`](@ref) with no producer measures.

`nothing` in the `ape` slot says *read the panel the data carrier already holds*. The carriers reach the kernel as the two keywords `pr` and `rd`, and this verb resolves the source by dispatch: a [`ReturnsResult`](@ref) in either slot answers its `pnl`, and `rd` wins when both hold one, because the data carrier is where a panel is data rather than a by-product. `Pr_RR` admits a [`ReturnsResult`](@ref) in the `pr` slot, which is what `clusterise(cle, rd)` and every [`Pipeline`](@ref) step pass, so the second method is not a fallback but the shortest public call.

A prior result alone carries no panel, so it raises an [`IsNothingError`](@ref) naming the two ways forward.

# Algorithm

The method that Julia selects is the algorithm.

 1. `rd` is a [`ReturnsResult`](@ref): answer `rd.pnl`.
 2. `pr` is a [`ReturnsResult`](@ref) and there is no `rd`: answer `pr.pnl`.
 3. Neither slot holds a data carrier: raise.

Each of the first two checks that the carrier it read holds a panel, with [`assert_asset_panel_supplied`](@ref).

# Arguments

  - `ape`: `nothing`, which reads the carrier's panel.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `X`: Returns matrix of the subproblem. Unread here; a producer reads it.

# Validation

  - A data carrier is present, and it holds an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the data carrier holds.

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`assert_asset_panel_supplied`](@ref)
  - [`FeatureDistance`](@ref)
  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`RegressionPanel`](@ref)
  - [`PhylogenyPanel`](@ref)
"""
function asset_panel(::Nothing, ::Any, rd::ReturnsResult, ::Any)
    return assert_asset_panel_supplied(rd.pnl)
end
function asset_panel(::Nothing, pr::ReturnsResult, ::Nothing, ::Any)
    return assert_asset_panel_supplied(pr.pnl)
end
function asset_panel(::Nothing, ::Any, ::Nothing, ::Any)
    return throw(IsNothingError("`FeatureDistance` with no producer reads the Asset Panel off the data carrier, and this call supplied none: only a prior result reached it, and a prior result carries no panel. Two ways forward:\n  1. Pass the `ReturnsResult` that holds the panel, which every forwarder takes as `rd`.\n  2. Set a producer on the estimator, `FeatureDistance(; ape = RegressionPanel())`, which builds a panel from the prior it is handed."))
end
"""
    assert_asset_panel_supplied(pnl::AssetPanel) -> AssetPanel
    assert_asset_panel_supplied(pnl::Nothing) -> Union{}

Assert that the data carrier a [`FeatureDistance`](@ref) read holds an [`AssetPanel`](@ref), and return it.

The carrier's `pnl` is optional, so a carrier built without one reaches the kernel as `nothing`. This is the one place that turns it into a diagnostic, and it returns the panel so the caller reads one verb rather than a check and an access.

# Algorithm

The method that Julia selects is the algorithm. A panel is returned; `nothing` raises.

# Arguments

  - `pnl`: The carrier's Asset Panel, or `nothing`.

# Validation

  - `!isnothing(pnl)`. Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel.

# Related

  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`FeatureDistance`](@ref)
  - [`IsNothingError`](@ref)
"""
function assert_asset_panel_supplied(pnl::AssetPanel)
    return pnl
end
function assert_asset_panel_supplied(::Nothing)
    return throw(IsNothingError("`FeatureDistance` with no producer reads the Asset Panel off the data carrier, and the carrier holds none. Build one with `asset_panel(inputs)` and pass it as `ReturnsResult(; …, pnl = pnl)`, or set a producer on the estimator, `FeatureDistance(; ape = RegressionPanel())`."))
end

export PricesResult, ReturnsResult, prices_to_returns, returns_result_picker,
       fit_preprocessing, apply_preprocessing, PricesToReturns, CatchUpGapReturn,
       MissingDataFilter, MissingDataFilterResult, Imputer, ImputerResult,
       AssetSelectorResult, train_test_split, TrainTestSplit, TrainTestSplitResult
