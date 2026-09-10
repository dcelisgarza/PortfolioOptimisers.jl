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
export PricesResult
