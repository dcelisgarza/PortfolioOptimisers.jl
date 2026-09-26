"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all price-level data result types.

Every concrete type that holds price-level data subtypes `AbstractPricesResult`. [`PricesResult`](@ref) is the member the library builds. The fold generation of cross-validation, the pipeline and the price-level preprocessing steps dispatch on this supertype, as they dispatch on [`AbstractReturnsResult`](@ref) for returns-level data. No check enforces the interface below.

# Interfaces

To implement a new price-level carrier that a pipeline and a cross-validation fold can read, subtype `AbstractPricesResult` with these fields:

  - `X`: The asset prices, a `TimeSeries.TimeArray` of `observations × assets`. Fold generation reads its timestamps and its row count, and asset subset sampling reads its column count.
  - `pnl`: An [`AssetPanel`](@ref) over the assets of `X`, or `nothing`. Fold generation reads it together with `X` to find the assets that hold data.

Then implement the following method:

## `port_opt_view`

  - `port_opt_view(pr::MyPricesResult, i, j = :) -> MyPricesResult`: Return the carrier on the observations at `i` and the assets at `j`.

A subtype that carries a panel subselects it together with `X`. A subtype that implements no method gets the fallback, which throws an `ArgumentError`.

### Arguments

  - `pr`: The concrete price-level carrier.
  - `i`: Indices or timestamps of the observations to keep.
  - `j`: Indices of the assets to keep.

### Returns

  - `pr::MyPricesResult`: A carrier of the same type that holds only the selected observations and assets.

# Related

  - [`AbstractResult`](@ref)
  - [`PricesResult`](@ref)
  - [`AbstractReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractPricesResult <: AbstractResult end
"""
    assert_nonneg_where_present(val::AbstractArray, sym::Union{Symbol, <:AbstractString} = :val) -> nothing

Refuse a negative or an infinite value, and let an absence through.

The ingestion layer carries an absent implied volatility as `NaN`, as it carries an absent price, and the carriers accept it. [`ImpliedVolatility`](@ref) reads the series from a carrier, and narrows its Coverage Universe to the assets whose values are complete. So an absent value removes the asset from the fit, and the fit does not fail. The carrier refuses a present value that is not a volatility, which is a negative value or an infinite one. An infinite value is neither a volatility nor the marker of an absence. [`assert_cross_sectional_matrix`](@ref) applies the same rule to a returns matrix. A carrier built by hand can also hold `missing`, and this check accepts it as an absence too. [`prices_to_returns`](@ref) changes each `missing` to `NaN` before an estimator reads the series.

# Arguments

  - `val`: The array to check.
  - $(arg_dict[:sym_msg])

# Validation

  - Every element is `missing`, `NaN`, or finite and non-negative. A breach raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`price_ingestion`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_cross_sectional_matrix`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
function assert_nonneg_where_present(val::AbstractArray, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> ismissing(x) || isnan(x) || (isfinite(x) && zero(x) <= x), val),
              DomainError(val,
                          "all(x -> ismissing(x) || isnan(x) || (isfinite(x) && 0 <= x), $sym) must hold: an absent value is carried, and a present one is finite and non-negative. Got\ncount(x -> !ismissing(x) && !isnan(x) && !(isfinite(x) && 0 <= x), $sym) => $(count(x -> !ismissing(x) && !isnan(x) && !(isfinite(x) && zero(x) <= x), val))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds asset prices with optional factor, benchmark and implied volatility series, each a `TimeSeries.TimeArray`.

It is the price-level counterpart of [`ReturnsResult`](@ref). The price-level preprocessing steps and [`prices_to_returns`](@ref) take it as input, and [`port_opt_view`](@ref) cuts it into the timestamp windows of pipeline cross-validation.

The asset price series `X` is the master clock. [`port_opt_view`](@ref) selects observation windows on `X`, and each other series keeps its rows at the timestamps that `X` keeps.

The [`AssetPanel`](@ref) `pnl` and the Listing Span `span` hold no timestamps, so no timestamp aligns them. A Panel Field can be a 3-dimensional array, a static panel has no observation axis, and a span can be two integers per asset. Their axes run parallel to `X` by position instead. The asset axis follows `TimeSeries.colnames(X)`, and the observation axis of a time-varying panel follows `TimeSeries.timestamp(X)` row for row. A routine that drops an asset or an observation from `X` must drop it from `pnl` and `span` in the same step. [`port_opt_view`](@ref) does this through [`panel_carrier_view`](@ref) and [`span_carrier_view`](@ref), and [`MissingDataFilter`](@ref) and [`prices_to_returns`](@ref) do it too.

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
  - If `iv` is not `nothing`: `!isempty(iv)`, `size(values(iv), 2) == size(values(X), 2)`, and every present value is finite and non-negative. An absent value is `NaN`, or `missing` on a carrier built by hand. See [`assert_nonneg_where_present`](@ref).
  - If `ivpa` is not `nothing`: `!isempty(ivpa)`, `all(isfinite, ivpa)` and `all(x -> x > 0, ivpa)`. If `ivpa` is a vector: `length(ivpa) == size(values(X), 2)`.
  - If `pnl` is not `nothing`: its asset axis has the length `size(values(X), 2)`, and the observation axis of a time-varying panel has the length `size(values(X), 1)`. See [`check_asset_panel`](@ref).
  - If `span` is not `nothing`: `size(span) == size(values(X))`.

An empty series or an empty `ivpa` raises an `IsEmptyError`. A wrong width or a wrong shape raises a `DimensionMismatch`. A value outside its domain raises a `DomainError`.

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
  - [`check_asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`PortfolioOptimisers.ListingSpan`](@ref)
"""
@concrete struct PricesResult <: AbstractPricesResult
    """
    Asset prices, `observations × assets`. It is the master clock of every timestamp window.
    """
    X
    """
    Optional factor prices, `observations × factors`.
    """
    F
    """
    Optional benchmark prices, `observations × 1` or `observations × assets`.
    """
    B
    """
    Optional implied volatilities, `observations × assets`. An absent value is `NaN`, and [`ImpliedVolatility`](@ref) removes the asset from its fit.
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    Optional [`AssetPanel`](@ref) that holds the Panel Fields of the universe and its two universe masks. It is not a `TimeArray`, and its axes run parallel to `X` by position.
    """
    pnl
    """
    Optional Listing Span, `observations × assets`, that states which assets are listed at each observation of the price clock. Its axes run parallel to `X` by position. It is a [`PortfolioOptimisers.ListingSpan`](@ref) when [`PriceIngestion`](@ref) derives it by the Span Rule, and any other `AbstractMatrix{Bool}` when a caller states their own listing calendar. It is `nothing` when the ingestion layer did not build the carrier.
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
            @argcheck(!isempty(iv), IsEmptyError)
            @argcheck(size(values(iv), 2) == size(values(X), 2), DimensionMismatch)
            assert_nonneg_where_present(values(iv), :iv)
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

Return the `PricesResult` on the observation window `i` and the assets `j` of the asset price series `X`.

Indexing a `TimeArray` copies its rows, so `X`, `F`, `B` and `iv` of the result are copies. `ivpa`, the Asset Panel and the Listing Span of the result can share memory with `pr`, and the call with two `Colon`s returns `pr` itself.

The asset price series is the master clock. `i` selects rows of `X`, and the factor, benchmark and implied volatility series keep their rows at the timestamps that `X` keeps. The rows come back in clock order whatever the order of `i`, and a timestamp of `i` that `X` does not hold selects no row. `j` selects asset columns in the order it gives them, and its default is `:`. So a call that gives only `i` is an observation window over the whole universe.

The first index selects observations here. The two-argument method of [`ReturnsResult`](@ref) selects assets, so `i` names a different axis on the two carriers.

# Algorithm

The method that Julia selects is the algorithm. The timestamp methods do the work, and the integer method sends the call to them.

 1. `i` and `j` are both `Colon`: return `pr` itself. No view is built.

 2. `i` is a vector of timestamps and `j` is a `Colon`: index `X`, `F`, `B` and `iv` by the timestamps `i`. Recover the rows of a time-varying Asset Panel from the kept timestamps with [`feature_row_indices`](@ref), and view the panel on that observation axis with [`panel_carrier_view`](@ref). A static panel has no observation axis and ignores the row index. View the Listing Span at the same timestamps with [`span_carrier_view`](@ref). Pass `ivpa` through unchanged, because the asset index does not reach it. Rebuild the [`PricesResult`](@ref).

 3. `i` is a vector of timestamps and `j` is a vector of asset indices:

     1. Index `X` by the timestamps `i`, then keep the asset columns `j`.
     2. Index `F` by the timestamps `i` alone. `j` is an asset index, and the factors are a separate axis, so every factor column stays.
     3. Index `B` by the timestamps `i`. Keep its columns `j` when `B` holds one column per asset, and keep its single column otherwise. The test reads the width of `B`, because a shared benchmark has only one column to give.
     4. Index `iv` by the timestamps `i` and the asset columns `j`, and view `ivpa` at `j`.
     5. Recover the rows of a time-varying Asset Panel with [`feature_row_indices`](@ref). View the panel at those rows and the assets `j` with [`panel_carrier_view`](@ref), and give it the asset names. It then also cuts the label axis of a tensor Panel Field whose labels are the asset names, see [`features_are_assets`](@ref).
     6. View the Listing Span at the kept timestamps and the assets `j` with [`span_carrier_view`](@ref).
     7. Rebuild the [`PricesResult`](@ref).

 4. `i` and `j` are integer indices, ranges or `Colon`s: read the timestamps `TimeSeries.timestamp(pr.X)[i]`, and call step 2 or step 3 with them. A call such as `port_opt_view(pr, 2:3)` reaches this method.

 5. Any other call on an [`AbstractPricesResult`](@ref): throw an `ArgumentError` that names the type of each index argument, the keyword arguments, and the two call shapes of the interface. A call with one integer, with three indices or with a keyword argument reaches this step. So does a subtype that implements no method.

# Arguments

  - `pr`: A `PricesResult` object.
  - `i`: Observation window into the rows of `pr.X`. Either integer indices (`AbstractVector{<:Integer}`, `AbstractRange`, or `Colon`) or a vector of timestamps (`AbstractVector{<:Dates.AbstractTime}`).
  - `j`: Asset window into the columns of `pr.X`. Integer indices, an `AbstractRange`, or `Colon` for the whole universe. A `Colon` keeps every asset column of `X`, `B` and `iv`, and passes `ivpa` through unchanged.

# Validation

  - A window that keeps no row raises the `IsEmptyError` of the [`PricesResult`](@ref) constructor.
  - A call shape that step 5 takes raises an `ArgumentError`.

# Returns

  - `new_pr::PricesResult`: The carrier on the selected observations and assets. It is `pr` itself when `i` and `j` are both `Colon`s.

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
  - [`AbstractPricesResult`](@ref)
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
function port_opt_view(pr::AbstractPricesResult, args...; kwargs...)
    kws = keys(kwargs)
    kwmsg = isempty(kws) ? "" : " and the keyword argument(s) " * join(kws, ", ")
    return throw(ArgumentError("port_opt_view has no method for a $(nameof(typeof(pr))) with the index argument type(s) ($(join(typeof.(args), ", ")))$(kwmsg). A price-level carrier takes port_opt_view(pr, observations) or port_opt_view(pr, observations, assets), with no keyword argument. The observations are integer indices, a range, a Colon or a vector of timestamps, and the assets are integer indices, a range or a Colon. A subtype of AbstractPricesResult implements these two shapes."))
end
export PricesResult
public AbstractPricesResult
