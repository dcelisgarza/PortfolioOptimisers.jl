"""
    unify_gaps(A::TimeSeries.TimeArray) -> TimeSeries.TimeArray

Spell every absent price of one series the one way the ingestion layer carries.

A source spells an absent price either way — an outer join of ragged per-asset histories pads with `NaN`, a wide table built from a tidy one leaves `missing` — and the layer unifies them as `NaN`, which no deletion step reads. This is what defines a gap for every later piece, so it runs before the Span Rule and before the join.

# Algorithm

 1. A series whose values are already floating point carries its gaps as `NaN`. Return it untouched.
 2. Otherwise rebuild it, mapping `missing` to `NaN` and converting every other entry to `Float64`.

# Arguments

  - `A`: One price series.

# Returns

  - `A′::TimeSeries.TimeArray`: The same series, with every absent price spelled `NaN`.

# Related

  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
  - [`listing_span`](@ref)
"""
function unify_gaps(A::TimeSeries.TimeArray)
    v = values(A)
    if isa(v, AbstractArray{<:AbstractFloat})
        return A
    end
    return TimeSeries.TimeArray(TimeSeries.timestamp(A),
                                [ismissing(x) ? NaN : Float64(x) for x in v],
                                TimeSeries.colnames(A))
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator assembling raw price series into the span-carrying price carrier the ingestion layer converts.

`PriceIngestion` runs **once on the whole panel**, and it is deliberately *not* a [`Pipeline`](@ref) step. The rule that puts it outside is the observation clock: unification, the factor and benchmark join, and the frequency collapse each move or renumber the observations, folds are cut once on the carrier's clock, and a hyperparameter that changes the test set cannot be scored against one that does not. A step that only touches values leaves the clock alone and stays inside.

Running outside the `Pipeline` is also what lets the **Span Rule** read the whole panel. A step only ever sees a window, and a window-local span reads a delisting straddling the window end as an asset that was never listed. What licenses the panel-wide read where a split otherwise forbids work before it is that a listing calendar is a fact about the *instruments*, not an estimate from returns.

The carrier it emits holds the derived **Listing Span** in its `span` field, so the conversion can project it onto the returns clock and hand the returns carrier an [`AssetPanel`](@ref) stating the universe. A caller holding their own listing calendar passes it as `span` and replaces the Span Rule's answer outright.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceIngestion(;
        join_method::Symbol = :outer,
        collapse_args::Tuple = (),
        span::Option{<:AbstractMatrix{Bool}} = nothing,
    ) -> PriceIngestion

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [NaN 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = price_ingestion(PriceIngestion(), X);

julia> pr.span
ListingSpan(3 × 2)

julia> Matrix(pr.span)
3×2 Matrix{Bool}:
 0  1
 1  1
 1  1
```

# Related

  - [`price_ingestion`](@ref)
  - [`PricesResult`](@ref)
  - [`PricesToReturns`](@ref)
  - [`listing_span`](@ref)
  - [`universe_masks`](@ref)
  - [`Option`](@ref)
"""
@concrete struct PriceIngestion <: AbstractEstimator
    """
    How the asset, factor and benchmark series are joined onto one clock (`:outer`, `:inner`, etc.).
    """
    join_method
    """
    Arguments for collapsing the joined series to a lower frequency, as `TimeSeries.collapse` takes them. Empty leaves the clock alone.
    """
    collapse_args
    """
    Optional listing statement of the caller's own, `price observations × assets`, replacing the **Span Rule**'s answer outright. A listing calendar, or a constituency that leaves and rejoins, is any `AbstractMatrix{Bool}`. `nothing` derives the span from the gaps.
    """
    span
    function PriceIngestion(join_method::Symbol, collapse_args::Tuple,
                            span::Option{<:AbstractMatrix{Bool}})
        return new{typeof(join_method), typeof(collapse_args), typeof(span)}(join_method,
                                                                             collapse_args,
                                                                             span)
    end
end
function PriceIngestion(; join_method::Symbol = :outer, collapse_args::Tuple = (),
                        span::Option{<:AbstractMatrix{Bool}} = nothing)::PriceIngestion
    return PriceIngestion(join_method, collapse_args, span)
end
"""
    price_ingestion(est::PriceIngestion, X::TimeSeries.TimeArray;
                    F::Option{<:TimeSeries.TimeArray} = nothing,
                    B::Option{<:TimeSeries.TimeArray} = nothing,
                    iv::Option{<:TimeSeries.TimeArray} = nothing,
                    ivpa::Option{<:Num_VecNum} = nothing,
                    pnl::Option{<:AssetPanel} = nothing) -> PricesResult
    price_ingestion(est::PriceIngestion, pr::PricesResult) -> PricesResult

Assemble raw price series into the span-carrying price carrier.

The three clock-moving steps run here rather than in a [`Pipeline`](@ref), and the **Span Rule** reads the whole panel once they have. What the caller gets back is an ordinary [`PricesResult`](@ref) whose `span` field states which assets are listed at each observation, ready for [`PricesToReturns`](@ref) to project onto the returns clock.

The layer spells an absent price `NaN`, which is the library's one spelling for absence, and the conversion carries it into the returns rather than deleting the observation or the asset that holds one.

# Algorithm

 1. Check that the asset, factor and benchmark series can still be named after the join with [`assert_distinct_series_names`](@ref). The join renames a name two tables share, so a block would otherwise be taken apart into another block's column.
 2. Unify the absent-price convention of every series with [`unify_gaps`](@ref), so that a gap means one thing from here on.
 3. Join the factor and the benchmark series onto the asset clock under `join_method`. An outer join adds the rows one series has and another does not, padding them as gaps.
 4. Collapse the joined series to a lower frequency when `collapse_args` is non-empty, which renumbers every observation.
 5. Split the joined series back into their asset, factor and benchmark blocks, all now on one clock.
 6. Align the implied volatilities to that clock, and carry `ivpa` through: an implied volatility is a volatility rather than a price, so it is carried, never converted.
 7. Put a caller's [`AssetPanel`](@ref) on the emitted clock with [`project_panel_clock`](@ref). A caller states a Panel Field on the clock of the table they hold, and the collapse is the only step here that renumbers it, so this is where it is projected: the aggregated period takes the values of the row at its representative timestamp, which is last-observation semantics and matches [`LastObservation`](@ref). A static panel has no observation axis and is carried through untouched.
 8. Read the **Listing Span** off the asset block with [`listing_span`](@ref), unless the caller declared one, in which case theirs is taken outright.
 9. Return the [`PricesResult`](@ref) carrying all of it.

# Arguments

  - `est`: The [`PriceIngestion`](@ref) estimator.
  - `X`: Asset prices, `observations × assets`.
  - `F`: Optional factor prices.
  - `B`: Optional benchmark prices, one column or one per asset.
  - `iv`: Optional implied volatilities, one column per asset.
  - `ivpa`: Optional implied volatility adjustment.
  - `pnl`: Optional [`AssetPanel`](@ref) of Panel Fields the caller already holds.
  - `pr`: A [`PricesResult`](@ref), for the second form, whose series are re-ingested.

# Validation

  - `!isempty(X)`. Raises an [`IsEmptyError`](@ref).
  - The asset, factor and benchmark column names are pairwise disjoint, and none of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref) naming the offending columns.
  - A declared `span` is `size(values(X))` after the join and the collapse. Raises a `DimensionMismatch`.
  - The emitted clock is a subset of `iv`'s timestamps, when `iv` is given. Raises an [`IsEmptyError`](@ref).

# Returns

  - `pr::PricesResult`: The price carrier, holding the Listing Span in its `span` field.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 4),
                     [100.0 missing; 102.0 103.0; 101.0 104.0; 103.0 102.0], [\"A\", \"B\"]);

julia> pr = price_ingestion(PriceIngestion(), X);

julia> values(pr.X)[1, :]
2-element Vector{Float64}:
 100.0
 NaN

julia> Matrix(pr.span)
4×2 Matrix{Bool}:
 1  0
 1  1
 1  1
 1  1
```

# Related

  - [`PriceIngestion`](@ref)
  - [`PricesResult`](@ref)
  - [`PricesToReturns`](@ref)
  - [`listing_span`](@ref)
  - [`unify_gaps`](@ref)
  - [`project_panel_clock`](@ref)
  - [`LastObservation`](@ref)
"""
function price_ingestion(est::PriceIngestion, X::TimeSeries.TimeArray;
                         F::Option{<:TimeSeries.TimeArray} = nothing,
                         B::Option{<:TimeSeries.TimeArray} = nothing,
                         iv::Option{<:TimeSeries.TimeArray} = nothing,
                         ivpa::Option{<:Num_VecNum} = nothing,
                         pnl::Option{<:AssetPanel} = nothing)::PricesResult
    @argcheck(!isempty(X),
              IsEmptyError("`X` cannot be empty: the ingestion layer states a universe from a price panel"))
    assert_distinct_series_names(X, F, B)
    nx = TimeSeries.colnames(X)
    M = unify_gaps(X)
    if !isnothing(F)
        M = TimeSeries.merge(M, unify_gaps(F); method = est.join_method)
    end
    if !isnothing(B)
        M = TimeSeries.merge(M, unify_gaps(B); method = est.join_method)
    end
    if !isempty(est.collapse_args)
        M = TimeSeries.collapse(M, est.collapse_args...)
    end
    Xa = M[nx]
    Fa = isnothing(F) ? nothing : M[TimeSeries.colnames(F)]
    Ba = isnothing(B) ? nothing : M[TimeSeries.colnames(B)]
    iva = if isnothing(iv)
        nothing
    else
        ts = TimeSeries.timestamp(Xa)
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)),
                  IsEmptyError("the implied volatilities are carried on the clock the ingestion emits, so that clock is a subset of theirs; $(length(setdiff(ts, TimeSeries.timestamp(iv)))) of the $(length(ts)) emitted observations are absent from iv"))
        iv[ts]
    end
    #! The collapse is the one step here that renumbers an observation, and a Panel Field is
    #! stated on the clock of the table the caller holds, so the projection is owed here
    #! rather than by the conversion: after the door, the carrier states one clock and
    #! everything it carries is on it.
    pnl = project_panel_clock(pnl, TimeSeries.timestamp(Xa), TimeSeries.timestamp(X),
                              string.(nx))
    span = isnothing(est.span) ? listing_span(values(Xa)) : est.span
    return PricesResult(; X = Xa, F = Fa, B = Ba, iv = iva, ivpa = ivpa, pnl = pnl,
                        span = span)
end
function price_ingestion(est::PriceIngestion, pr::PricesResult)::PricesResult
    return price_ingestion(est, pr.X; F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                           pnl = pr.pnl)
end
"""
    project_panel_clock(pnl::Nothing, ts_new, ts_old, nx::VecStr) -> nothing
    project_panel_clock(pnl::AssetPanel, ts_new, ts_old, nx::VecStr) -> AssetPanel

Put a caller's [`AssetPanel`](@ref) on the clock the ingestion emits.

A caller states a Panel Field on the clock of the table they hold, and the collapse is the one step of [`price_ingestion`](@ref) that renumbers an observation, so the projection is owed at the door: after it, the carrier states one clock and everything the carrier holds is on it. A static panel has no observation axis, so [`feature_row_indices`](@ref) answers `Colon()` for one and it rides through unchanged.

# Algorithm

 1. A carrier holding no panel projects to none.
 2. Otherwise check the panel against the asset axis and the incoming clock with [`check_asset_panel`](@ref).
 3. Recover the emitted clock's rows in the incoming one with [`feature_row_indices`](@ref), and view the panel over them with [`port_opt_view`](@ref). The asset axis is whole, and it is still named so that a square tensor Panel Field is cut on its label axis too.

# Arguments

  - `pnl`: The caller's [`AssetPanel`](@ref), or `nothing`.
  - `ts_new`: The timestamps the ingestion emits.
  - `ts_old`: The timestamps of the asset table the caller handed in.
  - `nx`: The asset names.

# Validation

  - The panel describes `length(nx)` assets and `length(ts_old)` observations. Raises a `DimensionMismatch`.

# Returns

  - `pnl′::Option{<:AssetPanel}`: The panel on the emitted clock, or `nothing`.

# Related

  - [`price_ingestion`](@ref)
  - [`AssetPanel`](@ref)
  - [`feature_row_indices`](@ref)
  - [`port_opt_view`](@ref)
  - [`check_asset_panel`](@ref)
  - [`LastObservation`](@ref)
"""
function project_panel_clock(::Nothing, ::Any, ::Any, ::VecStr)
    return nothing
end
function project_panel_clock(pnl::AssetPanel, ts_new, ts_old, nx::VecStr)::AssetPanel
    check_asset_panel(pnl, length(nx), length(ts_old), "the number of asset price columns")
    return port_opt_view(pnl, feature_row_indices(pnl, ts_new, ts_old),
                         collect(eachindex(nx)), nx)
end
"""
    span_carrier_view(span::Nothing, ts_new, ts_old, j) -> nothing
    span_carrier_view(span::AbstractMatrix{Bool}, ts_new, ts_old, j) -> SubArray
    span_carrier_view(span::ListingSpan, ts_new, ts_old, j)

View a price carrier's Listing Span over the surviving timestamps and the assets `j`, or return `nothing` when the carrier holds none.

The span is a statement about the **instruments**, so a window's span is the panel-wide one *viewed*, never one re-derived from the window: re-deriving reads a delisting that straddles the window end as an asset that was never listed, which is the divergence a panel-wide derivation exists to avoid. A view is exact and, over a [`PortfolioOptimisers.ListingSpan`](@ref), allocates no cell.

The span is held positionally parallel to the price clock, so its rows are recovered from the surviving timestamps the same way a time-varying [`AssetPanel`](@ref)'s are, by [`matched_row_indices`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: return `nothing`, without matching any timestamp.
 2. `span` is a matrix: recover its rows with [`matched_row_indices`](@ref), and view it at those rows and the assets `j`.
 3. `span` is a [`PortfolioOptimisers.ListingSpan`](@ref) and the recovered rows are the whole clock in order: subset the two bound vectors instead, so the two integers per asset survive the cut rather than being expanded into a view of booleans. Any other row selection can split an interval in half, which no interval can say, and falls back to step 2.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `ts_new`: Timestamps that survived the selection.
  - `ts_old`: Timestamps of the price clock the span is parallel to.
  - `j`: Asset index.

# Returns

  - A view of `span`, or `nothing`.

# Related

  - [`PricesResult`](@ref)
  - [`PortfolioOptimisers.ListingSpan`](@ref)
  - [`listing_span`](@ref)
  - [`matched_row_indices`](@ref)
  - [`port_opt_view`](@ref)
"""
function span_carrier_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function span_carrier_view(span::AbstractMatrix{Bool}, ts_new, ts_old, j)
    return view(span, matched_row_indices(ts_new, ts_old), j)
end
function span_carrier_view(span::ListingSpan, ts_new, ts_old, j)
    i = matched_row_indices(ts_new, ts_old)
    #! A row selection that keeps the whole clock in order leaves every interval intact,
    #! so the two integers per asset survive the cut. Any other selection can split one
    #! in half, which no interval can say, and falls back to the ordinary view.
    return if i == axes(span, 1)
        ListingSpan(span.first[j], span.last[j], span.n)
    else
        view(span, i, j)
    end
end
"""
    assert_span_shape(span::Nothing, nobs, na) -> nothing
    assert_span_shape(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer) -> nothing

Check that a Listing Span fits the price panel it rides on.

A span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices. A carrier that states none states no universe, and there is nothing to check.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: the carrier states no universe, so there is nothing to check.
 2. Otherwise check the span's shape against the price panel's.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `nobs`: Observation count of the price panel.
  - `na`: Asset count of the price panel.

# Validation

  - `size(span) == (nobs, na)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`prices_to_returns`](@ref)
  - [`PricesToReturns`](@ref)
  - [`price_ingestion`](@ref)
  - [`returns_universe_masks`](@ref)
"""
function assert_span_shape(::Nothing, ::Any, ::Any)::Nothing
    return nothing
end
function assert_span_shape(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer)::Nothing
    @argcheck(size(span) == (nobs, na),
              DimensionMismatch("a Listing Span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices; got size(span) = $(size(span)) and $nobs × $na prices"))
    return nothing
end
"""
    series_names(A::Nothing) -> Vector{Symbol}
    series_names(A::TimeSeries.TimeArray) -> Vector{Symbol}

Name the columns of one optional price table, and name none when the caller passed none.

A series the caller omitted contributes no column name, so it contributes an empty list rather than an `isnothing` branch at every site that reads one.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return an empty name list.
 2. `A` is a table: return its column names.

# Arguments

  - `A`: One price table, or `nothing`.

# Returns

  - `n::Vector{Symbol}`: The table's column names, empty when there is no table.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`price_ingestion`](@ref)
  - [`prices_to_returns`](@ref)
"""
function series_names(::Nothing)
    return Symbol[]
end
function series_names(A::TimeSeries.TimeArray)
    return TimeSeries.colnames(A)
end
"""
    assert_disjoint_series_names(a::AbstractVector{Symbol}, b::AbstractVector{Symbol}, na::String, nb::String) -> nothing

Refuse a column name that two of the price tables both carry.

A column name is what says which series a column came from: the asset, factor and benchmark tables are merged onto one clock, and the blocks are taken apart by name afterwards. `TimeSeries.merge` renames the second of two columns that share a name, appending `_1`, so a shared name silently makes one block take the other's column, and the minted name belongs to no block at all. A shared name is also not the conversion's to resolve — `X`'s `AAPL` and `F`'s `AAPL` are different series, and keeping either one is worse than refusing both.

# Arguments

  - `a`: Column names of the first table.
  - `b`: Column names of the second table.
  - `na`: How the caller spells the first table.
  - `nb`: How the caller spells the second table.

# Validation

  - `isdisjoint(a, b)`. Raises a [`ConflictingArgumentError`](@ref) naming the shared columns.

# Returns

  - `nothing`.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`assert_unreserved_series_names`](@ref)
  - [`series_names`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_disjoint_series_names(a::AbstractVector{Symbol}, b::AbstractVector{Symbol},
                                      na::String, nb::String)::Nothing
    shared = intersect(a, b)
    @argcheck(isempty(shared),
              ConflictingArgumentError("the asset, factor and benchmark series are joined onto one clock and a column name is what says which series a column came from, so `$na` and `$nb` cannot share one; both carry $(shared). Rename the colliding columns before the call."))
    return nothing
end
"""
    assert_unreserved_series_names(n::AbstractVector{Symbol}, nn::String) -> nothing

Refuse a series named after the observation clock.

The conversion writes the clock into a column named `timestamp`, and a series of the same name takes that column's place: the clock keeps the name and the series is renamed `timestamp_1`, so the block the series belongs to reads the dates as prices. The name is the layer's, and a caller holding a series of that name renames it.

# Arguments

  - `n`: Column names of one table.
  - `nn`: How the caller spells that table.

# Validation

  - `:timestamp ∉ n`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`assert_disjoint_series_names`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_unreserved_series_names(n::AbstractVector{Symbol}, nn::String)::Nothing
    @argcheck(:timestamp ∉ n,
              ConflictingArgumentError("the conversion writes the observation clock into a column named `timestamp`, so a series cannot carry that name; `$nn` does. Rename it before the call."))
    return nothing
end
"""
    assert_distinct_series_names(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray} = nothing, B::Option{<:TimeSeries.TimeArray} = nothing) -> nothing

Check that every price series reaching the layer can still be named after the join.

This is the door's check, and both doors take it: [`price_ingestion`](@ref) before it merges, and [`prices_to_returns`](@ref) before it does. What it buys is that every later piece may split the merged table by name — a name belongs to exactly one of the asset, factor and benchmark blocks, and none of them is the clock's.

Two series of one table cannot be checked here. Every `TimeSeries.TimeArray` constructor runs `TimeSeries.replace_dupes!` over its column names, so a table's own duplicates are renamed before the table exists and no duplicate reaches this function.

# Algorithm

 1. Read the three name lists, an absent table naming none, with [`series_names`](@ref).
 2. Refuse a name shared by two of them with [`assert_disjoint_series_names`](@ref), over all three pairs.
 3. Refuse the clock's own name in any of them with [`assert_unreserved_series_names`](@ref).

# Arguments

  - `X`: Asset prices, `observations × assets`.
  - `F`: Optional factor prices.
  - `B`: Optional benchmark prices.

# Validation

  - The asset, factor and benchmark names are pairwise disjoint. Raises a [`ConflictingArgumentError`](@ref) naming the shared columns.
  - None of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_disjoint_series_names`](@ref)
  - [`assert_unreserved_series_names`](@ref)
  - [`series_names`](@ref)
  - [`price_ingestion`](@ref)
  - [`prices_to_returns`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_distinct_series_names(X::TimeSeries.TimeArray,
                                      F::Option{<:TimeSeries.TimeArray} = nothing,
                                      B::Option{<:TimeSeries.TimeArray} = nothing)::Nothing
    nx = TimeSeries.colnames(X)
    nf = series_names(F)
    nb = series_names(B)
    assert_disjoint_series_names(nx, nf, "X", "F")
    assert_disjoint_series_names(nx, nb, "X", "B")
    assert_disjoint_series_names(nf, nb, "F", "B")
    assert_unreserved_series_names(nx, "X")
    assert_unreserved_series_names(nf, "F")
    assert_unreserved_series_names(nb, "B")
    return nothing
end
"""
    compress_all_true(amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool})

Store a pair of universe masks that is true everywhere as the constant it is.

A gapless ingestion states a universe in which every asset is listed at every observation and every return is finite, so both masks are `true` throughout. [`PortfolioOptimisers.AllTrueMask`](@ref) says exactly that in two integers, which is what makes *always emit a panel* cost `O(1)` rather than `observations × assets` bits. The estimation mask is a subset of the active one, so testing the estimation mask alone answers for both.

# Algorithm

 1. Every entry of `emsk` is `true`: return one [`PortfolioOptimisers.AllTrueMask`](@ref) as both masks.
 2. Otherwise return the two masks unchanged.

# Arguments

  - `amsk`: The active mask.
  - `emsk`: The estimation mask.

# Returns

  - `(amsk, emsk)`: The two masks, compressed when they admit it.

# Related

  - [`PortfolioOptimisers.AllTrueMask`](@ref)
  - [`universe_masks`](@ref)
  - [`returns_universe_masks`](@ref)
  - [`AssetPanel`](@ref)
"""
function compress_all_true(amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool})
    if !all(emsk)
        return amsk, emsk
    end
    msk = AllTrueMask(size(emsk, 1), size(emsk, 2))
    return msk, msk
end
"""
    returns_universe_masks(span::Nothing, R::AbstractMatrix)
    returns_universe_masks(span::AbstractMatrix{Bool}, R::AbstractMatrix)

Derive the two universe masks the returns carrier's [`AssetPanel`](@ref) states.

The layer's carrier arrives with a **Listing Span**, and this projects it onto the returns clock and intersects it with finiteness. A carrier built outside the layer arrives without one and states **no universe**, whether or not its prices hold a gap: a window-local derivation reads a delisting straddling the window end as an asset that was never listed, so it answers a question it cannot answer correctly. The gaps of such a carrier are still handled — with no panel the Coverage Universe reads finiteness alone — and `pnl === nothing` keeps its single meaning: the carrier was not built by the layer.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: return `nothing, nothing`.
 2. `span` is given: project and intersect it with [`universe_masks`](@ref), then compress the pair with [`compress_all_true`](@ref).

# Arguments

  - `span`: The Listing Span over the surviving price rows and assets, or `nothing`.
  - `R`: The returns panel the conversion produced, `observations × assets`.

# Returns

  - `(amsk, emsk)`: The two masks, or `nothing, nothing` when the carrier states no universe.

# Related

  - [`universe_masks`](@ref)
  - [`compress_all_true`](@ref)
  - [`attach_universe_masks`](@ref)
  - [`prices_to_returns`](@ref)
"""
function returns_universe_masks(::Nothing, ::AbstractMatrix)
    return nothing, nothing
end
function returns_universe_masks(span::AbstractMatrix{Bool}, R::AbstractMatrix)
    return compress_all_true(universe_masks(span, R)...)
end
"""
    attach_universe_masks(pnl, amsk::Nothing, emsk::Nothing)
    attach_universe_masks(pnl::Nothing, amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool}) -> AssetPanel
    attach_universe_masks(pnl::AssetPanel, amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool}) -> AssetPanel

Put the two universe masks onto the [`AssetPanel`](@ref) the returns carrier holds.

The layer emits **one carrier**, and the masks ride on it: a [`Pipeline`](@ref) step has one out-slot, and mask emission cannot leave [`PricesToReturns`](@ref) without landing on the price clock, where the masks are not stated. Carrying them makes [`port_opt_view`](@ref) slice the universe in step with the returns for free.

A caller's Panel Fields are kept. Only the masks are the layer's to state: an estimation mask is a statement about the *data* rather than about the instruments, so it is re-derived in every case and `emsk ⊆ amsk` holds by construction rather than by refusal.

# Algorithm

The method that Julia selects is the algorithm.

 1. No masks: return the panel unchanged. The carrier states no universe, so `pnl === nothing` keeps its one meaning — the carrier was not built by the layer.
 2. Masks and no panel: return an [`AssetPanel`](@ref) of the two masks and no Panel Field. This is the layer's common case: a caller holding only prices has no feature data.
 3. Masks and a panel: keep its Panel Fields and replace its masks. A static panel's fields carry no observation axis, so they are lifted onto the masks' clock with [`panel_field_lift`](@ref) first.

# Arguments

  - `pnl`: The Asset Panel the price carrier held, or `nothing`.
  - `amsk`: The active mask, or `nothing`.
  - `emsk`: The estimation mask, or `nothing`.

# Returns

  - `pnl′::Option{<:AssetPanel}`: The panel the returns carrier holds.

# Related

  - [`AssetPanel`](@ref)
  - [`returns_universe_masks`](@ref)
  - [`panel_field_lift`](@ref)
  - [`prices_to_returns`](@ref)
"""
function attach_universe_masks(pnl, ::Nothing, ::Nothing)
    return pnl
end
function attach_universe_masks(::Nothing, amsk::AbstractMatrix{Bool},
                               emsk::AbstractMatrix{Bool})::AssetPanel
    return AssetPanel(; amsk = amsk, emsk = emsk)
end
function attach_universe_masks(pnl::AssetPanel, amsk::AbstractMatrix{Bool},
                               emsk::AbstractMatrix{Bool})::AssetPanel
    pf = if panel_is_static(pnl)
        [panel_field_lift(f, size(amsk, 1)) for f in pnl.pf]
    else
        pnl.pf
    end
    return AssetPanel(; pf = pf, amsk = amsk, emsk = emsk)
end

export PriceIngestion, price_ingestion
