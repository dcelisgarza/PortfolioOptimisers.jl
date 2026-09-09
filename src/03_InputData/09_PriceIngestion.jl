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

Convert it with `PricesToReturns(; nan_to_missing = false)`: the layer spells an absent price `NaN` precisely so that no deletion step reads it, and a conversion that reads it as absent again deletes the gaps the span describes. The conversion warns when it is asked to.

# Algorithm

 1. Unify the absent-price convention of every series with [`unify_gaps`](@ref), so that a gap means one thing from here on.
 2. Join the factor and the benchmark series onto the asset clock under `join_method`. An outer join adds the rows one series has and another does not, padding them as gaps.
 3. Collapse the joined series to a lower frequency when `collapse_args` is non-empty, which renumbers every observation.
 4. Split the joined series back into their asset, factor and benchmark blocks, all now on one clock.
 5. Align the implied volatilities to that clock, and carry `ivpa` through: an implied volatility is a volatility rather than a price, so it is carried, never converted.
 6. Read the **Listing Span** off the asset block with [`listing_span`](@ref), unless the caller declared one, in which case theirs is taken outright.
 7. Return the [`PricesResult`](@ref) carrying all of it.

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
"""
function price_ingestion(est::PriceIngestion, X::TimeSeries.TimeArray;
                         F::Option{<:TimeSeries.TimeArray} = nothing,
                         B::Option{<:TimeSeries.TimeArray} = nothing,
                         iv::Option{<:TimeSeries.TimeArray} = nothing,
                         ivpa::Option{<:Num_VecNum} = nothing,
                         pnl::Option{<:AssetPanel} = nothing)::PricesResult
    @argcheck(!isempty(X),
              IsEmptyError("`X` cannot be empty: the ingestion layer states a universe from a price panel"))
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
    span = isnothing(est.span) ? listing_span(values(Xa)) : est.span
    return PricesResult(; X = Xa, F = Fa, B = Ba, iv = iva, ivpa = ivpa, pnl = pnl,
                        span = span)
end
function price_ingestion(est::PriceIngestion, pr::PricesResult)::PricesResult
    return price_ingestion(est, pr.X; F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                           pnl = pr.pnl)
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
    assert_span_convertible(span::Nothing, nobs, na, nan_to_missing::Bool, strict::Bool) -> nothing
    assert_span_convertible(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer,
                            nan_to_missing::Bool, strict::Bool) -> nothing

Check that a Listing Span fits the price panel it rides on, and that the conversion will not delete the gaps it describes.

The ingestion layer spells an absent price `NaN` precisely so that no deletion step reads it. `nan_to_missing = true` reads it as absent again and deletes the observation or the asset that holds one, so the universe the emitted [`AssetPanel`](@ref) states is the one that *survives the deletion* rather than the one the span describes — an all-true panel over a table whose gaps were the point. That is a silently wrong answer rather than a broken one, so it warns by name, and refuses under `strict`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: the carrier states no universe, so there is nothing to check.
 2. Otherwise check the span's shape against the price panel's, and then the conversion's own convention against the span's purpose.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `nobs`: Observation count of the price panel.
  - `na`: Asset count of the price panel.
  - `nan_to_missing`: Whether the conversion reads a `NaN` price as absent and deletes it.
  - `strict`: Whether the contradiction is refused rather than warned about.

# Validation

  - `size(span) == (nobs, na)`. Raises a `DimensionMismatch`.
  - `nan_to_missing` is `false` under `strict`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`prices_to_returns`](@ref)
  - [`PricesToReturns`](@ref)
  - [`price_ingestion`](@ref)
  - [`returns_universe_masks`](@ref)
"""
function assert_span_convertible(::Nothing, ::Any, ::Any, ::Bool, ::Bool)::Nothing
    return nothing
end
function assert_span_convertible(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer,
                                 nan_to_missing::Bool, strict::Bool)::Nothing
    @argcheck(size(span) == (nobs, na),
              DimensionMismatch("a Listing Span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices; got size(span) = $(size(span)) and $nobs × $na prices"))
    if !nan_to_missing
        return nothing
    end
    msg = "the price carrier states a Listing Span, and nan_to_missing = true reads its gaps as absent prices and deletes them, so the universe the emitted Asset Panel states is the one that survives the deletion rather than the one the span describes. Convert with PricesToReturns(; nan_to_missing = false)."
    if strict
        throw(ConflictingArgumentError(msg))
    end
    @warn(msg)
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
    returns_universe_masks(span::Option{<:AbstractMatrix{Bool}}, P::AbstractMatrix,
                           R::AbstractMatrix, strict::Bool)

Derive the two universe masks the returns carrier's [`AssetPanel`](@ref) states.

The layer's carrier arrives with a **Listing Span**, and this projects it onto the returns clock and intersects it with finiteness. A carrier built outside the layer arrives without one: if its prices hold no gap there is no universe to state and the conversion emits no panel, and if they do the span is derived **window-locally**, which warns because a delisting straddling the window end reads as an asset that was never listed.

# Algorithm

 1. `span` is given: project and intersect it with [`universe_masks`](@ref).
 2. `span` is `nothing` and `P` holds no gap: return `nothing, nothing`. The carrier was not built by the layer and states no universe.
 3. `span` is `nothing` and `P` holds a gap: raise under `strict`, warn otherwise, derive the span from `P` with [`listing_span`](@ref), and project it.
 4. Compress the pair with [`compress_all_true`](@ref).

# Arguments

  - `span`: The Listing Span over the surviving price rows and assets, or `nothing`.
  - `P`: The surviving price panel the conversion read, `price observations × assets`.
  - `R`: The returns panel the conversion produced, `observations × assets`.
  - `strict`: Whether a carrier holding gaps and no span is refused rather than warned about.

# Validation

  - `span` is not `nothing` when `P` holds a gap and `strict` is `true`. Raises an [`IsNothingError`](@ref).

# Returns

  - `(amsk, emsk)`: The two masks, or `nothing, nothing` when the carrier states no universe.

# Related

  - [`universe_masks`](@ref)
  - [`listing_span`](@ref)
  - [`compress_all_true`](@ref)
  - [`attach_universe_masks`](@ref)
  - [`prices_to_returns`](@ref)
"""
function returns_universe_masks(span::Option{<:AbstractMatrix{Bool}}, P::AbstractMatrix,
                                R::AbstractMatrix, strict::Bool)
    if isnothing(span)
        if !any(is_missing_value, P)
            return nothing, nothing
        end
        msg = "the price carrier holds gaps and no Listing Span, so the universe they imply is derived from this window alone. A window-local derivation reads a delisting that straddles the window end as an asset that was never listed, which is why the panel-wide derivation exists. Build the carrier with price_ingestion(PriceIngestion(), X), or declare a listing calendar as its span."
        if strict
            throw(IsNothingError(msg))
        end
        @warn(msg)
        span = listing_span(P)
    end
    amsk, emsk = universe_masks(span, R)
    return compress_all_true(amsk, emsk)
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
