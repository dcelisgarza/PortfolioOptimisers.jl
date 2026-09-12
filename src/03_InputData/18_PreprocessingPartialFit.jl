"""
    vcat_carrier_rows(a::PricesResult, b::PricesResult)
    vcat_carrier_rows(a::ReturnsResult, b::ReturnsResult)

Concatenates the observations of two carriers of the same universe, `a` first.

The row-concatenation every online form of a data step reads: [`PricesToReturns`](@ref) puts the last price row it kept in front of a new block so the conversion reads consecutive prices, and the input-carrier buffer `Online(pipe)` seeds appends every block it is handed. The universe is pinned by `a` — the asset, factor and benchmark names, the implied-volatility adjustment, a static [`AssetPanel`](@ref) — and a block carrying a different one is refused by name, as [`ReturnsBufferState`](@ref)'s step refuses one. A column one carrier holds and the other does not is refused too: a factor series that comes and goes is not one series.

A time-varying panel is concatenated mask by mask. Its Panel Fields are refused, as [`step_active_mask`](@ref) refuses them: a time-varying field's rows are sample, and no concatenation of a categorical or a tensor field exists here.

# Arguments

  - `a`: The earlier observations.
  - `b`: The later observations.

# Validation

  - The names, the implied-volatility adjustment and a static panel agree. An `ArgumentError` is thrown otherwise.
  - Every optional column is held by both carriers or by neither. An `ArgumentError` is thrown otherwise.
  - A time-varying panel holds no Panel Field. An `ArgumentError` is thrown otherwise.

# Returns

  - `c`: A carrier of the same type as `a`, holding the observations of `a` followed by those of `b`.

# Related

  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`vcat_panel_rows`](@ref)
  - [`partial_fit!`](@ref)
"""
function vcat_carrier_rows(a::PricesResult, b::PricesResult)
    assert_pinned_carrier(a.ivpa, b.ivpa, :ivpa)
    return PricesResult(; X = vcat_optional(a.X, b.X, :X), F = vcat_optional(a.F, b.F, :F),
                        B = vcat_optional(a.B, b.B, :B),
                        iv = vcat_optional(a.iv, b.iv, :iv), ivpa = a.ivpa,
                        pnl = vcat_panel_rows(a.pnl, b.pnl),
                        span = vcat_optional(a.span, b.span, :span))
end
function vcat_carrier_rows(a::ReturnsResult, b::ReturnsResult)
    assert_pinned_carrier(a.nx, b.nx, :nx)
    assert_pinned_carrier(a.nf, b.nf, :nf)
    assert_pinned_carrier(a.nb, b.nb, :nb)
    assert_pinned_carrier(a.ivpa, b.ivpa, :ivpa)
    return ReturnsResult(; nx = a.nx, X = vcat_optional(a.X, b.X, :X), nf = a.nf,
                         F = vcat_optional(a.F, b.F, :F), nb = a.nb,
                         B = vcat_optional(a.B, b.B, :B),
                         ts = vcat_optional(a.ts, b.ts, :ts),
                         iv = vcat_optional(a.iv, b.iv, :iv), ivpa = a.ivpa,
                         pnl = vcat_panel_rows(a.pnl, b.pnl))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a block whose pinned context differs from the first block's, naming the field.

# Related

  - [`vcat_carrier_rows`](@ref)
  - [`assert_pinned_context`](@ref)
"""
function assert_pinned_carrier(pinned, given, name::Symbol)::Nothing
    @argcheck(isequal(pinned, given),
              ArgumentError("the `$name` of a block of observations must equal the one the first block pinned, because a carrier that silently took a new one would put the next observation on the wrong column. Got $(pinned_repr(given)) against the pinned $(pinned_repr(pinned))."))
    return nothing
end
"""
    vcat_optional(a::Nothing, b::Nothing, name::Symbol)
    vcat_optional(a, b, name::Symbol)

Concatenates one optional column of two carriers along the observation axis, or refuses a column one holds and the other does not.

# Related

  - [`vcat_carrier_rows`](@ref)
"""
function vcat_optional(::Nothing, ::Nothing, ::Symbol)
    return nothing
end
function vcat_optional(a, b, name::Symbol)
    @argcheck(!isnothing(a) && !isnothing(b),
              ArgumentError("the `$name` column is held by $(isnothing(a) ? "the later" : "the earlier") block of observations and not by the other, so the two cannot be one series: a column is present at every observation or at none."))
    return vcat(a, b)
end
"""
    vcat_panel_rows(a::Nothing, b::Nothing)
    vcat_panel_rows(a::AssetPanel, b::AssetPanel)

Concatenates the [`AssetPanel`](@ref)s of two blocks along the observation axis.

A static panel carries no observation axis, so the two must agree and the first is kept. A time-varying panel is concatenated mask by mask, and one carrying a Panel Field is refused: a time-varying field's rows are sample, and this seam holds masks alone.

# Related

  - [`vcat_carrier_rows`](@ref)
  - [`AssetPanel`](@ref)
  - [`step_active_mask`](@ref)
"""
function vcat_panel_rows(::Nothing, ::Nothing)
    return nothing
end
function vcat_panel_rows(a::AssetPanel, b::AssetPanel)
    if panel_is_static(a) || panel_is_static(b)
        assert_pinned_carrier(a, b, :pnl)
        return a
    end
    @argcheck(isempty(a.pf) && isempty(b.pf),
              ArgumentError("a time-varying Asset Panel is concatenated through its masks alone, and this one holds $(length(a.pf) + length(b.pf)) Panel Field(s): a time-varying field's rows are sample, and the online step has nowhere to keep them. Drop the fields from the panel handed to the step, or fit in batch."))
    return AssetPanel(; amsk = vcat(a.amsk, b.amsk), emsk = vcat(a.emsk, b.emsk))
end
function vcat_panel_rows(a, b)
    return throw(ArgumentError("an Asset Panel is held by $(isnothing(a) ? "the later" : "the earlier") block of observations and not by the other, so the two did not come from one ingestion."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Counts the observations of a carrier.

# Related

  - [`vcat_carrier_rows`](@ref)
"""
carrier_rows(pr::PricesResult) = length(TimeSeries.timestamp(pr.X))
carrier_rows(rd::ReturnsResult) = size(rd.X, 1)
"""
    partial_fit_transform(est, data) -> (est′, data′)

Folds a block of observations into a data step and emits the block the step's transform gives it.

The online form of a preprocessing step that changes the rows it is handed, decided by [ADR 0142](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/adr/0142-a-pipeline-is-a-host-its-steps-fold-or-defer-to-a-view-and-a-step-with-no-online-form-is-refused-unless-the-pipeline-declares-a-refit.md). A [`Pipeline`](@ref) walking its steps hands each one the block it received from the step before and takes back the block the step emits, so a step's fold and its transform are one call: the transform of the new rows reads the state the earlier rows left, and the state is advanced past them. The two halves are inseparable — the first return of a new block is computed from the last price row the step kept — so the verb returns both.

[`partial_fit!`](@ref) on a data step is this verb's first element, and a step's read-out is [`fit_preprocessing`](@ref) with no data. A caller's own preprocessing estimator joins the host route by writing those three: this verb, the read-out, and [`supports_partial_fit`](@ref) answering `true`.

# Arguments

  - `est`: The data step, carrying its state in `cache` or `nothing` before the first block.
  - `data`: The block of observations, as the step's batch verb takes it.

# Returns

  - `(est′, data′)`: The step with the block folded, and the block as the step emits it.

# Related

  - [`partial_fit!`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`supports_partial_fit`](@ref)
  - [`PricesToReturns`](@ref)
  - [`PriceGapFill`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function partial_fit_transform(est::AbstractPreprocessingEstimator, ::Prices_RR)
    return throw(ArgumentError("a `$(typeof(est).name.name)` has no online form: no `partial_fit_transform` method folds a block of observations into it. Give the step a `partial_fit_transform` and a data-less `fit_preprocessing` read-out, or declare a refit with `Online(pipe)`."))
end
function partial_fit!(est::Union{<:PricesToReturns, <:PriceGapFill, <:MissingDataFilter},
                      data::Prices_RR)
    return partial_fit_transform(est, data)[1]
end
#! Begin: PricesToReturns.
"""
$(DocStringExtensions.TYPEDEF)

The state a [`PricesToReturns`](@ref) keeps between two blocks of prices, so that the next block converts exactly as the whole history would.

A return reads two consecutive prices, so the conversion of a new block needs the last price row of the block before it, and nothing else of that block: `tail` is that row, kept as a one-row [`PricesResult`](@ref) so the span, the panel and the implied volatilities of the row ride with the prices. A [`CatchUpGapReturn`](@ref) reads further back — the observation that ends a gap books the move against the last observed price, which may lie any number of blocks behind — so `anchor` keeps that price per series column, `NaN` where none has been observed. It is `nothing` under the default rule, which reads no earlier price.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturnsState(; tail::PricesResult, anchor::Option{<:AbstractVector} = nothing)

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`PricesToReturns`](@ref)
  - [`partial_fit_transform`](@ref)
"""
@concrete struct PricesToReturnsState <: AbstractPartialFitState
    """
    The last price row folded, as a one-row [`PricesResult`](@ref).
    """
    tail
    """
    The last observed price of every series column — the assets, then the factors, then the benchmark — or `NaN` where none has been observed. `nothing` under the default gap rule.
    """
    anchor
end
function PricesToReturnsState(; tail::PricesResult,
                              anchor::Option{<:AbstractVector} = nothing)::PricesToReturnsState
    @argcheck(carrier_rows(tail) == 1,
              DimensionMismatch("the tail of a PricesToReturnsState is the last price row folded, so it holds one observation; got $(carrier_rows(tail))"))
    return PricesToReturnsState(tail, anchor)
end
function Base.copy(x::PricesToReturnsState)
    return PricesToReturnsState(x.tail, isnothing(x.anchor) ? nothing : copy(x.anchor))
end
function merge_states(a::PricesToReturnsState, b::PricesToReturnsState)
    anchor = if isnothing(a.anchor)
        nothing
    else
        [is_missing_value(y) ? x : y for (x, y) in zip(a.anchor, b.anchor)]
    end
    return PricesToReturnsState(; tail = b.tail, anchor = anchor)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lays the series columns of a price carrier side by side, in the order [`prices_to_returns`](@ref) converts them.

# Related

  - [`PricesToReturnsState`](@ref)
  - [`prices_to_returns`](@ref)
"""
function series_values(pr::PricesResult)
    cols = Any[values(unify_gaps(pr.X))]
    if !isnothing(pr.F)
        push!(cols, values(unify_gaps(pr.F)))
    end
    if !isnothing(pr.B)
        push!(cols, values(unify_gaps(pr.B)))
    end
    return reduce(hcat, cols)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Advances the per-column anchor of a [`PricesToReturnsState`](@ref) past a block: the last observed price of each series column, or the one carried when the block observes none.

# Related

  - [`PricesToReturnsState`](@ref)
"""
function advance_anchor(anchor::AbstractVector, P::AbstractMatrix)
    out = copy(anchor)
    for j in axes(P, 2)
        t = findlast(x -> !is_missing_value(x), view(P, :, j))
        if !isnothing(t)
            out[j] = P[t, j]
        end
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the cells a Gap Return algorithm writes in the returns of a new block, reading the anchor of the state for the price before the block.

The batch conversion hands [`gap_return`](@ref) a whole column; a block hands it the column extended by two rows in front, the anchor and the last price row kept, so the algorithm's walk meets the observed price a gap's move is booked against and the immediate predecessor of the block's first row exactly as it would in the whole history. Only the cells of the new rows are copied back, and only those [`gap_return_writable`](@ref) admits over the extended column, which is the batch invariant restated: an anchor is an earlier observed price, and a column with none admits nothing before its first observation.

# Arguments

  - `alg`: The Gap Return algorithm.
  - `R`: The returns of the new block, `observations × series`, mutated in place.
  - `P`: The prices of the new block, `observations × series`.
  - `anchor`: The last observed price per series column before the block.
  - `last`: The last price row before the block, `1 × series`.
  - `ret_method`: `:simple` or `:log`.

# Returns

  - `R`: The same matrix, with the writable cells of the new rows resolved.

# Related

  - [`gap_return`](@ref)
  - [`gap_return_writable`](@ref)
  - [`PricesToReturnsState`](@ref)
"""
function block_gap_return!(alg::AbstractGapReturnAlgorithm, R::AbstractMatrix,
                           P::AbstractMatrix, anchor::AbstractVector, last::AbstractMatrix,
                           ret_method::Symbol)
    for j in axes(P, 2)
        p = vcat(anchor[j], last[1, j], view(P, :, j))
        r = vcat(oftype(R[1, j], NaN), view(R, :, j))
        w = gap_return_writable(p, r)
        if !any(w)
            continue
        end
        out = gap_return(alg, p, r, ret_method)
        @argcheck(length(out) == length(r), DimensionMismatch)
        for t in axes(R, 1)
            if w[t + 1]
                R[t, j] = out[t + 1]
            end
        end
    end
    return R
end
"""
    partial_fit_transform(ptr::PricesToReturns, pr::PricesResult) -> (ptr′, rd)

Converts a block of prices to returns as the whole history would, and keeps what the next block needs.

# Algorithm

 1. On the first block, convert it alone with [`prices_to_returns`](@ref); the block is the history.
 2. On every later block, put the kept last price row in front of it with [`vcat_carrier_rows`](@ref), convert the window under the step's own `ret_method` and `padding` and the default gap rule, and keep the last `n` return rows, `n` the block's observations: the extra row is the seam, and `padding` adds one row the same way in the window and in the history.
 3. Under a [`CatchUpGapReturn`](@ref), resolve the writable cells of the new rows with [`block_gap_return!`](@ref), which extends every column by the anchor and the kept row so the algorithm meets the same neighbours it meets in the history.
 4. Keep the block's last price row as the new tail, and advance the anchor past the block.

The result is exact: every return of the emitted rows is the one the batch conversion of the whole history writes at that row, because a return reads two consecutive prices and a Gap Return reads the last observed one, and the state carries both.

# Related

  - [`partial_fit_transform`](@ref)
  - [`PricesToReturnsState`](@ref)
  - [`prices_to_returns`](@ref)
"""
function partial_fit_transform(ptr::PricesToReturns, pr::PricesResult)
    n = carrier_rows(pr)
    @argcheck(n > 0, IsEmptyError("a block of prices holds at least one observation"))
    P = series_values(pr)
    state = ptr.cache
    rd = if isnothing(state)
        prices_to_returns(ptr, pr)
    else
        window = vcat_carrier_rows(state.tail, pr)
        plain = PricesToReturns(; ret_method = ptr.ret_method, padding = ptr.padding)
        rw = prices_to_returns(plain, window)
        m = size(rw.X, 1)
        rd = port_opt_view(rw, (m - n + 1):m, :)
        if isnothing(ptr.gap_return_alg)
            rd
        else
            R = series_values_returns(rd)
            block_gap_return!(ptr.gap_return_alg, R, P, state.anchor,
                              series_values(state.tail), ptr.ret_method)
            returns_with_series(rd, R, ptr.gap_return_alg)
        end
    end
    tail = port_opt_view(pr, n:n, :)
    anchor = if isnothing(ptr.gap_return_alg)
        nothing
    else
        advance_anchor(if isnothing(state)
                           fill(oftype(P[1], NaN), size(P, 2))
                       else
                           state.anchor
                       end, P)
    end
    return rebuild_estimator(ptr,
                             (;
                              cache = PricesToReturnsState(; tail = tail, anchor = anchor))),
           rd
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lays the series columns of a returns carrier side by side, in the order of [`series_values`](@ref): the assets, then the factors, then the benchmark. The matrix is a copy.

# Related

  - [`series_values`](@ref)
  - [`returns_with_series`](@ref)
"""
function series_values_returns(rd::ReturnsResult)
    cols = Any[Matrix(rd.X)]
    if !isnothing(rd.F)
        push!(cols, Matrix(rd.F))
    end
    if !isnothing(rd.B)
        push!(cols, isa(rd.B, AbstractVector) ? reshape(collect(rd.B), :, 1) : Matrix(rd.B))
    end
    return reduce(hcat, cols)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuilds a returns carrier from the series matrix [`series_values_returns`](@ref) laid out, re-deriving the estimation mask the way the conversion does, because a Gap Return writes a finite return where the conversion left none.

# Related

  - [`series_values_returns`](@ref)
  - [`returns_universe_masks`](@ref)
"""
function returns_with_series(rd::ReturnsResult, R::AbstractMatrix,
                             ::AbstractGapReturnAlgorithm)
    N = length(rd.nx)
    X = R[:, 1:N]
    k = N
    F = if isnothing(rd.F)
        nothing
    else
        nf = size(rd.F, 2)
        k += nf
        R[:, (N + 1):(N + nf)]
    end
    B = if isnothing(rd.B)
        nothing
    elseif isa(rd.B, AbstractVector)
        R[:, k + 1]
    else
        R[:, (k + 1):(k + size(rd.B, 2))]
    end
    pnl = rd.pnl
    if !isnothing(pnl) && !panel_is_static(pnl)
        amsk, emsk = compress_all_true(Matrix(pnl.amsk), Matrix(pnl.amsk) .& isfinite.(X))
        pnl = AssetPanel(; pf = pnl.pf, amsk = amsk, emsk = emsk)
    end
    return ReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = F, nb = rd.nb, B = B,
                         ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa, pnl = pnl)
end
"""
    fit_preprocessing(ptr::PricesToReturns)

Reads a stepped [`PricesToReturns`](@ref) out: the conversion is stateless to a reader, so its fitted object is the estimator with the state dropped, exactly what the batch fit returns.

# Related

  - [`partial_fit_transform`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function fit_preprocessing(ptr::PricesToReturns)
    return rebuild_estimator(ptr, (; cache = nothing))
end
function supports_partial_fit(ptr::PricesToReturns)
    return isnothing(ptr.gap_return_alg) || isa(ptr.gap_return_alg, CatchUpGapReturn)
end
#! End: PricesToReturns.
#! Begin: PriceGapFill.
"""
$(DocStringExtensions.TYPEDEF)

The state a [`PriceGapFill`](@ref) with a [`CarriedPrice`](@ref) keeps between two blocks: the last observed price of every asset, which is the carry the next block's gaps take and the seed the fitted result replays.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceGapFillState(; nx::AbstractVector{Symbol}, v::AbstractVector)

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
"""
@concrete struct PriceGapFillState <: AbstractPartialFitState
    """
    Names of the asset columns, pinned by the first block.
    """
    nx
    """
    The last observed price per asset, `missing` where none has been observed.
    """
    v
end
function PriceGapFillState(; nx::AbstractVector{Symbol},
                           v::AbstractVector)::PriceGapFillState
    @argcheck(length(nx) == length(v), DimensionMismatch)
    return PriceGapFillState(nx, v)
end
function Base.copy(x::PriceGapFillState)
    return PriceGapFillState(copy(x.nx), copy(x.v))
end
function merge_states(a::PriceGapFillState, b::PriceGapFillState)
    assert_pinned_carrier(a.nx, b.nx, :nx)
    return PriceGapFillState(; nx = a.nx,
                             v = [ismissing(y) ? x : y for (x, y) in zip(a.v, b.v)])
end
"""
    partial_fit_transform(est::PriceGapFill, pr::PricesResult) -> (est′, pr′)

Fills the gaps of a block of prices as the fit over the whole history would, and carries the last observed prices forward.

# Algorithm

 1. Resolve the Listing Span bounding the fill with [`gap_fill_span`](@ref), as the batch replay does.
 2. Seed each column's walk. An asset observed in an earlier block is seeded with the carried price. One first observed in this block is seeded with its last observed price of the block, which is what the batch fit of [`PriceGapFill`](@ref) seeds a window with. One observed nowhere yet is left alone.
 3. Walk each seeded column with [`gap_fill_column!`](@ref) under the step's convention.
 4. Advance the carry to each column's last observed price of the block.

The convention must be [`CarriedPrice`](@ref); a statistic fill re-prices every earlier gap when the window grows and has no online form, which [`supports_partial_fit`](@ref) states and the Pipeline refuses at warm-up.

# Related

  - [`partial_fit_transform`](@ref)
  - [`PriceGapFillState`](@ref)
  - [`PriceGapFill`](@ref)
"""
function partial_fit_transform(est::PriceGapFill, pr::PricesResult)
    @argcheck(isa(est.fill, CarriedPrice),
              ArgumentError("a `PriceGapFill` with a `$(typeof(est.fill))` fill has no online form: a statistic fitted over the window re-prices every earlier gap when the window grows, so no state folded at one step can be corrected at the next. Use `PriceGapFill(CarriedPrice())`, or declare a refit with `Online(pipe)`."))
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(unify_gaps(pr.X)))
    state = est.cache
    v = if isnothing(state)
        Vector{Union{Missing, eltype(vals)}}(missing, length(names))
    else
        assert_pinned_carrier(state.nx, names, :nx)
        state.v
    end
    span = gap_fill_span(carrier_listing_span(pr), vals, est.strict)
    vnew = copy(v)
    for j in axes(vals, 2)
        t = findlast(x -> !is_missing_value(x), view(vals, :, j))
        seed = if !ismissing(v[j])
            v[j]
        elseif !isnothing(t)
            vals[t, j]
        else
            continue
        end
        if !isnothing(t)
            vnew[j] = vals[t, j]
        end
        gap_fill_column!(est.fill, vals, span, j, seed)
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, TimeSeries.colnames(pr.X))
    out = PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                       pnl = pr.pnl, span = pr.span)
    return rebuild_estimator(est, (; cache = PriceGapFillState(; nx = names, v = vnew))),
           out
end
"""
    fit_preprocessing(est::PriceGapFill)

Reads a stepped [`PriceGapFill`](@ref) out as the [`PriceGapFillResult`](@ref) the batch fit over the same rows gives: the assets observed so far, each with its last observed price.

# Related

  - [`partial_fit_transform`](@ref)
  - [`PriceGapFillState`](@ref)
"""
function fit_preprocessing(est::PriceGapFill)
    state = partial_fit_cache(est)
    keep = findall(!ismissing, state.v)
    v = identity.([state.v[j] for j in keep])
    return PriceGapFillResult(state.nx[keep], v, est.fill, est.strict)
end
function supports_partial_fit(est::PriceGapFill)
    return isa(est.fill, CarriedPrice)
end
#! End: PriceGapFill.
#! Begin: MissingDataFilter.
"""
$(DocStringExtensions.TYPEDEF)

The state a [`MissingDataFilter`](@ref) keeps between two blocks: the observations counted and the missing count per asset, from which the column filter is read out.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilterState(; nx::AbstractVector{Symbol}, miss::AbstractVector{<:Integer}, n::Integer)

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`MissingDataFilterResult`](@ref)
"""
@concrete struct MissingDataFilterState <: AbstractPartialFitState
    """
    Names of the asset columns, pinned by the first block.
    """
    nx
    """
    Missing observations per asset, over every block folded.
    """
    miss
    """
    Observations folded.
    """
    n
end
function MissingDataFilterState(; nx::AbstractVector{Symbol},
                                miss::AbstractVector{<:Integer},
                                n::Integer)::MissingDataFilterState
    @argcheck(length(nx) == length(miss), DimensionMismatch)
    assert_nonneg(n, :n)
    return MissingDataFilterState(nx, miss, n)
end
function Base.copy(x::MissingDataFilterState)
    return MissingDataFilterState(copy(x.nx), copy(x.miss), x.n)
end
function merge_states(a::MissingDataFilterState, b::MissingDataFilterState)
    assert_pinned_carrier(a.nx, b.nx, :nx)
    return MissingDataFilterState(; nx = a.nx, miss = a.miss .+ b.miss, n = a.n + b.n)
end
"""
    partial_fit_transform(mdf::MissingDataFilter, pr::PricesResult) -> (mdf′, pr)

Counts the gaps of a block of prices and passes the block through unchanged.

The column filter is universe-only — a longer window changes which assets survive and nothing else — so it is deferred to the read-out, where [`fit_preprocessing`](@ref) with no data answers the [`MissingDataFilterResult`](@ref) of the whole history and the Pipeline applies it as a view. The row filter at `row_thr = 1` drops nothing, so every row passes; a `row_thr < 1` has no online form, which [`supports_partial_fit`](@ref) states and the Pipeline refuses at warm-up.

# Related

  - [`partial_fit_transform`](@ref)
  - [`MissingDataFilterState`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function partial_fit_transform(mdf::MissingDataFilter, pr::PricesResult)
    @argcheck(isone(mdf.row_thr),
              ArgumentError("a `MissingDataFilter` with `row_thr = $(mdf.row_thr)` has no online form: a row kept at one step is dropped at the next when a column leaves the fitted universe, and the returns either side of it move. Use `row_thr = 1`, or declare a refit with `Online(pipe)`."))
    names = TimeSeries.colnames(pr.X)
    vals = values(pr.X)
    miss = vec(count(is_missing_value, vals; dims = 1))
    state = mdf.cache
    state = if isnothing(state)
        MissingDataFilterState(; nx = names, miss = miss, n = size(vals, 1))
    else
        assert_pinned_carrier(state.nx, names, :nx)
        MissingDataFilterState(; nx = names, miss = state.miss .+ miss,
                               n = state.n + size(vals, 1))
    end
    return rebuild_estimator(mdf, (; cache = state)), pr
end
"""
    fit_preprocessing(mdf::MissingDataFilter)

Reads a stepped [`MissingDataFilter`](@ref) out as the [`MissingDataFilterResult`](@ref) the batch fit over the same rows gives.

# Related

  - [`partial_fit_transform`](@ref)
  - [`MissingDataFilterState`](@ref)
"""
function fit_preprocessing(mdf::MissingDataFilter)
    state = partial_fit_cache(mdf)
    keep = state.miss / state.n .<= mdf.col_thr
    @argcheck(any(keep),
              IsEmptyError("MissingDataFilter with col_thr = $(mdf.col_thr) drops every asset over the observations folded"))
    return MissingDataFilterResult(state.nx[keep], mdf.row_thr)
end
function supports_partial_fit(mdf::MissingDataFilter)
    return isone(mdf.row_thr)
end
#! End: MissingDataFilter.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a data step but its `cache`, which appears only where a state is set, so no rendering of a step that took no step moves. ADR 0105 records the decision.

# Related

  - [`show_fields`](@ref)
"""
function show_fields(est::Union{<:PricesToReturns, <:PriceGapFill, <:MissingDataFilter})
    fns = fieldnames(typeof(est))
    return isnothing(est.cache) ? filter(!=(:cache), fns) : fns
end
