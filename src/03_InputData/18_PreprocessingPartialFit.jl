"""
    vcat_carrier_rows(a::PricesResult, b::PricesResult) -> PricesResult
    vcat_carrier_rows(a::ReturnsResult, b::ReturnsResult) -> ReturnsResult

Concatenates the observations of two carriers of one universe, `a` first.

Each online form of a data step reads this concatenation. [`PricesToReturns`](@ref) puts the last price row it kept in front of a new block, so that the conversion reads consecutive prices. The input-carrier buffer of `Online(pipe)` appends each block that it receives. The carrier `a` pins the universe: the names of the assets, the factors and the benchmark, the implied-volatility adjustment, and a static [`AssetPanel`](@ref). The function refuses a block `b` with a different universe, and the message names the field, as the step of [`ReturnsBufferState`](@ref) does. It also refuses a column that one carrier holds and the other does not, because a factor series that some blocks hold and others do not is not one series.

The function concatenates a time-varying panel mask by mask, and refuses its Panel Fields, as [`step_active_mask`](@ref) does. The rows of a time-varying field are sample data, and this function concatenates masks only.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`PricesResult`](@ref): check that the implied-volatility adjustment `ivpa` of `b` equals the one of `a`. Concatenate the prices `X`, the factors `F`, the benchmark `B` and the implied volatilities `iv` with [`vcat_optional`](@ref), which also checks the column names. Concatenate the panel with [`vcat_panel_rows`](@ref), and the Listing Span `span` with [`vcat_optional`](@ref).
 2. [`ReturnsResult`](@ref): check that the names `nx`, `nf` and `nb` and the adjustment `ivpa` of `b` equal the ones of `a`. Concatenate `X`, `F`, `B`, the timestamps `ts` and `iv` with [`vcat_optional`](@ref), and the panel with [`vcat_panel_rows`](@ref).

# Arguments

  - `a`: The earlier observations.
  - `b`: The later observations.

# Validation

  - The names, the implied-volatility adjustment and a static panel of `b` equal the ones of `a`. An `ArgumentError` is thrown otherwise.
  - Each optional column is held by both carriers or by neither. An `ArgumentError` is thrown otherwise.
  - A time-varying panel holds no Panel Field. An `ArgumentError` is thrown otherwise.

# Returns

  - `c`: A carrier of the type of `a`, with the observations of `a` followed by the observations of `b`.

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

Refuses a block whose pinned value differs from the value that the first block pinned, and names the field.

# Arguments

  - `pinned`: The value that the first block pinned.
  - `given`: The value that the new block carries.
  - `name`: The name of the field, which the message quotes.

# Validation

  - `isequal(pinned, given)`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

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
    vcat_optional(a::Nothing, b::Nothing, name::Symbol) -> Nothing
    vcat_optional(a::TimeSeries.TimeArray, b::TimeSeries.TimeArray, name::Symbol) -> TimeSeries.TimeArray
    vcat_optional(a, b, name::Symbol)

Concatenates one optional column of two carriers along the observation axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. `a` and `b` are `nothing`: return `nothing`.
 2. `a` and `b` are `TimeArray`s: check that the column names of `b` equal the ones of `a`, in the same order, and concatenate the two.
 3. Otherwise check that neither is `nothing`, and concatenate the two with `vcat`.

# Arguments

  - `a`: The column of the earlier block, or `nothing`.
  - `b`: The column of the later block, or `nothing`.
  - `name`: The name of the field, which the message of a refusal quotes.

# Validation

  - Both blocks hold the column, or neither does. An `ArgumentError` is thrown otherwise.
  - Two `TimeArray`s have the same column names in the same order. An `ArgumentError` is thrown otherwise.

# Returns

  - `c`: The rows of `a` followed by the rows of `b`, or `nothing`.

# Related

  - [`vcat_carrier_rows`](@ref)
"""
function vcat_optional(::Nothing, ::Nothing, ::Symbol)
    return nothing
end
function vcat_optional(a::TimeSeries.TimeArray, b::TimeSeries.TimeArray, name::Symbol)
    na = string.(TimeSeries.colnames(a))
    nb = string.(TimeSeries.colnames(b))
    @argcheck(isequal(na, nb),
              ArgumentError("the columns of `$name` in a block of observations must be the ones the first block pinned, in the same order, because a carrier that silently took new ones would put the next observation on the wrong column. Got $(pinned_repr(nb)) against the pinned $(pinned_repr(na))."))
    return vcat(a, b)
end
function vcat_optional(a, b, name::Symbol)
    @argcheck(!isnothing(a) && !isnothing(b),
              ArgumentError("the `$name` column is held by $(isnothing(a) ? "the later" : "the earlier") block of observations and not by the other, so the two cannot be one series: a column is present at every observation or at none."))
    return vcat(a, b)
end
"""
    vcat_panel_rows(a::Nothing, b::Nothing) -> Nothing
    vcat_panel_rows(a::AssetPanel, b::AssetPanel) -> AssetPanel
    vcat_panel_rows(a, b)

Concatenates the [`AssetPanel`](@ref)s of two blocks along the observation axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. `a` and `b` are `nothing`: return `nothing`.
 2. `a` or `b` is a static panel: check that the two panels are equal, and return `a`. A static panel has no observation axis, so there is nothing to concatenate.
 3. `a` and `b` are time-varying: check that neither holds a Panel Field, and concatenate the active masks `amsk` and the estimation masks `emsk`. The rows of a time-varying field are sample data, and the online step keeps no copy of them.
 4. One block holds a panel and the other does not: throw, because the two blocks did not come from one ingestion.

# Arguments

  - `a`: The panel of the earlier block, or `nothing`.
  - `b`: The panel of the later block, or `nothing`.

# Validation

  - Both blocks hold a panel, or neither does. An `ArgumentError` is thrown otherwise.
  - A static panel equals the other panel. An `ArgumentError` is thrown otherwise.
  - A time-varying panel holds no Panel Field. An `ArgumentError` is thrown otherwise.

# Returns

  - `pnl`: The panel of the concatenated observations, or `nothing`.

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

# Returns

  - `n::Int`: The number of rows of `X`.

# Related

  - [`vcat_carrier_rows`](@ref)
"""
carrier_rows(pr::PricesResult) = length(TimeSeries.timestamp(pr.X))
carrier_rows(rd::ReturnsResult) = size(rd.X, 1)
"""
    partial_fit_transform(est, data) -> (est′, data′)

Folds a block of observations into a data step, and returns the block that the transform of the step gives.

This verb is the online form of a preprocessing step that changes the rows it receives. A [`Pipeline`](@ref) gives each step the block that the step before it returned, and takes back the block that the step returns. So the fold and the transform of a step are one call. The transform of the new rows reads the state that the earlier rows left, and the fold moves the state past the new rows. The two halves cannot be separated, because the first return of a new block reads the last price row that the step kept. So the verb returns both.

For [`PricesToReturns`](@ref), [`PriceGapFill`](@ref) and [`MissingDataFilter`](@ref), [`partial_fit!`](@ref) returns the first element of this verb. The read-out of a step is [`fit_preprocessing`](@ref) with no data. A caller's own preprocessing estimator joins the online route of a `Pipeline` with three methods: this verb, the read-out, and a [`supports_partial_fit`](@ref) method that returns `true`. It needs no `partial_fit!` method.

# Arguments

  - `est`: The data step. A step of the library keeps its state in `cache`, which is `nothing` before the first block.
  - `data`: The block of observations, as the batch verb of the step takes it.

# Validation

  - A method of this verb exists for the type of `est`. The fallback throws an `ArgumentError` that names the type.

# Returns

  - `(est′, data′)`: The step with the block folded, and the block as the step returns it.

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

Keeps the prices that an online conversion to returns needs from the blocks before the current one.

[`PricesToReturns`](@ref) keeps this state in `cache`. A return reads two consecutive prices. So the conversion of a new block needs the last price row of the block before it, and nothing more of that block. `tail` is that row. It is a one-row [`PricesResult`](@ref), so that the span, the panel and the implied volatilities of the row stay with its prices.

A [`CatchUpGapReturn`](@ref) reads further back. The observation that ends a gap books the move against the last observed price, and that price can be any number of blocks before. So `anchor` keeps that price for each series column. Under the default rule, which reads no earlier price, `anchor` is `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturnsState(; tail::PricesResult, anchor::Option{<:AbstractVector} = nothing)

## Validation

  - `tail` holds one observation. A `DimensionMismatch` is thrown otherwise.

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
    The last observed price of each series column, in the order of [`series_values`](@ref), or `NaN` where no price was observed. `nothing` under the default gap rule.
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

Puts the series columns of a price carrier side by side: the assets, then the factors, then the benchmark.

[`series_values_returns`](@ref) lays the columns of a returns carrier out in the same order, so a column index of one matrix names the same series in the other.

# Algorithm

 1. Read the asset prices `X` with each gap written as `NaN`, giving the first block of `cols`.
 2. When the carrier holds factor prices `F`, append them to `cols` in the same way.
 3. When the carrier holds benchmark prices `B`, append them to `cols` in the same way.
 4. Concatenate `cols` horizontally.

# Returns

  - `P::AbstractMatrix`: The prices, `observations × series`, with each gap as `NaN`.

# Related

  - [`PricesToReturnsState`](@ref)
  - [`series_values_returns`](@ref)
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

Moves the anchor of each series column of a [`PricesToReturnsState`](@ref) past a block.

The new anchor of a column is its last observed price in the block. A column with no observed price in the block keeps its anchor.

# Algorithm

 1. Copy `anchor` into `out`.
 2. For each column `j` of `P`, find the last row `t` whose price is observed. When there is one, write `P[t, j]` into `out[j]`.

# Arguments

  - `anchor`: The last observed price of each series column before the block, or `NaN`.
  - `P`: The prices of the block, `observations × series`.

# Returns

  - `out::AbstractVector`: A new vector. `anchor` does not change.

# Related

  - [`PricesToReturnsState`](@ref)
  - [`is_missing_value`](@ref)
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

Writes the cells that a Gap Return algorithm resolves in the returns of a new block, and reads the anchor of the state for the prices before the block.

The batch conversion gives [`gap_return`](@ref) a whole column. This function gives it the column of the block with two rows in front: the anchor and the last price row kept. So the algorithm meets the observed price that a gap books its move against, and the price just before the first row of the block, as it does in the whole history. The function copies back only the cells of the new rows, and only the cells that [`gap_return_writable`](@ref) admits over the extended column. The batch conversion keeps the same rule. An anchor is an earlier observed price, and a column with no observed price admits no cell before its first observation.

# Algorithm

 1. Make `p`, a column of prices two rows longer than `P`, and `r`, a column of returns one row longer than `R`.
 2. For each series column `j`, write `anchor[j]` and `last[1, j]` into the first two rows of `p`, and the prices of the block after them. Write `NaN` into the first row of `r`, and the returns of the block after it.
 3. Find the writable cells `w` of the extended column with [`gap_return_writable`](@ref). Go to the next column when no cell is writable.
 4. Apply `alg` to the extended column with [`gap_return`](@ref), giving `out`.
 5. Copy `out` into `R` at each writable cell of the new rows.

# Arguments

  - `alg`: The Gap Return algorithm.
  - `R`: The returns of the new block, `observations × series`, changed in place.
  - `P`: The prices of the new block, `observations × series`.
  - `anchor`: The last observed price of each series column before the block.
  - `last`: The last price row before the block, `1 × series`.
  - `ret_method`: `:simple` or `:log`.

# Validation

  - `alg` returns one value for each return cell of the extended column. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `R`: The same matrix, with the writable cells of the new rows written.

# Related

  - [`gap_return`](@ref)
  - [`gap_return_writable`](@ref)
  - [`PricesToReturnsState`](@ref)
"""
function block_gap_return!(alg::AbstractGapReturnAlgorithm, R::AbstractMatrix,
                           P::AbstractMatrix, anchor::AbstractVector, last::AbstractMatrix,
                           ret_method::Symbol)
    p = similar(P, promote_type(eltype(anchor), eltype(last), eltype(P)), size(P, 1) + 2)
    r = similar(R, size(R, 1) + 1)
    for j in axes(P, 2)
        p[1], p[2] = anchor[j], last[1, j]
        copyto!(view(p, 3:length(p)), view(P, :, j))
        r[1] = oftype(R[1, j], NaN)
        copyto!(view(r, 2:length(r)), view(R, :, j))
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

Converts a block of prices to returns as the whole history does, and keeps what the next block needs.

Each return of the rows that the verb returns equals the return that the batch conversion of the whole history writes at that row. A return reads two consecutive prices, a Gap Return reads the last observed price, and the state keeps both.

# Algorithm

 1. Check that the gap rule has an online form, and that the block holds one observation at least.
 2. On the first block, check that the block holds two observations when `padding` is `false`. Convert the block alone with [`prices_to_returns`](@ref), giving `rd`. The block is the whole history.
 3. On a later block, put the kept row `tail` in front of the block with [`vcat_carrier_rows`](@ref), giving `window`. Convert `window` under the `ret_method` and the `padding` of the step and the default gap rule, giving `rw`. Keep the last `n` rows of `rw` as `rd`, where `n` is the number of observations of the block. The kept row gives the return of the first new observation, and the step drops the row that `padding` adds at the front of `window`.
 4. Under a [`CatchUpGapReturn`](@ref), on a later block, write the resolved cells of the new rows into `R` with [`block_gap_return!`](@ref), and rebuild `rd` from `R` with [`returns_with_series`](@ref).
 5. Keep the last price row of the block as the new `tail`. Under a [`CatchUpGapReturn`](@ref), move the anchor past the block with [`advance_anchor`](@ref), from an anchor of `NaN` on the first block.

# Arguments

  - `ptr`: The conversion, with its state in `cache`, or `nothing` before the first block.
  - `pr`: The block of prices.

# Validation

  - The gap rule is `nothing` or a [`CatchUpGapReturn`](@ref). A caller's own rule can read the whole price column, so its online form can differ from the batch conversion. An `ArgumentError` is thrown otherwise.
  - The block holds one observation at least. An `IsEmptyError` is thrown otherwise.
  - Without `padding`, the first block holds two observations at least, because a conversion of one price row gives no return. An `IsEmptyError` is thrown otherwise.
  - A later block has the universe of the first block. [`vcat_carrier_rows`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `(ptr′, rd)`: The conversion with the block folded, and the returns of the block as a [`ReturnsResult`](@ref).

# Related

  - [`partial_fit_transform`](@ref)
  - [`PricesToReturnsState`](@ref)
  - [`prices_to_returns`](@ref)
"""
function partial_fit_transform(ptr::PricesToReturns, pr::PricesResult)
    @argcheck(supports_partial_fit(ptr),
              ArgumentError("a `PricesToReturns` with a `$(typeof(ptr.gap_return_alg).name.name)` gap rule has no online form: a Gap Return algorithm may read the whole price column, and a block holds the prices up to its own end only, so the returns written at one step can differ from the ones the whole history gives. Use `CatchUpGapReturn()` or the default rule, or declare a refit with `Online(pipe)`."))
    n = carrier_rows(pr)
    @argcheck(n > 0, IsEmptyError("a block of prices holds at least one observation"))
    P = series_values(pr)
    state = ptr.cache
    rd = if isnothing(state)
        @argcheck(ptr.padding || n > 1,
                  IsEmptyError("the first block of prices converts alone, and without padding a conversion of n price rows gives n - 1 returns, so the first block holds at least two observations; got one. Fold a longer first block, or set `padding = true`."))
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

Puts the series columns of a returns carrier side by side, in the order of [`series_values`](@ref): the assets, then the factors, then the benchmark.

# Algorithm

 1. Copy the asset returns `X` into the first block of `cols`.
 2. When the carrier holds factor returns `F`, append a copy to `cols`.
 3. When the carrier holds benchmark returns `B`, append a copy to `cols`, as one column when `B` is a vector.
 4. Concatenate `cols` horizontally.

# Returns

  - `R::Matrix`: The returns, `observations × series`. The matrix is a copy, so a change to it does not reach `rd`.

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

Rebuilds a returns carrier from the series matrix that [`series_values_returns`](@ref) lays out.

A Gap Return writes a finite return where the default conversion left none. So the function derives the estimation mask again, as the conversion does.

# Algorithm

 1. Split `R` into the asset returns `X`, the factor returns `F` and the benchmark returns `B`, in the layout of [`series_values_returns`](@ref). `B` is one column when the benchmark of `rd` is a vector.
 2. When `rd` holds a time-varying panel, keep its active mask `amsk`, and derive the estimation mask `emsk` as the cells of `amsk` where `X` is finite. When `emsk` is true everywhere, [`compress_all_true`](@ref) replaces the two masks with one all-true mask.
 3. Build the returns carrier from `X`, `F`, `B`, the panel, and the other fields of `rd`.

# Returns

  - `rd′::ReturnsResult`: The returns carrier of `R`.

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

Reads a stepped [`PricesToReturns`](@ref) out.

The conversion keeps no fitted value. So its fitted object is the estimator with `cache` set to `nothing`, which is the object that the batch fit returns.

# Returns

  - `ptr′::PricesToReturns`: The estimator with no state.

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

Keeps the carried price of each asset and the end of the last block for an online gap fill with a carried price.

[`PriceGapFill`](@ref) with a [`CarriedPrice`](@ref) keeps this state in `cache`. The carried price is the price that the gaps of the next block take, and the seed that the fitted result replays. The fitted result records the end of the last block as the end of its training window.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceGapFillState(; nx::AbstractVector{Symbol}, v::AbstractVector, te)

## Validation

  - `length(nx) == length(v)`. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
"""
@concrete struct PriceGapFillState <: AbstractPartialFitState
    """
    $(field_dict[:pf_nx_pinned])
    """
    nx
    """
    The last observed price of each asset, or `missing` where no price was observed.
    """
    v
    """
    The last timestamp of the last block folded. The carried prices precede each observation after it, and the fitted result records it as the end of its training window.
    """
    te
end
function PriceGapFillState(; nx::AbstractVector{Symbol}, v::AbstractVector,
                           te)::PriceGapFillState
    @argcheck(length(nx) == length(v), DimensionMismatch)
    return PriceGapFillState(nx, v, te)
end
function Base.copy(x::PriceGapFillState)
    return PriceGapFillState(copy(x.nx), copy(x.v), x.te)
end
function merge_states(a::PriceGapFillState, b::PriceGapFillState)
    assert_pinned_carrier(a.nx, b.nx, :nx)
    return PriceGapFillState(; nx = a.nx,
                             v = [ismissing(y) ? x : y for (x, y) in zip(a.v, b.v)],
                             te = b.te)
end
"""
    partial_fit_transform(est::PriceGapFill, pr::PricesResult) -> (est′, pr′)

Fills the gaps of a block of prices as the fit over the whole history does, and carries the last observed prices forward.

A column that no earlier block priced has no carried price. A gap that opens such a column stays a gap until the first observed price of the column, because the batch replay of [`PriceGapFill`](@ref) leaves it so on the training window. A price later in the block is not a seed for a gap before it.

# Algorithm

 1. Check that the convention is a [`CarriedPrice`](@ref).
 2. Read the asset names `names` and a copy `vals` of the prices, with each gap as `NaN`. When the step keeps a state, check that `names` equals the names that the state pinned, and read the carried prices `v` from the state. Otherwise `v` is `missing` for each asset.
 3. Resolve the Listing Span `span` that bounds the fill with [`gap_fill_span`](@ref), as the batch replay does.
 4. Find `t0`, the first row of the block after the end `te` of the state. With no state, `t0` is one row past the end of the block.
 5. For each column `j`, find the last observed row `t`. Go to the next column when `v[j]` is `missing` and the block observes no price of the column. When `t` exists, the new carried price `vnew[j]` is `vals[t, j]`.
 6. Fill the gaps of column `j` of `vals` with [`gap_fill_column!`](@ref). A column with a carried price takes it as the seed from row `t0`. A column with no carried price gets a seed from one row past the end of the block, which no row reads, so its gaps fill from the earlier prices of the block only.
 7. Build the filled price carrier from `vals`, and the new state from `names`, `vnew` and the last timestamp of the block.

# Arguments

  - `est`: The gap fill, with its state in `cache`, or `nothing` before the first block.
  - `pr`: The block of prices.

# Validation

  - The convention is a [`CarriedPrice`](@ref). A fill by a statistic of the window changes each earlier gap when the window grows, so it has no online form. [`supports_partial_fit`](@ref) answers `false` for it, and a `Pipeline` refuses it at warm-up. An `ArgumentError` is thrown otherwise.
  - A later block has the asset names of the first block. An `ArgumentError` is thrown otherwise.
  - Under `strict`, a block with a gap comes from a carrier with a Listing Span. [`gap_fill_span`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `(est′, pr′)`: The gap fill with the block folded, and the block with its gaps filled.

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
    ts = TimeSeries.timestamp(pr.X)
    #! A carried price is written only after the rows it was read from, as the batch replay
    #! writes its seed only after the training window. With no state nothing precedes the
    #! block, so no seed is written; with one, the block follows the state's end.
    t0 = isnothing(state) ? size(vals, 1) + 1 : searchsortedlast(ts, state.te) + 1
    vnew = copy(v)
    for j in axes(vals, 2)
        t = findlast(x -> !is_missing_value(x), view(vals, :, j))
        if ismissing(v[j]) && isnothing(t)
            continue
        end
        if !isnothing(t)
            vnew[j] = vals[t, j]
        end
        #! A column the state has not priced has no seed. The walk is handed the block's
        #! last price with a start past the block's end, so no row reads it: the column's
        #! gaps fill from the block's own earlier prices alone.
        seed, tj = ismissing(v[j]) ? (vals[t, j], size(vals, 1) + 1) : (v[j], t0)
        gap_fill_column!(est.fill, vals, span, j, seed, tj)
    end
    X = TimeSeries.TimeArray(ts, vals, TimeSeries.colnames(pr.X))
    out = PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                       pnl = pr.pnl, span = pr.span)
    cache = PriceGapFillState(; nx = names, v = vnew, te = last(ts))
    return rebuild_estimator(est, (; cache = cache)), out
end
"""
    fit_preprocessing(est::PriceGapFill)

Reads a stepped [`PriceGapFill`](@ref) out as the [`PriceGapFillResult`](@ref) that the batch fit over the same rows gives.

The result holds each asset that the folded blocks priced, with its last observed price, and the last folded timestamp as the end of the training window.

# Algorithm

 1. Read the state with [`partial_fit_cache`](@ref).
 2. Keep the assets with a carried price, giving `keep`, and read their prices into `v`, whose element type drops `Missing`.
 3. Build the [`PriceGapFillResult`](@ref) from the kept names, `v`, the end `te` of the state, and the `fill` and `strict` of the step.

# Validation

  - The step keeps a state. [`partial_fit_cache`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `res::PriceGapFillResult`: The fitted gap fill of the folded observations.

# Related

  - [`partial_fit_transform`](@ref)
  - [`PriceGapFillState`](@ref)
"""
function fit_preprocessing(est::PriceGapFill)
    state = partial_fit_cache(est)
    keep = findall(!ismissing, state.v)
    v = identity.([state.v[j] for j in keep])
    return PriceGapFillResult(state.nx[keep], v, state.te, est.fill, est.strict)
end
function supports_partial_fit(est::PriceGapFill)
    return isa(est.fill, CarriedPrice)
end
#! End: PriceGapFill.
#! Begin: MissingDataFilter.
"""
$(DocStringExtensions.TYPEDEF)

Keeps the number of observations and the missing count of each asset for an online missing-data filter.

[`MissingDataFilter`](@ref) keeps this state in `cache`, and its read-out computes the column filter from the counts.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilterState(; nx::AbstractVector{Symbol}, miss::AbstractVector{<:Integer}, n::Integer)

## Validation

  - `length(nx) == length(miss)`. A `DimensionMismatch` is thrown otherwise.
  - `n >= 0`. A `DomainError` is thrown otherwise.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`MissingDataFilterResult`](@ref)
"""
@concrete struct MissingDataFilterState <: AbstractPartialFitState
    """
    $(field_dict[:pf_nx_pinned])
    """
    nx
    """
    The number of missing observations of each asset, over each block folded.
    """
    miss
    """
    The number of observations folded.
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

Counts the gaps of a block of prices, and returns the block unchanged.

The column filter changes only which assets survive, and a longer window can change that set. So the verb leaves the column filter to the read-out. There, [`fit_preprocessing`](@ref) with no data gives the [`MissingDataFilterResult`](@ref) of the whole history, and a `Pipeline` applies it as a view. At `row_thr = 1` the row filter drops no row, so every row passes.

# Algorithm

 1. Check that `row_thr` is one.
 2. Count the missing observations of each asset in the block, giving `miss`.
 3. With no state, make a state from the asset names, `miss` and the number of observations of the block. With a state, check that the asset names equal the names that it pinned, and add `miss` and the number of observations to its counts.
 4. Return the step with the new state, and `pr` unchanged.

# Arguments

  - `mdf`: The filter, with its state in `cache`, or `nothing` before the first block.
  - `pr`: The block of prices.

# Validation

  - `row_thr == 1`. A lower threshold drops a row that an earlier step kept when a column leaves the universe, and the returns on each side of that row change. [`supports_partial_fit`](@ref) answers `false` for it, and a `Pipeline` refuses it at warm-up. An `ArgumentError` is thrown otherwise.
  - A later block has the asset names of the first block. An `ArgumentError` is thrown otherwise.

# Returns

  - `(mdf′, pr)`: The filter with the block folded, and the block itself.

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

Reads a stepped [`MissingDataFilter`](@ref) out as the [`MissingDataFilterResult`](@ref) that the batch fit over the same rows gives.

# Algorithm

 1. Read the state with [`partial_fit_cache`](@ref).
 2. Keep the assets whose share of missing observations `miss / n` does not exceed `col_thr`, giving `keep`, and check that one asset at least survives. [`share_at_most`](@ref) computes the share in the type of `col_thr`, as the batch fit does.
 3. Build the [`MissingDataFilterResult`](@ref) from the surviving asset names and `row_thr`.

# Validation

  - The step keeps a state. [`partial_fit_cache`](@ref) throws an `ArgumentError` otherwise.
  - One asset at least survives the column filter. An `IsEmptyError` is thrown otherwise.

# Returns

  - `res::MissingDataFilterResult`: The fitted filter of the folded observations.

# Related

  - [`partial_fit_transform`](@ref)
  - [`MissingDataFilterState`](@ref)
"""
function fit_preprocessing(mdf::MissingDataFilter)
    state = partial_fit_cache(mdf)
    keep = share_at_most.(state.miss, state.n, mdf.col_thr)
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

Lists the fields that the display of a data step shows.

The display shows `cache` only when the step keeps a state, so a step that has folded no block shows no `cache` line.

# Returns

  - `fns::Tuple`: The names of the fields to show.

# Related

  - [`show_fields`](@ref)
"""
function show_fields(est::Union{<:PricesToReturns, <:PriceGapFill, <:MissingDataFilter})
    fns = fieldnames(typeof(est))
    return isnothing(est.cache) ? filter(!=(:cache), fns) : fns
end
