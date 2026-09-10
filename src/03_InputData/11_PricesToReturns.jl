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

 1. Check `X`, and check that the asset, factor and benchmark series can still be named after the merge with [`assert_distinct_series_names`](@ref). Read the asset names and the asset timestamps from `X`, and check `pnl` against them with [`check_asset_panel`](@ref).
 2. Merge the factor prices `F` into `X` under `join_method`, and record the factor names.
 3. Merge the benchmark prices `B` into `X` under `join_method`, and record the benchmark names. A benchmark is one shared column, or one column per asset.
 4. Apply `map_func` to every entry, when one is given.
 5. Collapse the time series with `collapse_args`, when they are given. This is the step that changes the frequency.
 6. Convert the table to a `DataFrames.DataFrame`.
 7. Replace every `missing` with `NaN`, so that the two conventions a source spells an absent price with become one: an outer join of per-asset series pads with `NaN`, and a wide table built from a tidy one leaves `missing`. This is the only unification, it runs unconditionally, and it is what makes the two ragged-history sources behave alike. It is also the last step that touches an absent price: every row and every column of the table reaches the conversion, whatever it holds.
 8. Convert the prices to returns with `TimeSeries.percentchange` under `ret_method` and `padding`. This is the step that applies the formula above. It computes both branches through logarithms — the log return is ``\\ln P_{t,i} - \\ln P_{t-1,i}``, and the simple return is `expm1` of it — so the two agree with the closed forms above to floating point rather than to the last bit. When `padding` is `true` the first observation is kept and its return is `NaN`, so the returns keep the length of the price clock. **A gap carried here does not spread.** The formula reads two prices, so a run of `k` gapped prices makes exactly the `k + 1` returns that read one of them non-finite, and every later return of that column is computed from two observed prices and is finite. A gap is confined to its own column for the same reason: no asset's return reads another's price.
 9. Resolve the cells the conversion left non-finite with [`apply_gap_return`](@ref), under `gap_return_alg`. `nothing` is the default rule, and its method returns the table untouched, so the arithmetic step 8 produced is bit-identical. An algorithm may write only a non-finite cell inside a column's Listing Span that has an earlier observed price, which is what freezes every return computed from two observed prices, and it reports an `@info` when it finds no such cell.
10. Name the three blocks. Step 1 refused every name two of the tables shared and the clock's own name `timestamp`, so the asset names `nx`, the factor names `nf` and the benchmark names `nb` are the lists read off the three tables, and `ts` is the `timestamp` column the `DataFrames.DataFrame` conversion wrote. Each is the typed vector its table held, rather than whatever is left once the other groups have taken what they recognise.
11. Index the implied volatilities `iv` by `ts`, then check `iv` and `ivpa` against the asset count.
12. Subselect the [`AssetPanel`](@ref). Read the assets' positions `acols` in the original asset names, recover the surviving rows with [`feature_row_indices`](@ref), and view the panel with [`port_opt_view`](@ref), handing it the asset names so that a square tensor Panel Field is cut on its label axis too. The conversion removes no column, so `acols` is the whole asset axis and the subselection that bites is the observation one: a time-varying panel is cut to the surviving observations, matched back into the original price timestamps, and a surviving timestamp absent from that clock throws. Under `collapse_args` this gives the aggregated period the values of the row at its representative timestamp, which is last-observation semantics and matches [`LastObservation`](@ref).
13. State the universe. Cut `span` to the price rows with [`span_carrier_view`](@ref), and hand it and the converted returns to [`returns_universe_masks`](@ref), which projects it onto the returns clock and intersects it with finiteness. A carrier that states no span states no universe, and the conversion emits no panel. [`attach_universe_masks`](@ref) puts the pair onto the Asset Panel, keeping whatever Panel Fields it already carried, and mints one with no field when the carrier held none.
14. Build the asset, factor and benchmark matrices from the columns of each group. The asset group is always present, because the conversion removes no column; a factor or benchmark group given no column is `nothing`.
15. Return the [`ReturnsResult`](@ref).

**The conversion removes no observation and no asset.** Deleting either is a **Universe Policy**, and a policy is fitted on a training window and replayed by name, which a stateless conversion cannot do; [`MissingDataFilter`](@ref) owns it, with `col_thr` deleting an asset and `row_thr` an observation. ADR 0133 states the rule that a keyword survives here if and only if it changes the arithmetic of a return.

# Arguments

  - `X`: Asset price data (observations × assets).
  - `F`: Optional Factor price data (observations × factors).
  - `B`: Optional Benchmark price data (observations × assets) or (observations × 1).
  - `iv`: Optional Implied volatility data.
  - `ivpa`: Optional Implied volatility risk premium adjustment.
  - `ret_method`: Return calculation method (`:simple` or `:log`).
  - `padding`: Whether to pad missing values in returns calculation.
  - `gap_return_alg`: What the observations a price gap left non-finite carry. `nothing` is the arithmetic — a return is the change between two consecutive observations, so a run of `k` gapped prices leaves `k + 1` non-finite returns and the move across the gap is recorded nowhere — and [`CatchUpGapReturn`](@ref) books that move on the observation the asset resumes trading instead, shortening the Held Gap to `k`. Any algorithm may write only a non-finite cell inside an asset's Listing Span that has an earlier observed price in its column, so a return computed from two observed prices is frozen whichever one is stated. It has no cell to write over a gap-free table, and reports an `@info` there.
  - `collapse_args`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func`: Optional function to apply to the data before returns calculation.
  - `join_method`: How to join asset, factor data and benchmark data (`:outer`, `:inner`, etc.).
  - `pnl`: Optional [`AssetPanel`](@ref), as [`asset_panel`](@ref) returns it.
  - `span`: Optional **Listing Span** on the price clock, as [`PriceIngestion`](@ref) derives it or a caller declares it. Given one, the conversion projects it onto the returns clock with [`universe_masks`](@ref) and hands the returns carrier an [`AssetPanel`](@ref) stating the universe — always, a gapless panel included, so `pnl === nothing` means one thing only: the carrier was not built by the ingestion layer. `nothing` states no universe and emits no panel, whether or not the prices hold a gap; a window-local span cannot answer the question, because a delisting straddling the window end reads there as an asset that was never listed.

# Validation

  - Every price reaching step 11 is positive. `TimeSeries.percentchange` takes a logarithm on both branches, so a negative price raises a `DomainError` from inside it, on the simple branch as well.
  - `!isempty(X)`.
  - The asset, factor and benchmark column names are pairwise disjoint, and none of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref) naming the offending columns.
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
                           collapse_args::Tuple = (),
                           map_func::Option{<:Function} = nothing,
                           join_method::Symbol = :outer,
                           pnl::Option{<:AssetPanel} = nothing,
                           span::Option{<:AbstractMatrix{Bool}} = nothing)
    @argcheck(!isempty(X), IsEmptyError)
    assert_distinct_series_names(X, F, B)
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
    P = X
    X = TimeSeries.percentchange(TimeSeries.TimeArray(X; timestamp = :timestamp),
                                 ret_method; padding = padding)
    X = DataFrames.DataFrame(X)
    X = apply_gap_return(gap_return_alg, X, P, ret_method)
    #! The three name lists are pairwise disjoint, and none of them is `timestamp`, because
    #! `assert_distinct_series_names` refused every collision at the door. So each block is
    #! named outright rather than recovered by intersection, and the clock is read as the
    #! one column that carries it. Recovering the clock as "whatever column no block
    #! claims" is what built a `Vector{Any}` of interleaved dates and prices out of a
    #! collision, which is issue #990.
    nx = asset_names
    nf = factor_names
    nb = benchmark_names
    N = length(nx)
    ts = X[!, :timestamp]
    if !isnothing(iv)
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)),
                  ArgumentError("ts must be a subset of the timestamps in iv"))
        iv = values(iv[ts])
        assert_nonempty_nonneg_finite_val(iv, :iv)
        assert_nonempty_gt0_finite_val(ivpa, :ivpa)
        @argcheck(size(iv) == (DataFrames.DataAPI.nrow(X), N), DimensionMismatch)
        if isa(ivpa, VecNum)
            @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
        end
    end
    #! The conversion removes no column, so the assets reach it in their original order and
    #! `acols` is the whole asset axis. It is still read rather than assumed, because it is
    #! what pairs a Panel Field and a span column with the asset they describe.
    acols = Vector{Int}(indexin(nx, asset_names))
    if !isnothing(pnl)
        rows = feature_row_indices(pnl, ts, asset_ts)
        pnl = port_opt_view(pnl, rows, acols, asset_names)
    end
    #! The span is on the price clock and the masks are on the returns clock, so the span is
    #! cut to the price rows the merge and the collapse left and universe_masks does the
    #! crossing. Both padding conventions reach it, and it reads which from the two row
    #! counts.
    amsk, emsk = returns_universe_masks(span_carrier_view(span, P[!, :timestamp], asset_ts,
                                                          acols), Matrix(X[!, nx]))
    pnl = attach_universe_masks(pnl, amsk, emsk)
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
    X = Matrix(X[!, nx])
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, iv = iv,
                         ivpa = ivpa, pnl = pnl)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator converting price-level data into returns-level data.

`PricesToReturns` is the estimator form of [`prices_to_returns`](@ref): it consumes a [`PricesResult`](@ref) and produces a [`ReturnsResult`](@ref). It is stateless — applying it to any window simply runs the conversion — so its fitted object is the estimator itself.

Missing-data filtering is deliberately *not* part of this estimator, and [`prices_to_returns`](@ref) carries no keyword that would put it there; use [`MissingDataFilter`](@ref) and [`PriceGapFill`](@ref) as separate, independently tunable steps. Deleting an observation or an asset is a **Universe Policy**, and a policy is fitted on a training window and replayed by name; this step is stateless, so it holds none.

The step is stateless, and it does not need to be stateful to fix an asset universe: the carrier states one. A [`PricesResult`](@ref) that [`price_ingestion`](@ref) built carries a **Listing Span**, and this step projects it onto the returns clock and hands the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) whose two masks say which assets are in the universe and which of them can be estimated at each observation. The asset axis is fixed before the split, so every window of every fold carries every asset and a window can no longer silently lose a column.

!!! warning

    A carrier the ingestion layer did not build states no universe, and the conversion does not guess one from the window: a window-local span reads a delisting straddling the window end as an asset that was never listed. Its gaps are still carried and still handled — with no panel the Coverage Universe reads finiteness alone — but the fold is left to infer the universe it would otherwise have been told. Build the carrier with [`price_ingestion`](@ref), or declare a listing calendar as its `span`.

# Algorithm

The estimator is stateless, so both verbs are thin.

 1. [`fit_preprocessing`](@ref) returns the estimator itself. There is no state to fit.
 2. [`apply_preprocessing`](@ref) calls [`prices_to_returns`](@ref) with the six fields as keywords, and with `X`, `F`, `B`, `iv`, `ivpa`, `pnl` and `span` read off the [`PricesResult`](@ref). It returns the [`ReturnsResult`](@ref).

Every row and every column of the window reaches the conversion, because the conversion has no way to drop one. `gap_return_alg` is a field, because it decides what the observations a gap left non-finite carry, which is the arithmetic of a return rather than a policy about the universe.

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
export prices_to_returns, PricesToReturns, CatchUpGapReturn
