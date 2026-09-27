"""
$(DocStringExtensions.TYPEDEF)

Supertype of the policies that write a return into the cells that a price gap left non-finite.

A return is the change between two consecutive observations. So a run of `k` gapped prices inside the series leaves `k + 1` non-finite returns, and a run at either end leaves `k`. Under the default rule no return records the move across the gap, and `nothing` on [`prices_to_returns`](@ref) selects that rule. A caller who wants the move booked gives one of these algorithms instead.

An algorithm writes **only** a cell that reads a gapped price, inside the asset's Listing Span, after the first observed price of its column. [`gap_return_writable`](@ref) finds that set, and [`apply_gap_return`](@ref) restores every other cell. So a return computed from two observed prices keeps its value whatever the algorithm returns. A gap changes no cell that reads none of its prices, and no return appears before an asset's first price.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractGapReturnAlgorithm` and implement the following methods:

## `gap_return`

  - `gap_return(alg::AbstractGapReturnAlgorithm, p::AbstractVector, r::AbstractVector, ret_method::Symbol) -> Vector`: One column's returns, with the writable cells resolved.

### Arguments

  - `alg`: The concrete subtype instance.
  - `p`: One column's prices along the observation axis, gaps included.
  - `r`: The returns `TimeSeries.percentchange` computed from `p`, so `length(p) - length(r)` is `0` under `padding` and `1` otherwise.
  - `ret_method`: `:simple` or `:log`. Compute a value with [`gap_return_value`](@ref), which holds the two branches.

### Returns

  - `out::Vector`: The same length as `r`. [`apply_gap_return`](@ref) reads back only the cells that [`gap_return_writable`](@ref) admits. So a method can return every other cell unchanged, and it does not need to check the rule itself.

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

Books the whole move across a Held Gap on the observation that ends it, so the gap holds `k` non-finite returns.

A suspension of `k` observations leaves `k + 1` non-finite returns under the default rule. This algorithm writes ``p_{t+k,\\,i} / p_{t-1,\\,i} - 1`` on the observation where the asset trades again, and it leaves the `k` observations inside the gap non-finite. The compounded return across the gap then equals the price ratio, and the Held Gap is the run of unpriced observations. The algorithm writes nothing at an asset's inception, because no earlier observed price exists there.

The estimation mask reads the values and not the algorithm that wrote them. So the mask marks the cell where the asset trades again as estimable, and a moment estimator reads a `(k + 1)`-period return as one draw of a one-period return. Under independent increments, the scale of that draw is about ``\\sqrt{k + 1}`` times the scale of a one-period return.

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

Compute one return from two prices that need not be consecutive.

This function holds the two `ret_method` branches for the Gap Return family. A new algorithm chooses the two prices, and this function turns them into a return. It uses the arithmetic of `TimeSeries.percentchange`, which computes both branches through logarithms. So a value that an algorithm writes uses the same formula as the cells around it.

# Arguments

  - `ret_method`: `:simple` or `:log`.
  - `pt`: The later price.
  - `p0`: The earlier price, the return's anchor.

# Returns

  - `r::Number`: ``\\ln p_{t} - \\ln p_{0}`` under `:log`, and `expm1` of it otherwise.

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

Find the cells of one column that a Gap Return algorithm can write.

Every algorithm of the family obeys this one rule, and no algorithm states it again. [`apply_gap_return`](@ref) restores every cell outside the returned set. So no algorithm can change a return computed from two observed prices, write a return before an asset's first price, or write one after its last price.

The bounds are the Span Rule and its projection, which [`listing_span`](@ref) and [`PortfolioOptimisers.project_span`](@ref) state for a whole panel. This function reads them off one price column, for two reasons. The writable set is per column. And the table that reaches the conversion step of [`prices_to_returns`](@ref) can be a filtered table, and not the caller's table.

# Algorithm

 1. Read the offset between the two clocks as `length(p) - length(r)`, which is `0` when `padding` kept the first observation and `1` when it did not. Return cell `j` is then the change onto price row `t = j + off`, and it reads the prices at rows `t - 1` and `t`.
 2. Find the column's Listing Span on the price clock, from the first observed price to the last. A column with no observed price admits no cell.
 3. Admit return cell `j` when `t` lies in `[first + 1, last]` and one of the prices at rows `t - 1` and `t` is gapped. The interval is the span projected onto the returns clock, because a return reads the earlier price of its pair. A cell that reads a gapped price is always non-finite under the default rule. The function does not admit a cell that reads two observed prices, even a non-finite one. A zero price gives a return of `-1` or `-Inf` on its own observation and `Inf` on the next, and those cells keep the values of the default rule.

# Arguments

  - `p`: One column's prices along the observation axis, gaps included.
  - `r`: The returns `TimeSeries.percentchange` computed from `p`.

# Returns

  - `w::BitVector`: The same length as `r`, true on the cells that an algorithm can write.

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
        w[j] = i1 < t <= i2 && (is_missing_value(p[t - 1]) || is_missing_value(p[t]))
    end
    return w
end
"""
    gap_return(alg::CatchUpGapReturn, p::AbstractVector, r::AbstractVector, ret_method::Symbol) -> Vector

Resolve the writable cells of one column's returns.

Dispatch on `alg` selects the algorithm. The library ships [`CatchUpGapReturn`](@ref) alone. The family is a type and not a flag, so another rule costs one type and one method. A rule that spreads the move of a suspension across its observations is an example.

# Algorithm

[`CatchUpGapReturn`](@ref) reads the observations in order, and keeps the row of the last observed price as the anchor.

 1. On a gapped price, write nothing and keep the anchor. The observation is inside the Held Gap, and its return stays non-finite.
 2. On an observed price whose predecessor is also observed, write nothing. `TimeSeries.percentchange` computed that cell from two consecutive prices, and [`gap_return_writable`](@ref) does not admit it.
 3. On an observed price whose predecessor is gapped, write [`gap_return_value`](@ref) of this price against the anchor price. This observation ends the gap, and it carries the whole move across the gap.
 4. After step 2 or step 3, move the anchor to the row of this price.

The first observed price of a column has no anchor, so the algorithm writes nothing on it. That is why an inception needs no case of its own.

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

Apply the `gap_return_alg` of [`prices_to_returns`](@ref) to the converted table.

The method for `nothing` is the default rule. It returns the table unchanged, so the returns are bit for bit the ones that `TimeSeries.percentchange` computed. The method for an algorithm writes only the cells that [`gap_return_writable`](@ref) admits.

A Gap Return reads one column and no asset axis. So this function applies it to every series of the converted table, which holds the assets, the factors and the benchmarks.

# Algorithm

 1. For each series column of `R`, read the prices of the same name from `P`.
 2. Find the writable cells with [`gap_return_writable`](@ref). Go to the next column when no cell is writable.
 3. Call [`gap_return`](@ref) on the column, and copy back **only** the writable cells. Every other cell keeps its value, whatever the algorithm returned.
 4. Log an `@info` when no column admits a cell. A table with no gap admits none, and so does a table whose gaps are all at the ends of the Listing Spans. The returns are then correct, so the function does not throw. It does not warn either, because it cannot tell this table from one where the caller expected a gap.

# Arguments

  - `alg`: The Gap Return algorithm, or `nothing` for the default rule.
  - `R`: The converted table, `:timestamp` first and one column per series.
  - `P`: The price table that reaches the conversion, with the same series columns.
  - `ret_method`: `:simple` or `:log`.

# Validation

  - [`gap_return`](@ref) returns one value for each cell of the column. The function throws a `DimensionMismatch` otherwise.

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
        @info("`gap_return_alg` is a $(typeof(alg)) and no cell is writable, so the returns are the ones the default rule computed. A Gap Return writes only a return that reads a gapped price, inside an asset's Listing Span, after the first observed price of its column. The table that reaches the conversion holds no such return.")
    end
    return R
end
"""
    append_carrier_block!(P::DataFrames.DataFrame, A::Nothing, ts, sym::Symbol) -> Vector{String}
    append_carrier_block!(P::DataFrames.DataFrame, A::TimeSeries.TimeArray, ts, sym::Symbol) -> Vector{String}

Write one price block of the carrier beside the asset block, on the asset clock.

The carrier states one clock. So this function reads a factor or benchmark series at the asset timestamps, and does not join it onto them. A join adds or drops observations, and [`price_ingestion`](@ref) owns every change of the clock. The function writes the columns one at a time, so the blocks do not need the one value type that `TimeSeries.merge` needs. A `Float32` factor table beside a `Float64` asset table converts, and the conversion promotes the two types.

# Algorithm

 1. A block that is `nothing` adds no column and no name.
 2. Otherwise, check that the block states the asset clock. Throw an error that names the block if it does not.
 3. Write every absent price as `NaN` with [`unify_gaps`](@ref), which [`price_ingestion`](@ref) also runs. It changes nothing on a carrier that the ingestion layer built. On a carrier built by hand, it makes a `missing` convert as a `NaN` does.
 4. Write each column of the block into `P` under its own name, and return the names in order.

# Arguments

  - `P`: The table the conversion is assembling, already carrying the clock and the asset columns.
  - `A`: The factor or benchmark price series, or `nothing`.
  - `ts`: The asset timestamps, which are the carrier's clock.
  - `sym`: The block's name in the refusal, `:F` or `:B`.

# Validation

  - `TimeSeries.timestamp(A) == ts`. The function throws a [`ConflictingArgumentError`](@ref) otherwise. The message names [`price_ingestion`](@ref), which puts two series on one clock.

# Returns

  - `n::Vector{String}`: The column names of the block, empty when the block is `nothing`.

# Related

  - [`prices_to_returns`](@ref)
  - [`price_ingestion`](@ref)
  - [`unify_gaps`](@ref)
  - [`PricesResult`](@ref)
"""
function append_carrier_block!(::DataFrames.DataFrame, ::Nothing, ::Any, ::Symbol)
    return String[]
end
function append_carrier_block!(P::DataFrames.DataFrame, A::TimeSeries.TimeArray, ts,
                               sym::Symbol)
    @argcheck(TimeSeries.timestamp(A) == ts,
              ConflictingArgumentError("`$sym` is carried on the asset clock, and `price_ingestion` is what puts it there:\n\tlength(timestamp($sym)) => $(length(TimeSeries.timestamp(A)))\n\tlength(timestamp(X)) => $(length(ts))"))
    n = string.(TimeSeries.colnames(A))
    v = values(unify_gaps(A))
    for (j, nm) in pairs(n)
        P[!, nm] = v[:, j]
    end
    return n
end
"""
    prices_to_returns(
        pr::PricesResult;
        ret_method::Symbol = :simple,
        padding::Bool = false,
        gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing
    ) -> ReturnsResult
    prices_to_returns(
        X::TimeSeries.TimeArray;
        kwargs...
    ) -> ReturnsResult

Compute returns from the price carrier.

A keyword belongs to this function only when it changes the arithmetic of a return, and three keywords do. Every datum that the conversion reads is a field of the [`PricesResult`](@ref). These are the asset prices, the factors, the benchmark, the implied volatilities, the **Listing Span** and the [`AssetPanel`](@ref). A keyword for one of them would state again what the carrier states.

The second method takes a bare price table. It runs [`price_ingestion`](@ref) with a default [`PriceIngestion`](@ref), and converts the carrier that it returns. For a different join, a collapse, a declared span, or factor, benchmark and implied-volatility series, call the two steps.

An absent price is a `NaN`. The conversion carries it into the returns, and it does not delete the observation or the asset. [`PriceGapFill`](@ref) fills a gap and [`MissingDataFilter`](@ref) deletes one. Both are fitted steps.

# Mathematical definition

The conversion computes the returns from the prices as:

```math
\\begin{align}
x_{t,\\,i} &= \\begin{cases}
(p_{t,\\,i} - p_{t-1,\\,i}) / p_{t-1,\\,i} & \\text{simple} \\\\
\\ln(p_{t,\\,i} / p_{t-1,\\,i}) & \\text{log}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_ti_ret])
  - $(math_dict[:p_ti_price])

Both branches take the logarithm of each price, so a price must be non-negative. A zero price ``p_{t,\\,i} = 0`` gives ``x_{t,\\,i} = -1`` on the simple branch and ``x_{t,\\,i} = -\\infty`` on the log branch. On both branches it gives ``x_{t+1,\\,i} = \\infty``.

The conversion applies the same rule to a benchmark ``B``, and carries the benchmark returns ``b_{t,\\,i}`` **beside** the asset returns. It does not subtract them. [`returns_result_picker`](@ref) forms the excess return ``\\tilde{x}_{t,\\,i} = x_{t,\\,i} - b_{t,\\,i}``, and only when the optimisation tracks the benchmark.

# Algorithm

 1. Check with [`assert_distinct_series_names`](@ref) that the asset, factor and benchmark series have distinct names. Read the asset names and the asset timestamps from `pr.X`. Check `pr.pnl` against them with [`check_asset_panel`](@ref), and `pr.span` with [`assert_span_shape`](@ref).
 2. Write the three price blocks side by side on the clock of the carrier with [`append_carrier_block!`](@ref), which writes every absent price as `NaN` with [`unify_gaps`](@ref). The carrier states one clock. So the function reads a factor or benchmark series at the asset timestamps, and does not join it onto them. It refuses a series on a different clock and names the series, because a join adds or drops observations and [`price_ingestion`](@ref) owns every change of the clock. A benchmark is one shared column, or one column per asset.
 3. Convert the prices to returns with `TimeSeries.percentchange` under `ret_method` and `padding`. This step applies the formula above. It computes both branches through logarithms. The log return is ``\\ln p_{t,\\,i} - \\ln p_{t-1,\\,i}``, and the simple return is `expm1` of it. So the two agree with the closed forms above to floating point, and not always to the last bit. When `padding` is `true`, the step keeps the first observation with a `NaN` return, so the returns keep the length of the price clock.
 4. **A gap carried here does not spread.** The formula reads two prices, so a run of `k` gapped prices makes non-finite only the returns that read one of them. That is `k + 1` returns for a run inside the series, and `k` for a run at either end, because no return reads a price before the first row or after the last. Every later return of that column reads two observed prices. A gap also stays in its own column, because the return of an asset reads no price of another asset.
 5. Resolve the returns that a gap left non-finite with [`apply_gap_return`](@ref), under `gap_return_alg`. The method for `nothing` is the default rule. It returns the table unchanged, so the returns of step 3 stay bit for bit the same. An algorithm writes only a return that reads a gapped price, inside the Listing Span of its column, after the first observed price. So every return computed from two observed prices keeps its value. The function logs an `@info` when it finds no writable return.
 6. Name the three blocks. Step 1 refused a name that two tables share and the name `timestamp`. So the asset names `nx`, the factor names `nf` and the benchmark names `nb` are the column names of the three tables, and `ts` is the `timestamp` column of the converted table.
 7. Write each absent implied volatility as `NaN` with [`unify_gaps`](@ref), and index `pr.iv` by `ts`. Then check the implied volatilities and `pr.ivpa` against the asset count. The returns clock is the price clock, less the first observation when `padding` is `false`. So the implied volatilities of a carrier that the ingestion layer built cover it. The conversion carries an absent implied volatility as `NaN`, and the estimator that reads it excludes it.
 8. View the [`AssetPanel`](@ref) on the returns clock. Find the panel rows of the returns timestamps with [`feature_row_indices`](@ref), and view the panel with [`port_opt_view`](@ref). Give it the asset names, so that it also cuts a square tensor Panel Field on its label axis. The conversion removes no column, so the view keeps every asset. A time-varying panel loses only the observations that the returns clock does not hold.
 9. State the universe. Cut `pr.span` to the asset axis with [`span_carrier_view`](@ref). Give it and the asset returns to [`returns_universe_masks`](@ref), which projects the span onto the returns clock and intersects it with finiteness. [`attach_universe_masks`](@ref) puts the two masks on the Asset Panel and keeps its Panel Fields. When the carrier holds no panel, it makes a panel with no Panel Field. A carrier with no span states no universe, so the function attaches no masks, and the panel is the one of step 8, or `nothing`.
10. Build the factor and benchmark matrices from the columns of each block. A block with no column gives `nothing`, and a benchmark block with one column gives a vector. The asset matrix is always present, because the conversion removes no column.
11. Return the [`ReturnsResult`](@ref).

**The conversion removes no observation and no asset.** A deletion is a **Universe Policy**. A policy is fit on a training window and applied again by name, and a stateless conversion cannot do that. [`MissingDataFilter`](@ref) owns the deletion. Its `col_thr` deletes an asset and its `row_thr` deletes an observation.

# Arguments

  - `pr`: The price carrier, as [`price_ingestion`](@ref) returns it or a caller builds it.
  - `X`: Asset price data (observations × assets), converted through a default [`PriceIngestion`](@ref).
  - `ret_method`: The return formula, `:simple` or `:log`.
  - `padding`: When `true`, keep the first observation with a `NaN` return, so the returns keep the length of the price clock. When `false`, drop it.
  - `gap_return_alg`: The rule for the returns that a price gap leaves non-finite. `nothing` is the default rule, under which no return records the move across the gap. A run of `k` gapped prices inside the series then leaves `k + 1` non-finite returns, and a run at either end leaves `k`. [`CatchUpGapReturn`](@ref) books the move on the observation where the asset trades again, so the Held Gap holds `k` non-finite returns. An algorithm writes only a return that reads a gapped price, inside the Listing Span of its column, after the first observed price. So a return computed from two observed prices keeps its value under every algorithm. When no return is writable, the function logs an `@info`.

# Validation

  - `ret_method` is `:simple` or `:log`. `TimeSeries.percentchange` throws an `ArgumentError` otherwise.
  - Every price that reaches step 3 is non-negative. `TimeSeries.percentchange` takes a logarithm on both branches, so a negative price throws a `DomainError` from inside it, on the simple branch too. A zero price gives the returns that the mathematical definition states.
  - The asset, factor and benchmark column names are pairwise disjoint, and none of them is `timestamp`. The function throws a [`ConflictingArgumentError`](@ref) that names the columns otherwise.
  - If `pr.F` or `pr.B` is not `nothing`, its timestamps equal the asset timestamps. The function throws a [`ConflictingArgumentError`](@ref) otherwise. The message names [`price_ingestion`](@ref), which puts two series on one clock.
  - If `pr.span` is not `nothing`, `size(pr.span) == size(values(pr.X))`. The function throws a `DimensionMismatch` otherwise.
  - If `pr.iv` is not `nothing`, the returns timestamps are a subset of `TimeSeries.timestamp(pr.iv)`. The function throws an `ArgumentError` otherwise. Then, with `iv = values(unify_gaps(pr.iv)[ts])`, `!isempty(iv)` holds, `size(iv) == size(X)` holds, and every present value is finite and non-negative. An absent value is `NaN`, and [`assert_nonneg_where_present`](@ref) skips it.
  - The function checks `pr.ivpa` in the same branch, so only when `pr.iv` is given. `all(x -> x > 0, ivpa)` and `all(x -> isfinite(x), ivpa)` hold, and a vector `ivpa` has `length(ivpa) == size(iv, 2)`. The bound is strict, so the function refuses a zero adjustment.

# Returns

  - `rr::ReturnsResult`: The asset, factor and benchmark returns, their names, the returns timestamps, the implied volatilities and the Asset Panel.

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
   pnl ┼ AssetPanel
       │     pf ┼ Vector{PortfolioOptimisers.AbstractPanelField}: PortfolioOptimisers.AbstractPanelField[]
       │   amsk ┼ 2×2 PortfolioOptimisers.AllTrueMask
       │   emsk ┴ 2×2 PortfolioOptimisers.AllTrueMask
```

# Related

  - [`PricesResult`](@ref)
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
  - [`append_carrier_block!`](@ref)
  - [`unify_gaps`](@ref)
  - [`returns_universe_masks`](@ref)
  - [`attach_universe_masks`](@ref)
  - [`span_carrier_view`](@ref)
  - [`returns_result_picker`](@ref): subtracts the carried benchmark, and only when the optimisation tracks it.
"""
function prices_to_returns(pr::PricesResult; ret_method::Symbol = :simple,
                           padding::Bool = false,
                           gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing)::ReturnsResult
    assert_distinct_series_names(pr.X, pr.F, pr.B)
    asset_names = string.(TimeSeries.colnames(pr.X))
    asset_ts = TimeSeries.timestamp(pr.X)
    N = length(asset_names)
    check_asset_panel(pr.pnl, N, length(asset_ts), "the number of asset price columns")
    assert_span_shape(pr.span, length(asset_ts), N)
    P = DataFrames.DataFrame(values(unify_gaps(pr.X)), asset_names)
    DataFrames.insertcols!(P, 1, :timestamp => asset_ts)
    factor_names = append_carrier_block!(P, pr.F, asset_ts, :F)
    benchmark_names = append_carrier_block!(P, pr.B, asset_ts, :B)
    X = TimeSeries.percentchange(TimeSeries.TimeArray(P; timestamp = :timestamp),
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
    ts = X[!, :timestamp]
    iv = pr.iv
    ivpa = pr.ivpa
    if !isnothing(iv)
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)),
                  ArgumentError("ts must be a subset of the timestamps in iv"))
        iv = values(unify_gaps(iv)[ts])
        @argcheck(!isempty(iv), IsEmptyError)
        @argcheck(size(iv) == (DataFrames.DataAPI.nrow(X), N), DimensionMismatch)
        assert_nonneg_where_present(iv, :iv)
        assert_nonempty_gt0_finite_val(ivpa, :ivpa)
        if isa(ivpa, VecNum)
            @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
        end
    end
    #! The conversion removes no column, so the assets reach it in their original order and
    #! `acols` is the whole asset axis. It is still handed to the panel view, because it is
    #! what pairs a square tensor Panel Field's label axis with the assets it describes.
    acols = collect(1:N)
    pnl = pr.pnl
    if !isnothing(pnl)
        rows = feature_row_indices(pnl, ts, asset_ts)
        pnl = port_opt_view(pnl, rows, acols, asset_names)
    end
    #! The span is on the price clock and the masks are on the returns clock, and
    #! `universe_masks` does the crossing. Both padding conventions reach it, and it reads
    #! which from the two row counts.
    RX = Matrix(X[!, nx])
    amsk, emsk = returns_universe_masks(span_carrier_view(pr.span, asset_ts, asset_ts,
                                                          acols), RX)
    pnl = attach_universe_masks(pnl, amsk, emsk)
    F = isempty(nf) ? nothing : Matrix(X[!, nf])
    B = if isempty(nb)
        nothing
    else
        length(nb) == 1 ? X[!, nb[1]] : Matrix(X[!, nb])
    end
    return ReturnsResult(; ts = ts, nx = nx, X = RX, nf = isempty(nf) ? nothing : nf, F = F,
                         nb = isempty(nb) ? nothing : nb, B = B, iv = iv, ivpa = ivpa,
                         pnl = pnl)
end
function prices_to_returns(X::TimeSeries.TimeArray; kwargs...)::ReturnsResult
    return prices_to_returns(price_ingestion(PriceIngestion(), X); kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator that converts price-level data into returns-level data.

`PricesToReturns` is the estimator form of [`prices_to_returns`](@ref). It reads a [`PricesResult`](@ref) and returns a [`ReturnsResult`](@ref). It has no fitted state, so its application to a window runs the conversion, and its fitted object is the estimator itself.

Its first three fields are the three keywords of [`prices_to_returns`](@ref), which are the keywords that change the arithmetic of a return. A join and a collapse change the observation clock, and [`PriceIngestion`](@ref) owns them. [`PriceGapFill`](@ref) fills a gap and [`MissingDataFilter`](@ref) deletes one, and both are fitted steps. Every datum that the conversion reads is a field of the [`PricesResult`](@ref).

The step needs no state to fix an asset universe, because the carrier states one. A [`PricesResult`](@ref) that [`price_ingestion`](@ref) built carries a **Listing Span**. This step projects the span onto the returns clock, and gives the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) with two masks. At each observation, the masks state which assets are in the universe, and which of them are estimable. The asset axis is fixed before the split, so every window of every fold carries every asset, and no window loses a column.

!!! warning

    A carrier that the ingestion layer did not build states no universe, and the conversion does not infer one from the window. A span computed from the window alone reads a delisting that crosses the end of the window as an asset that was never listed. The conversion still carries the gaps of such a carrier. With no panel, the Coverage Universe reads finiteness alone, and the fold infers the universe that a span would state. Build the carrier with [`price_ingestion`](@ref), or give a listing calendar as its `span`.

# Algorithm

The estimator has no fitted state, so both verbs are short.

 1. [`fit_preprocessing`](@ref) returns the estimator itself. There is no state to fit.
 2. [`apply_preprocessing`](@ref) calls [`prices_to_returns`](@ref) with the whole [`PricesResult`](@ref) and the three fields as keywords, and returns the [`ReturnsResult`](@ref).

Every row and every column of the window reaches the conversion, because the conversion cannot drop one. `gap_return_alg` is a field because it sets the returns that a gap leaves non-finite. That is the arithmetic of a return, and not a policy about the universe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturns(;
        ret_method::Symbol = :simple,
        padding::Bool = false,
        gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> PricesToReturns

Keywords correspond to the struct's fields.

## Validation

  - `ret_method in (:simple, :log)`. The constructor throws an `ArgumentError` otherwise.

# Online form

A return reads two consecutive prices. So [`partial_fit_transform`](@ref) converts a block of prices as the whole history does, because it keeps the last price row in `cache`. Under a [`CatchUpGapReturn`](@ref), it also keeps the last observed price of every series column. [`fit_preprocessing`](@ref) with no data returns the estimator with an empty `cache`. A caller's own Gap Return algorithm has no online form, and [`supports_partial_fit`](@ref) returns `false` for it.

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
  - [`PriceIngestion`](@ref)
  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct PricesToReturns <: AbstractPreprocessingEstimator
    """
    The return formula, `:simple` or `:log`.
    """
    ret_method
    """
    When `true`, keep the first observation with a `NaN` return, so the returns keep the length of the price clock. When `false`, drop it.
    """
    padding
    """
    The rule for the returns that a price gap leaves non-finite. `nothing` is the default rule, and [`CatchUpGapReturn`](@ref) books the move across the gap on the observation that ends it. See [`AbstractGapReturnAlgorithm`](@ref).
    """
    gap_return_alg
    """
    $(field_dict[:pfcache])
    """
    cache
    function PricesToReturns(ret_method::Symbol, padding::Bool,
                             gap_return_alg::Option{<:AbstractGapReturnAlgorithm},
                             cache::Option{<:AbstractPartialFitState})
        @argcheck(ret_method in (:simple, :log),
                  ArgumentError("ret_method must be :simple or :log, got :$ret_method"))
        return new{typeof(ret_method), typeof(padding), typeof(gap_return_alg),
                   typeof(cache)}(ret_method, padding, gap_return_alg, cache)
    end
end
function PricesToReturns(; ret_method::Symbol = :simple, padding::Bool = false,
                         gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing,
                         cache::Option{<:AbstractPartialFitState} = nothing)::PricesToReturns
    return PricesToReturns(ret_method, padding, gap_return_alg, cache)
end
function prices_to_returns(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(pr; ret_method = ptr.ret_method, padding = ptr.padding,
                             gap_return_alg = ptr.gap_return_alg)
end
function fit_preprocessing(ptr::PricesToReturns, ::PricesResult)
    return ptr
end
function apply_preprocessing(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(ptr, pr)
end
export prices_to_returns, PricesToReturns, CatchUpGapReturn
public AbstractGapReturnAlgorithm, gap_return
