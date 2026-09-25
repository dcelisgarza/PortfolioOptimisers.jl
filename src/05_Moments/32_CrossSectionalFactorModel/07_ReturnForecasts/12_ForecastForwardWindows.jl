"""
    forecast_window_grid(horizon::Integer, lag::Integer, n::Integer,
                         kind::Symbol) -> Vector{Tuple{Int, Int}}

Return the forward windows that a holding-period table or a decay table reads.

Both tables score one Return Forecast against `n` forward windows, and they differ only in where the `p`-th window sits. `:cumulative` lengthens the window and keeps its start, so period `p` measures the forecast over a book held for `p` horizons. `:disjoint` keeps the length and moves the start, so period `p` measures what the forecast still predicts `p - 1` horizons after the first window it could act on. At `p = 1` both kinds give the base evaluation's own window.

# Mathematical definition

```math
\\begin{align}
\\text{cumulative:} \\quad & \\left[ t + \\ell,\\; t + \\ell + p h \\right)\\,, \\\\
\\text{disjoint:} \\quad & \\left[ t + \\ell + (p - 1) h,\\; t + \\ell + p h \\right)\\,.
\\end{align}
```

Where:

  - ``t``: The observation the forecast is scored at.
  - ``h``: The horizon of the base evaluation.
  - ``\\ell``: The lag of the base evaluation.
  - ``p``: The period, from ``1`` to ``n``.

Each pair is the `(horizon, lag)` that [`forward_mean_returns`](@ref) reads, and its window of observations is `t + lag` to `t + lag + horizon - 1`, the half-open interval above.

# Arguments

  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])
  - `n`: Number of forward periods.
  - `kind`: `:cumulative` for the holding-period table, `:disjoint` for the decay table.

# Validation

  - `n >= 1`. Raises a `DomainError`.
  - `kind` is `:cumulative` or `:disjoint`. Raises a [`ConflictingArgumentError`](@ref).

The function does not check `horizon` and `lag`. Both tables read them from an evaluation, and [`forecast_evaluation`](@ref) checks them there.

# Returns

  - `grid::Vector{Tuple{Int, Int}}`: One `(horizon, lag)` pair per period, in period order.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_window_grid(2, 1, 3, :cumulative)
3-element Vector{Tuple{Int64, Int64}}:
 (2, 1)
 (4, 1)
 (6, 1)

julia> PortfolioOptimisers.forecast_window_grid(2, 1, 3, :disjoint)
3-element Vector{Tuple{Int64, Int64}}:
 (2, 1)
 (2, 3)
 (2, 5)
```

# Related

  - [`forecast_holding_period`](@ref)
  - [`forecast_decay`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function forecast_window_grid(horizon::Integer, lag::Integer, n::Integer, kind::Symbol)
    @argcheck(n >= one(n), DomainError(n, "n must be >= 1"))
    @argcheck(kind in (:cumulative, :disjoint),
              ConflictingArgumentError("kind must be :cumulative or :disjoint, got :$(kind)"))
    h = Int(horizon)
    l = Int(lag)
    return if kind === :cumulative
        [(p * h, l) for p in 1:n]
    else
        [(h, l + (p - 1) * h) for p in 1:n]
    end
end
"""
    forecast_common_dates(alpha::MatNum, ys::AbstractVector{<:MatNum},
                          dates::AbstractVector{<:Integer},
                          min_count::Integer) -> Vector{Int}

Return the evaluation dates at which every forward window of a table can be scored.

A reader compares the rows of a table down the column, the row of period `10` with the row of period `1`. That comparison holds only when both rows use the same dates. A deeper window matures later, so it scores fewer dates at the end of the sample, and a row that dropped them would move for a reason that has nothing to do with the forecast. The two tables therefore intersect the dates across the whole grid before they take any statistic, and every row of a table reads the set this function returns.

A date stays when at least `min_count` assets carry a finite forecast and a finite target under every window. The Spearman coefficient and the two books count the same assets, so none of them is `NaN` for lack of assets on a date this function keeps. The Pearson coefficient and the coverage also drop an asset whose weight is zero or not finite, so under such a weight history the Pearson coefficient can be `NaN` on a kept date.

# Mathematical definition

```math
\\mathcal{D}_{c} = \\left\\{ t \\in \\mathcal{D} : \\left\\lvert \\left\\{ i : \\alpha_{ti} \\text{ and } y^{(p)}_{ti} \\text{ are finite} \\right\\} \\right\\rvert \\geq m \\text{ for } p = 1, \\ldots, P \\right\\}\\,.
```

Where:

  - $(math_dict[:alpha_ti_fc])
  - ``y^{(p)}_{ti}``: Forward target of asset ``i`` at observation ``t`` under the ``p``-th window, `ys[p]`.
  - ``\\mathcal{D}``: The dates of the base evaluation, `dates`.
  - ``P``: The number of windows, `length(ys)`.
  - ``m``: The least number of assets, `min_count`.
  - ``\\mathcal{D}_{c}``: The common dates.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `ys`: One forward target `observations × assets` per window, on the axis of `alpha`.
  - `dates`: Row indices of the observations the base evaluation scores.
  - `min_count`: Least number of assets a cross-section needs at a date, under every window.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.

# Returns

  - `common::Vector{Int}`: The common dates, in the order of `dates`. It is empty when no date stays, which is the answer for a grid deeper than the sample.

# Examples

```jldoctest
julia> alpha = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0];

julia> y1 = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> y2 = PortfolioOptimisers.forward_mean_returns(alpha, 2, 1);

julia> PortfolioOptimisers.forecast_common_dates(alpha, [y1, y2], [1, 2, 3], 2)
2-element Vector{Int64}:
 1
 2
```

# Related

  - [`forecast_holding_period`](@ref)
  - [`forecast_decay`](@ref)
  - [`forecast_window_grid`](@ref)
"""
function forecast_common_dates(alpha::MatNum, ys::AbstractVector{<:MatNum},
                               dates::AbstractVector{<:Integer}, min_count::Integer)
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    common = Int[]
    for t in dates
        keep = true
        for y in ys
            n = 0
            for i in axes(alpha, 2)
                n += isfinite(alpha[t, i]) && isfinite(y[t, i])
            end
            if n < min_count
                keep = false
                break
            end
        end
        if keep
            push!(common, Int(t))
        end
    end
    return common
end
"""
    forecast_window_row(fp::ForecastEvaluationResult, w::Option{<:MatNum},
                        min_count::Integer) -> NamedTuple

Return the eleven figures that one forward window adds to a table.

The row computes no statistic of its own. It reads [`forecast_ic`](@ref) and [`forecast_ic_summary`](@ref) for the two information coefficients, [`forecast_portfolio`](@ref) for each of the two books, and [`forecast_coverage`](@ref) for the share of the universe the window scored. A window whose evaluation carries no date gives `NaN` for every figure and raises nothing, because a table deeper than the sample must still print.

In [`forecast_window_table`](@ref) the portfolio series of a row has no gap. [`forecast_common_dates`](@ref) kept every date of `fp` because at least `min_count` assets carry a finite pair there under this window, so the compressed path that [`forecast_portfolio`](@ref) summarises is the whole path, and its caveat about `max_drawdown` and `calmar` does not apply.

# Algorithm

 1. If `fp.dates` is empty, give `NaN` for every figure.
 2. Score the two coefficients with [`forecast_ic`](@ref) at `min_count`, and summarise them with [`forecast_ic_summary`](@ref) at the [`forecast_ic_lags`](@ref) of `fp`.
 3. Build the `:rank` book and the `:zscore` book with [`forecast_portfolio`](@ref).
 4. Take the mean of the finite entries of [`forecast_coverage`](@ref), or `NaN` when no entry is finite.
 5. Convert every figure to the element type that `fp.alpha` and `fp.y` promote to.

# Arguments

  - `fp`: The evaluation of one forward window, on the table's common dates.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fp.alpha`, or `nothing` for equal weights.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported.

# Validation

  - The rules of [`forecast_ic`](@ref), [`forecast_portfolio`](@ref) and [`forecast_coverage`](@ref), when `fp.dates` is not empty.

# Returns

  - `row::NamedTuple`: `(; spearman_mean_ic, spearman_ic_ir, spearman_t_stat, pearson_mean_ic, pearson_ic_ir, pearson_t_stat, rank_ann_return, rank_sharpe, zscore_ann_return, zscore_sharpe, mean_coverage)`. Every figure has the element type that `fp.alpha` and `fp.y` promote to, whatever the element type of `w`.

# Related

  - [`forecast_holding_period`](@ref)
  - [`forecast_decay`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_portfolio`](@ref)
  - [`forecast_coverage`](@ref)
"""
function forecast_window_row(fp::ForecastEvaluationResult, w::Option{<:MatNum},
                             min_count::Integer)
    alpha::MatNum = fp.alpha
    y::MatNum = fp.y
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)))
    if isempty(fp.dates)
        nan = Tf(NaN)
        return (; spearman_mean_ic = nan, spearman_ic_ir = nan, spearman_t_stat = nan,
                pearson_mean_ic = nan, pearson_ic_ir = nan, pearson_t_stat = nan,
                rank_ann_return = nan, rank_sharpe = nan, zscore_ann_return = nan,
                zscore_sharpe = nan, mean_coverage = nan)
    end
    ic = forecast_ic_summary(forecast_ic(fp, w; min_count = min_count);
                             lags = forecast_ic_lags(fp))
    rk = forecast_portfolio(fp; kind = :rank)
    zs = forecast_portfolio(fp; kind = :zscore)
    c = forecast_coverage(fp, w)
    fc = c[isfinite.(c)]
    return (; spearman_mean_ic = Tf(ic.spearman.mean_ic),
            spearman_ic_ir = Tf(ic.spearman.ic_ir),
            spearman_t_stat = Tf(ic.spearman.t_stat),
            pearson_mean_ic = Tf(ic.pearson.mean_ic), pearson_ic_ir = Tf(ic.pearson.ic_ir),
            pearson_t_stat = Tf(ic.pearson.t_stat),
            rank_ann_return = Tf(rk.summary.ann_return),
            rank_sharpe = Tf(rk.summary.sharpe),
            zscore_ann_return = Tf(zs.summary.ann_return),
            zscore_sharpe = Tf(zs.summary.sharpe),
            mean_coverage = isempty(fc) ? Tf(NaN) : Tf(sum(fc) / length(fc)))
end
"""
    forecast_window_table(fe::ForecastEvaluationResult, X::MatNum, w::Option{<:MatNum},
                          grid::AbstractVector{<:Tuple{Integer, Integer}},
                          min_count::Integer) -> NamedTuple

Score one Return Forecast against a grid of forward windows, on one common date set.

[`forecast_holding_period`](@ref) and [`forecast_decay`](@ref) both call this function, and they differ only in the grid that [`forecast_window_grid`](@ref) gives it. This docstring states the date rule, the statistics and the shape of the result once, for both.

# Algorithm

 1. Check `w` against the shape of `fe.alpha`, so a wrong weight history raises at every depth, also when no date stays.
 2. Build the forward target of each window from `X` with [`forward_mean_returns`](@ref).
 3. Intersect the dates of the base evaluation across the whole grid with [`forecast_common_dates`](@ref).
 4. For each window, build a [`ForecastEvaluationResult`](@ref) that carries `fe.alpha` and `fe.umsk`, the window's target, the common dates, the window's own `horizon` and `lag`, and `min_count`, and read its row with [`forecast_window_row`](@ref).
 5. Transpose the rows into columns, one entry per period.

# Arguments

  - `fe`: The base evaluation, from [`forecast_evaluation`](@ref). Its `alpha`, `umsk`, `target`, `step` and `ppy` go into every window, and its `dates` are the dates the common set is taken from.
  - `X`: The target history `observations × assets` the forward windows are taken over, on the axis of `fe.alpha`. It is what [`forecast_target_history`](@ref) gives for `fe.target`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `grid`: One `(horizon, lag)` pair per period, from [`forecast_window_grid`](@ref).
  - `min_count`: Least number of assets a cross-section needs, for the date rule and for the statistics alike.

# Validation

  - `size(X) == size(fe.alpha)`. Raises a `DimensionMismatch`.
  - `!isempty(grid)`. Raises an [`IsEmptyError`](@ref).
  - `size(w) == size(fe.alpha)`, when `w` is not `nothing`. Raises a `DimensionMismatch`.
  - No finite weight of `w` is negative. Raises a `DomainError`.
  - The rules of [`forecast_common_dates`](@ref) and [`forecast_window_row`](@ref).

# Returns

  - `table::NamedTuple`: The columns [`forecast_holding_period`](@ref) documents.

# Related

  - [`forecast_holding_period`](@ref)
  - [`forecast_decay`](@ref)
  - [`forecast_window_grid`](@ref)
  - [`forecast_window_row`](@ref)
  - [`forecast_common_dates`](@ref)
"""
function forecast_window_table(fe::ForecastEvaluationResult, X::MatNum, w::Option{<:MatNum},
                               grid::AbstractVector{<:Tuple{Integer, Integer}},
                               min_count::Integer)
    alpha::MatNum = fe.alpha
    @argcheck(size(X, 1) == size(alpha, 1) && size(X, 2) == size(alpha, 2),
              DimensionMismatch("the target history ($(size(X, 1))×$(size(X, 2))) must match the Return Forecast history ($(size(alpha, 1))×$(size(alpha, 2)))"))
    @argcheck(!isempty(grid), IsEmptyError("grid cannot be empty"))
    forecast_ic_weights(alpha, w)
    ys = [forward_mean_returns(X, h, l) for (h, l) in grid]
    common = forecast_common_dates(alpha, ys, fe.dates, min_count)
    rows = [forecast_window_row(ForecastEvaluationResult(alpha, ys[p], fe.umsk, common,
                                                         fe.target, grid[p][1], grid[p][2],
                                                         fe.step, min_count, fe.ppy), w,
                                min_count) for p in eachindex(grid)]
    return (; period = collect(eachindex(grid)), horizon = [g[1] for g in grid],
            lag = [g[2] for g in grid], dates = common,
            spearman_mean_ic = [r.spearman_mean_ic for r in rows],
            spearman_ic_ir = [r.spearman_ic_ir for r in rows],
            spearman_t_stat = [r.spearman_t_stat for r in rows],
            pearson_mean_ic = [r.pearson_mean_ic for r in rows],
            pearson_ic_ir = [r.pearson_ic_ir for r in rows],
            pearson_t_stat = [r.pearson_t_stat for r in rows],
            rank_ann_return = [r.rank_ann_return for r in rows],
            rank_sharpe = [r.rank_sharpe for r in rows],
            zscore_ann_return = [r.zscore_ann_return for r in rows],
            zscore_sharpe = [r.zscore_sharpe for r in rows],
            mean_coverage = [r.mean_coverage for r in rows])
end
"""
    forecast_holding_period(fe::ForecastEvaluationResult, X::MatNum,
                            w::Option{<:MatNum} = nothing; n::Integer = 10,
                            min_count::Integer = fe.min_count) -> NamedTuple
    forecast_holding_period(fe::ForecastEvaluationResult, rd::ReturnsResult,
                            csfm::CrossSectionalFactorModel;
                            weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                            n::Integer = 10,
                            min_count::Integer = fe.min_count) -> NamedTuple

Score a Return Forecast against cumulative forward windows, one row per holding period.

The base evaluation measures the forecast over a book held for one horizon. This function measures it over `p` horizons held end to end, for `p` from `1` to `n`, and a caller reads the shape of that column to choose a rebalancing frequency. A coefficient that holds as the window lengthens says the forecast is about a slow quantity. A coefficient that falls says the book must turn over to capture it.

# Every row uses one date set

A deeper window matures later, so it scores fewer dates at the end of the sample. [`forecast_common_dates`](@ref) therefore intersects the dates across the whole grid before the table takes any statistic, so a fall down the column comes from the forecast and not from a change of sample.

The set shrinks as `n` grows. The rows of one table are comparable with each other, but a table at one depth is not comparable with a table at another depth. To compare two depths, read the deeper table and truncate it.

`n` is a keyword and not a field of the evaluation, so a caller reads the table at a second depth without a second pairing to build `fe`, which can cost a rolling refit.

# The t-statistic of a deeper row corrects for the overlap

The stride between two dates is the base evaluation's, and it does not grow with the window. From the second row on, consecutive windows therefore share returns. At the default stride, which is `fe.horizon`, the window of row `p` overlaps the windows of its `p - 1` neighbours on each side. When the forecast keeps its ranking from one date to the next, the consecutive coefficients of a deep row are then correlated, and for a forecast with no skill a t-statistic that treated them as independent would spread by about ``\\sqrt{p}`` down the column. A forecast that draws a new ranking at every date gives uncorrelated coefficients at every depth. The `t_stat` columns use the long-run standard error over [`forecast_ic_lags`](@ref) autocovariances at each row, as [`forecast_ic_summary`](@ref) states, so a t-statistic that holds down the column means that the forecast holds. The `ic_ir` columns are the per-date ratio, with no correction.

# Algorithm

 1. Build the grid with [`forecast_window_grid`](@ref) at `:cumulative`, giving the windows ``[t + \\ell,\\, t + \\ell + p h)``.
 2. Score it with [`forecast_window_table`](@ref).

# Arguments

  - `fe`: The base evaluation, from [`forecast_evaluation`](@ref). Its `horizon` and `lag` set the grid, its `alpha`, `umsk`, `target`, `step` and `ppy` go into every window, and its `dates` are the dates the common set is taken from.
  - `X`: The target history `observations × assets` the forward windows are taken over, on the axis of `fe.alpha`. The block method builds it from `fe.target` with [`forecast_target_history`](@ref). The bare method takes it, so a caller can score a history that the library did not build.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights. Only the Pearson coefficient and the coverage read it.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block the evaluation was built on.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref), resolved by [`cs_diagnostic_weights`](@ref).
  - `n`: Number of forward periods, the depth of the table.
  - `min_count`: Least number of assets a cross-section needs. It defaults to the threshold the evaluation carries, and it applies to the common dates and to the statistics alike.

# Validation

  - The rules of [`forecast_window_grid`](@ref) and [`forecast_window_table`](@ref), and, for the block method, of [`forecast_target_history`](@ref) and [`cs_diagnostic_weights`](@ref).

# Returns

  - `table::NamedTuple`: Fifteen columns. `dates` holds the common dates, and each of the other fourteen holds one entry per period.

      + `period::Vector{Int}`: `1` to `n`.
      + `horizon::Vector{Int}`: The window's horizon, `p` times `fe.horizon`.
      + `lag::Vector{Int}`: The window's lag, `fe.lag` at every period.
      + `dates::Vector{Int}`: The common evaluation dates every row was computed on.
      + `spearman_mean_ic`, `spearman_ic_ir`, `spearman_t_stat`: The Spearman coefficient's summary, from [`forecast_ic_summary`](@ref), with the t-statistic at the row's own [`forecast_ic_lags`](@ref).
      + `pearson_mean_ic`, `pearson_ic_ir`, `pearson_t_stat`: The Pearson coefficient's summary.
      + `rank_ann_return`, `rank_sharpe`: The annualised return and the ratio of the `:rank` book, from [`forecast_portfolio`](@ref).
      + `zscore_ann_return`, `zscore_sharpe`: The same two figures of the `:zscore` book.
      + `mean_coverage`: The mean of the finite entries of [`forecast_coverage`](@ref).

# Examples

```jldoctest
julia> alpha = [1.0 2.0 3.0; 3.0 1.0 2.0; 2.0 3.0 1.0; 1.0 3.0 2.0; 2.0 1.0 3.0];

julia> fe = forecast_evaluation(alpha, PortfolioOptimisers.forward_mean_returns(alpha, 1, 1));

julia> t = forecast_holding_period(fe, alpha; n = 2);

julia> t.horizon
2-element Vector{Int64}:
 1
 2

julia> t.dates
3-element Vector{Int64}:
 1
 2
 3
```

# Related

  - [`forecast_decay`](@ref)
  - [`forecast_window_grid`](@ref)
  - [`forecast_window_table`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`forecast_target_history`](@ref)
"""
function forecast_holding_period(fe::ForecastEvaluationResult, X::MatNum,
                                 w::Option{<:MatNum} = nothing; n::Integer = 10,
                                 min_count::Integer = fe.min_count)
    return forecast_window_table(fe, X, w,
                                 forecast_window_grid(fe.horizon, fe.lag, n, :cumulative),
                                 min_count)
end
function forecast_holding_period(fe::ForecastEvaluationResult, rd::ReturnsResult,
                                 csfm::CrossSectionalFactorModel;
                                 weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                 n::Integer = 10, min_count::Integer = fe.min_count)
    return forecast_holding_period(fe, forecast_target_history(fe.target, rd, csfm),
                                   cs_diagnostic_weights(weighting, csfm); n = n,
                                   min_count = min_count)
end
"""
    forecast_decay(fe::ForecastEvaluationResult, X::MatNum,
                   w::Option{<:MatNum} = nothing; n::Integer = 10,
                   min_count::Integer = fe.min_count) -> NamedTuple
    forecast_decay(fe::ForecastEvaluationResult, rd::ReturnsResult,
                   csfm::CrossSectionalFactorModel;
                   weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                   n::Integer = 10, min_count::Integer = fe.min_count) -> NamedTuple

Score a Return Forecast against disjoint forward windows, one row per period out.

[`forecast_holding_period`](@ref) lengthens the window, and this function moves it. Period `p` scores the same forecast against the horizon that starts `p - 1` horizons after the first window the forecast could act on, so the column shows how fast the forecast decays. A coefficient that is large at `p = 1` and near zero at `p = 2` says that the forecast is used up within one horizon, and a delay of one horizon loses it. A coefficient that persists says that a later entry still earns a return.

The two tables agree at `p = 1`, which is the base evaluation's own window, and they differ from `p = 2` on. Both use the dates at which every window can be scored, so the rows of one table are comparable with each other, and two tables of different depth are not.

# Algorithm

 1. Build the grid with [`forecast_window_grid`](@ref) at `:disjoint`, giving the windows ``[t + \\ell + (p - 1) h,\\, t + \\ell + p h)``.
 2. Score it with [`forecast_window_table`](@ref).

# Arguments

  - `fe`: The base evaluation, from [`forecast_evaluation`](@ref). Its `horizon` and `lag` set the grid.
  - `X`: The target history `observations × assets` the forward windows are taken over, on the axis of `fe.alpha`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block the evaluation was built on.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref), resolved by [`cs_diagnostic_weights`](@ref).
  - `n`: Number of forward periods, the depth of the table.
  - `min_count`: Least number of assets a cross-section needs. It applies to the common dates and to the statistics alike.

# Validation

  - The rules of [`forecast_window_grid`](@ref) and [`forecast_window_table`](@ref), and, for the block method, of [`forecast_target_history`](@ref) and [`cs_diagnostic_weights`](@ref).

# Returns

  - `table::NamedTuple`: The columns [`forecast_holding_period`](@ref) documents. `horizon` is `fe.horizon` at every period and `lag` is `fe.lag + (p - 1) * fe.horizon`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 3.0; 3.0 1.0 2.0; 2.0 3.0 1.0; 1.0 3.0 2.0; 2.0 1.0 3.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> t = forecast_decay(forecast_evaluation(y, y), alpha; n = 2);

julia> t.lag
2-element Vector{Int64}:
 1
 2

julia> t.spearman_mean_ic[1]
1.0
```

The forecast is the forward target itself, so the first period scores a perfect coefficient,
and the second scores what is left of it one horizon later.

# Related

  - [`forecast_holding_period`](@ref)
  - [`forecast_window_grid`](@ref)
  - [`forecast_window_table`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_decay(fe::ForecastEvaluationResult, X::MatNum,
                        w::Option{<:MatNum} = nothing; n::Integer = 10,
                        min_count::Integer = fe.min_count)
    return forecast_window_table(fe, X, w,
                                 forecast_window_grid(fe.horizon, fe.lag, n, :disjoint),
                                 min_count)
end
function forecast_decay(fe::ForecastEvaluationResult, rd::ReturnsResult,
                        csfm::CrossSectionalFactorModel;
                        weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                        n::Integer = 10, min_count::Integer = fe.min_count)
    return forecast_decay(fe, forecast_target_history(fe.target, rd, csfm),
                          cs_diagnostic_weights(weighting, csfm); n = n,
                          min_count = min_count)
end

export forecast_holding_period, forecast_decay
