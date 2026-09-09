"""
    forecast_window_grid(horizon::Integer, lag::Integer, n::Integer,
                         kind::Symbol) -> Vector{Tuple{Int, Int}}

Return the forward windows a holding-period or a decay table is read over.

Both tables score one Return Forecast against `n` forward windows instead of one, and they differ only in how the `p`-th window is placed. `:cumulative` lengthens the window and holds its start, so period `p` asks what the forecast is worth to a caller who holds the book for `p` horizons. `:disjoint` holds the length and pushes the start out, so period `p` asks what is left of the forecast `p - 1` horizons after it could first be acted on. The two coincide at `p = 1`, which is the base evaluation's own window.

# Mathematical definition

```math
\\begin{align}
\\text{cumulative:} \\quad & \\left[ t + \\ell,\\; t + \\ell + p h \\right)\\,, \\\\
\\text{disjoint:} \\quad & \\left[ t + \\ell + (p - 1) h,\\; t + \\ell + p h \\right)\\,.
\\end{align}
```

Where:

  - ``h``: The horizon of the base evaluation.
  - ``\\ell``: The lag of the base evaluation.
  - ``p``: The period, from ``1`` to ``n``.

# Arguments

  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])
  - `n`: Number of forward periods.
  - `kind`: `:cumulative` for the holding-period table, `:disjoint` for the decay table.

# Validation

  - `n >= 1`. Raises a `DomainError`.
  - `kind` is `:cumulative` or `:disjoint`. Raises a [`ConflictingArgumentError`](@ref).

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

Return the evaluation dates every forward window of a table can be scored at.

A table's rows are read down the column, so the row of period `10` is compared against the row of period `1`. That comparison means nothing unless both rows were computed on the same dates: a deeper window matures later, so it scores fewer dates at the end of the sample, and a row that quietly dropped them would fall or rise for a reason that is arithmetic rather than economic. The dates are therefore intersected across the whole grid **before** any statistic is taken, and every row of the table answers on the set this verb returns.

A date survives when at least `min_count` assets carry a finite forecast and a finite target there **under every window**, which is the same threshold the statistics themselves apply, so no surviving date is silenced by a row it was kept for.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `ys`: One forward target `observations × assets` per window, on the axis of `alpha`.
  - `dates`: Row indices of the observations the base evaluation scores.
  - `min_count`: Least number of assets a cross-section needs at a date, under every window.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.

# Returns

  - `common::Vector{Int}`: The surviving dates, in increasing order. It is empty when no date survives, which is what a grid deeper than the sample answers.

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

Return the eleven figures one forward window contributes to a table.

The row computes no statistic of its own: it reads [`forecast_ic`](@ref) and its summary for the two information coefficients, [`forecast_portfolio`](@ref) for each of the two books, and [`forecast_coverage`](@ref) for the share of the universe the window scored. A window whose evaluation carries no date answers `NaN` throughout rather than raising, because a table is read as a whole and a grid deeper than the sample must still print.

The portfolio series of a row carries no gap. Every date of `fp` was kept by [`forecast_common_dates`](@ref) precisely because it reaches `min_count` under this window, so the compressed path [`forecast_portfolio`](@ref) summarises is the whole path, and the caveat that verb states does not bite here.

# Arguments

  - `fp`: The evaluation of one forward window, on the table's common dates.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fp.alpha`, or `nothing` for equal weights.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported.

# Validation

  - The rules of [`forecast_ic`](@ref), [`forecast_portfolio`](@ref) and [`forecast_coverage`](@ref).

# Returns

  - `row::NamedTuple`: `(; spearman_mean_ic, spearman_ic_ir, spearman_t_stat, pearson_mean_ic, pearson_ic_ir, pearson_t_stat, rank_ann_return, rank_sharpe, zscore_ann_return, zscore_sharpe, mean_coverage)`.

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
    ic = forecast_ic_summary(forecast_ic(fp, w; min_count = min_count))
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
            mean_coverage = isempty(fc) ? Tf(NaN) : sum(fc) / length(fc))
end
"""
    forecast_window_table(fe::ForecastEvaluationResult, X::MatNum, w::Option{<:MatNum},
                          grid::AbstractVector{<:Tuple{Integer, Integer}},
                          min_count::Integer) -> NamedTuple

Score one Return Forecast against a grid of forward windows, on one common date set.

This is the kernel both [`forecast_holding_period`](@ref) and [`forecast_decay`](@ref) run: the two verbs differ only in the grid [`forecast_window_grid`](@ref) hands it, so the date rule, the statistics and the shape of the answer are stated once, here.

# Algorithm

 1. Build each window's forward target from `X` with [`forward_mean_returns`](@ref).
 2. Intersect the base evaluation's dates across the whole grid with [`forecast_common_dates`](@ref).
 3. For each window, rebuild a [`ForecastEvaluationResult`](@ref) carrying `fe.alpha`, that window's target, the common dates and that window's own `horizon` and `lag`, and read its row with [`forecast_window_row`](@ref).
 4. Transpose the rows into columns, one entry per period.

# Arguments

  - `fe`: The base evaluation, from [`forecast_evaluation`](@ref). Its `alpha`, `target`, `dates`, `step` and `ppy` are carried into every window.
  - `X`: The target history `observations × assets` the forward windows are taken over, on the axis of `fe.alpha`. It is what [`forecast_target_history`](@ref) answers for `fe.target`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `grid`: One `(horizon, lag)` pair per period, from [`forecast_window_grid`](@ref).
  - `min_count`: Least number of assets a cross-section needs, applied to the date rule and to the statistics alike.

# Validation

  - `size(X) == size(fe.alpha)`. Raises a `DimensionMismatch`.
  - `!isempty(grid)`. Raises an [`IsEmptyError`](@ref).
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
    ys = [forward_mean_returns(X, h, l) for (h, l) in grid]
    common = forecast_common_dates(alpha, ys, fe.dates, min_count)
    rows = [forecast_window_row(ForecastEvaluationResult(alpha, ys[p], common, fe.target,
                                                         grid[p][1], grid[p][2], fe.step,
                                                         min_count, fe.ppy), w, min_count)
            for p in eachindex(grid)]
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

The base evaluation answers what the forecast is worth to a caller who holds a book for one horizon. This verb answers what it is worth over `p` horizons held end to end, for `p` from `1` to `n`, and the shape of that column is what a caller reads to choose a rebalancing frequency: a coefficient that holds as the window lengthens says the forecast is about a slow quantity, and one that falls away says the book must be turned over to capture it.

# Every row is read on one date set

A deeper window matures later, so it scores fewer dates at the end of the sample. The dates are therefore intersected across the whole grid by [`forecast_common_dates`](@ref) before any statistic is taken, so a fall down the column is the forecast decaying rather than the sample changing under it.

That set shrinks as `n` grows, so a table read at one depth is internally comparable and is **not** comparable to a table read at another. A caller who compares two depths reads the deeper table and truncates it.

`n` is a keyword rather than a field of the evaluation, so a caller re-reads the table at a second depth without repeating the pairing that produced `fe`, which can cost a rolling refit.

# Algorithm

 1. Build the grid with [`forecast_window_grid`](@ref) at `:cumulative`, giving the windows ``[t + \\ell,\\, t + \\ell + p h)``.
 2. Score it with [`forecast_window_table`](@ref).

# Arguments

  - `fe`: The base evaluation, from [`forecast_evaluation`](@ref). Its `horizon` and `lag` set the grid, and its `alpha`, `target`, `dates`, `step` and `ppy` are carried into every window.
  - `X`: The target history `observations × assets` the forward windows are taken over, on the axis of `fe.alpha`. The block method builds it from `fe.target` with [`forecast_target_history`](@ref); the bare method takes it, so a caller scores a history the library did not build.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights. Only the Pearson coefficient and the coverage read it.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block the evaluation was built on.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref), resolved by [`cs_diagnostic_weights`](@ref).
  - `n`: Number of forward periods, the depth of the table.
  - `min_count`: Least number of assets a cross-section needs. It defaults to the threshold the evaluation carries, and it gates the common dates and the statistics alike.

# Validation

  - The rules of [`forecast_window_grid`](@ref) and [`forecast_window_table`](@ref), and, for the block method, of [`forecast_target_history`](@ref) and [`cs_diagnostic_weights`](@ref).

# Returns

  - `table::NamedTuple`: Fifteen columns, the last eleven of them one entry per period.

      + `period::Vector{Int}`: `1` to `n`.
      + `horizon::Vector{Int}`: The window's horizon, `p` times `fe.horizon`.
      + `lag::Vector{Int}`: The window's lag, `fe.lag` at every period.
      + `dates::Vector{Int}`: The common evaluation dates every row was computed on.
      + `spearman_mean_ic`, `spearman_ic_ir`, `spearman_t_stat`: The Spearman coefficient's summary, from [`forecast_ic_summary`](@ref).
      + `pearson_mean_ic`, `pearson_ic_ir`, `pearson_t_stat`: The Pearson coefficient's summary.
      + `rank_ann_return`, `rank_sharpe`: The `:rank` book's annualised return and its ratio, from [`forecast_portfolio`](@ref).
      + `zscore_ann_return`, `zscore_sharpe`: The `:zscore` book's.
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

Where [`forecast_holding_period`](@ref) lengthens the window, this verb slides it: period `p` scores the same forecast against the horizon that begins `p - 1` horizons after the first one the forecast could be acted on. The column is the forecast's half-life. A coefficient that is large at `p = 1` and gone by `p = 2` says the information is consumed within one horizon and any delay in acting on it throws the forecast away; one that persists says a later entry still earns.

The two tables agree at `p = 1`, which is the base evaluation's own window, and they answer different questions from `p = 2` on. Both are computed on the dates every window can be scored at, so the rows are comparable down the table and not across two depths.

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
  - `min_count`: Least number of assets a cross-section needs. It gates the common dates and the statistics alike.

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

The forecast is the forward target itself, so the first period scores a perfect coefficient
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
