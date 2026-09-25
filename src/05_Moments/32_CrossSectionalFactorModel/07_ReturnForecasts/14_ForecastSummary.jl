"""
    forecast_summary_names(names::Nothing, n::Integer) -> Vector{String}
    forecast_summary_names(names::AbstractVector{<:AbstractString},
                           n::Integer) -> Vector{String}

Return the names axis of a forecast summary, one name per evaluation.

A reader reads a summary as a table, and a row without a name tells the reader little. An evaluation carries the pairing and its parameters but no name, so the caller supplies the names. The method for `nothing` numbers the rows in the order of the evaluations.

# Arguments

  - `names`: One name per evaluation, or `nothing` to number them.
  - `n`: Number of evaluations the summary carries.

# Validation

  - `length(names) == n`. Raises a `DimensionMismatch`.

# Returns

  - `names::Vector{String}`: One name per evaluation, in the order of the evaluations.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`ForecastSummaryResult`](@ref)
"""
function forecast_summary_names(::Nothing, n::Integer)
    return ["Forecast $(k)" for k in 1:n]
end
function forecast_summary_names(names::AbstractVector{<:AbstractString}, n::Integer)
    @argcheck(length(names) == n,
              DimensionMismatch("names carries $(length(names)) entries and the summary carries $(n) evaluation(s)"))
    return String[String(nm) for nm in names]
end
"""
    forecast_summary_same_target(a::AbstractForecastTarget,
                                 b::AbstractForecastTarget) -> Bool

Return whether two evaluations scored against the same forward target.

Two evaluations are comparable only when they answer for the same quantity, and that quantity is the [`AbstractForecastTarget`](@ref) of the pairing. The test reads the member and its fields. [`PanelFieldTarget`](@ref) names a field of the panel, so two of them are the same target when they name the same field, and different targets when they do not.

The library defines no equality on a target, so the test compares the fields one by one, in declaration order, with `isequal`.

# Arguments

  - `a`: The forward target of the first evaluation.
  - `b`: The forward target of the second evaluation.

# Returns

  - `same::Bool`: `true` when the two targets are of one type and every pair of their fields is equal under `isequal`.

# Related

  - [`forecast_summary_assert_same_question`](@ref)
  - [`AbstractForecastTarget`](@ref)
  - [`PanelFieldTarget`](@ref)
"""
function forecast_summary_same_target(a::AbstractForecastTarget, b::AbstractForecastTarget)
    if Base.typename(typeof(a)) !== Base.typename(typeof(b))
        return false
    end
    for i in 1:nfields(a)
        if !isequal(getfield(a, i), getfield(b, i))
            return false
        end
    end
    return true
end
"""
    forecast_summary_assert_same_question(fes::AbstractVector{<:ForecastEvaluationResult})

Refuse a set of evaluations that do not answer one question, and name the field that differs.

A summary puts one row per forecast beside the others, so the rows must answer one question. Six fields of a [`ForecastEvaluationResult`](@ref) define the question:

  - the forward target sets the quantity that the evaluation scores,
  - `horizon` and `lag` set the window of that quantity,
  - `step` sets the stride of the sample,
  - `min_count` sets which cross-sections enter the sample,
  - `ppy` sets the units of the annualised columns.

The check refuses a difference in any of them, because the summary would otherwise report that difference as a difference in skill.

The universe is part of the question too. Two forecasts over two universes answer two questions, even when their parameters agree. So the asset axis must agree, and so must `umsk`, which is the denominator of every coverage column of the summary. The check compares neither `alpha` nor `y`. The forecasts differ by design, because the summary exists to compare them.

The check does not compare the evaluation `dates` either. Two evaluations that answer one question on two grids still answer one question. [`forecast_evaluation_align`](@ref) puts them on one grid, and [`forecast_summary_assert_comparable`](@ref) asks for one grid as well.

# Arguments

  - `fes`: The evaluations to compare, at least one.

# Validation

  - `!isempty(fes)`. Raises an [`IsEmptyError`](@ref).
  - Every evaluation agrees with the first on `target`, `horizon`, `lag`, `step`, `min_count`, `ppy`, the number of assets and `umsk`. Raises a [`ConflictingArgumentError`](@ref) that names the field.

# Returns

  - `nothing`.

# Related

  - [`forecast_summary_assert_comparable`](@ref)
  - [`forecast_evaluation_align`](@ref)
  - [`forecast_summary_same_target`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_summary_assert_same_question(fes::AbstractVector{<:ForecastEvaluationResult})
    @argcheck(!isempty(fes), IsEmptyError("fes cannot be empty"))
    a = first(fes)
    for k in 2:length(fes)
        b = fes[k]
        @argcheck(forecast_summary_same_target(a.target, b.target),
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `target`: $(b.target) against $(a.target)"))
        @argcheck(a.horizon == b.horizon,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `horizon`: $(b.horizon) against $(a.horizon)"))
        @argcheck(a.lag == b.lag,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `lag`: $(b.lag) against $(a.lag)"))
        @argcheck(a.step == b.step,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `step`: $(b.step) against $(a.step)"))
        @argcheck(a.min_count == b.min_count,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `min_count`: $(b.min_count) against $(a.min_count)"))
        @argcheck(a.ppy == b.ppy,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `ppy`: $(b.ppy) against $(a.ppy)"))
        @argcheck(size(a.alpha, 2) == size(b.alpha, 2),
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on the asset axis: $(size(b.alpha, 2)) against $(size(a.alpha, 2))"))
        @argcheck(a.umsk == b.umsk,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `umsk`: the two universes differ $(size(a.umsk) == size(b.umsk) ? "at $(count(a.umsk .!= b.umsk)) cell(s)" : "in shape, $(size(b.umsk)) against $(size(a.umsk))")"))
    end
    return nothing
end
"""
    forecast_summary_assert_comparable(fes::AbstractVector{<:ForecastEvaluationResult})

Refuse a set of evaluations whose rows do not mean the same thing, and name the field that differs.

Two evaluations are comparable when they answer one question, which [`forecast_summary_assert_same_question`](@ref) checks, and when they answer it on one sample. Two rows over different dates compare nothing, so the evaluation `dates` must agree as well. This check refuses a set that answers one question on two grids, and [`forecast_evaluation_align`](@ref) puts such a set on one grid.

# Arguments

  - `fes`: The evaluations to summarise, at least one.

# Validation

  - The rules of [`forecast_summary_assert_same_question`](@ref).
  - Every evaluation agrees with the first on `dates`. Raises a [`ConflictingArgumentError`](@ref) that names the field.

# Returns

  - `nothing`.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`forecast_summary_assert_same_question`](@ref)
  - [`forecast_evaluation_align`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_summary_assert_comparable(fes::AbstractVector{<:ForecastEvaluationResult})
    forecast_summary_assert_same_question(fes)
    a = first(fes)
    for k in 2:length(fes)
        b = fes[k]
        @argcheck(a.dates == b.dates,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `dates`: $(length(b.dates)) date(s) against $(length(a.dates))"))
    end
    return nothing
end
"""
    forecast_evaluation_align(fes::AbstractVector{<:ForecastEvaluationResult}) -> Vector{ForecastEvaluationResult}

Put a set of Return Forecast evaluations on the evaluation grid they share.

Two members of the Return Forecast family rarely share a grid. A member that publishes its history is scorable from the first row of the block, and a member that the library refits along the grid scores only after its warm-up. The two grids start at different observations, and [`forecast_evaluation_summary`](@ref) refuses them on `dates`.

This verb changes the dates of each evaluation and nothing else. The dates of every evaluation are a run of stride `step` on the observation axis of the block, so under one `step` the shared grid starts at the latest first date and stops at or before the earliest last date. The grid starts at that latest first date, so a member whose own grid started earlier on another phase moves onto it. The verb keeps `alpha`, `y` and `umsk` as they are, so no statistic of an aligned Result reads a value that the caller did not hand in. The verb keeps a date of the common grid at which a member scores no asset, as [`forecast_evaluation_dates`](@ref) keeps one, and the statistics of that member are `NaN` there.

# Mathematical definition

```math
\\begin{align}
\\underline{t} &= \\max_{k} t^{(k)}_{1}\\,, \\\\
\\overline{t} &= \\min_{k} t^{(k)}_{J_{k}}\\,, \\\\
\\mathcal{D} &= \\left\\{ \\underline{t} + q s : q \\in \\mathbb{Z}_{\\geq 0},\\ \\underline{t} + q s \\leq \\overline{t} \\right\\}\\,.
\\end{align}
```

Where:

  - ``t^{(k)}_{j}``: The ``j``-th evaluation date of evaluation ``k``.
  - ``J_{k}``: Number of evaluation dates of evaluation ``k``.
  - ``\\underline{t}``: Latest first date over the set.
  - ``\\overline{t}``: Earliest last date over the set.
  - ``\\mathcal{D}``: Common evaluation grid, which every aligned evaluation carries.
  - $(math_dict[:s_eval_stride])

# Algorithm

 1. Refuse a set that does not answer one question, with [`forecast_summary_assert_same_question`](@ref).
 2. Take the latest first date `lo` and the earliest last date `hi` over the set, and refuse a set with `lo > hi`.
 3. Build the common grid `dates` from `lo` to `hi` at stride `step`.
 4. Rebuild each [`ForecastEvaluationResult`](@ref) on `dates`, with its own pair, universe and parameters.

# Arguments

  - `fes`: The evaluations to align, at least one, from [`forecast_evaluation`](@ref).

# Validation

  - The rules of [`forecast_summary_assert_same_question`](@ref).
  - The latest first date is no later than the earliest last date. Raises an [`IsEmptyError`](@ref).

# Returns

  - `afes::Vector{ForecastEvaluationResult}`: One evaluation per input, in the order of the inputs, and every one carries the same `dates`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> late = copy(alpha);
       late[1, :] .= NaN;

julia> fes = [forecast_evaluation(alpha, y), forecast_evaluation(late, y)];

julia> [fe.dates for fe in fes]
2-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [2, 3]

julia> [fe.dates for fe in forecast_evaluation_align(fes)]
2-element Vector{Vector{Int64}}:
 [2, 3]
 [2, 3]
```

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`forecast_evaluation_dates`](@ref)
  - [`forecast_summary_assert_same_question`](@ref)
  - [`forecast_summary_assert_comparable`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_evaluation_align(fes::AbstractVector{<:ForecastEvaluationResult})
    forecast_summary_assert_same_question(fes)
    lo = maximum(first(fe.dates) for fe in fes)
    hi = minimum(last(fe.dates) for fe in fes)
    @argcheck(lo <= hi,
              IsEmptyError("the evaluations share no date: the latest first date is $(lo) and the earliest last date is $(hi)"))
    dates = collect(lo:(first(fes).step):hi)
    return [ForecastEvaluationResult(fe.alpha, fe.y, fe.umsk, dates, fe.target, fe.horizon,
                                     fe.lag, fe.step, fe.min_count, fe.ppy) for fe in fes]
end
"""
    forecast_summary_scored(fe::ForecastEvaluationResult, u::MatNum) -> Vector{<:Real}

Return the number of assets an evaluation scored, one entry per evaluation date.

This count is the numerator of [`forecast_coverage`](@ref). A summary reports it beside the share, because the two answer different questions. A coverage of one half over forty assets gives a cross-section large enough to trust a statistic, and the same share over six assets does not. On a point-in-time panel the count moves with the listings while the share stays at `1`.

# Mathematical definition

```math
\\begin{align}
n_{j} &= \\left| \\left\\{ i \\in \\mathcal{U}_{t_{j}} : \\alpha_{t_{j} i} \\text{ and } y_{t_{j} i} \\text{ are finite} \\right\\} \\right|\\,, \\\\
\\mathcal{U}_{t} &= \\left\\{ i : m_{ti} \\text{ and } 0 < u_{ti} < \\infty \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:n_j_scored])
  - $(math_dict[:U_t_univ])
  - $(math_dict[:alpha_ti_fc])
  - $(math_dict[:y_ti_fwd])
  - $(math_dict[:m_ti_univ])
  - $(math_dict[:u_ti_cs])
  - $(math_dict[:t_j_eval])

# Arguments

  - `fe`: An evaluation, from [`forecast_evaluation`](@ref).
  - `u`: Cross-sectional weight history `observations × assets`, from [`forecast_ic_weights`](@ref).

# Returns

  - `n::Vector{<:Real}`: One entry per evaluation date.

# Related

  - [`forecast_summary_coverage`](@ref)
  - [`forecast_coverage`](@ref)
  - [`forecast_ic_weights`](@ref)
"""
function forecast_summary_scored(fe::ForecastEvaluationResult, u::MatNum)
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    umsk::AbstractMatrix{Bool} = fe.umsk
    dates::AbstractVector{<:Integer} = fe.dates
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    n = Vector{Tf}(undef, length(dates))
    for (j, t) in enumerate(dates)
        s = 0
        for i in axes(alpha, 2)
            s += umsk[t, i] &&
                 0 < u[t, i] < Inf &&
                 isfinite(alpha[t, i]) &&
                 isfinite(y[t, i])
        end
        n[j] = Tf(s)
    end
    return n
end
"""
    forecast_summary_coverage(fe::ForecastEvaluationResult,
                              w::Option{<:MatNum}) -> NamedTuple

Return the four coverage figures of one evaluation in a summary.

The summary reports the share and the count, each at its mean and at its minimum. The mean states the sample that the evaluation ran on. The minimum states the smallest cross-section that any statistic of the evaluation read, and that figure tells whether a coefficient is a measurement or an accident. A date whose universe is empty has no coverage, so it enters neither share figure, but it enters both count figures with a count of zero.

# Mathematical definition

```math
\\begin{align}
\\bar{c} &= \\frac{1}{\\left| \\mathcal{J}_{c} \\right|} \\sum_{j \\in \\mathcal{J}_{c}} c_{j}\\,, \\\\
c_{\\min} &= \\min_{j \\in \\mathcal{J}_{c}} c_{j}\\,, \\\\
\\bar{n} &= \\frac{1}{J} \\sum_{j=1}^{J} n_{j}\\,, \\\\
n_{\\min} &= \\min_{1 \\leq j \\leq J} n_{j}\\,.
\\end{align}
```

Where:

  - ``\\bar{c}``: Mean coverage over the dates that carry a finite coverage.
  - ``c_{\\min}``: Least coverage over the same dates.
  - ``\\bar{n}``: Mean scored count over the evaluation dates.
  - ``n_{\\min}``: Least scored count over the evaluation dates.
  - ``\\mathcal{J}_{c}``: Evaluation dates at which ``c_{j}`` is finite, which are the dates whose universe is not empty.
  - ``J``: Number of evaluation dates.
  - $(math_dict[:c_j_cov])
  - $(math_dict[:n_j_scored])

# Arguments

  - `fe`: An evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.

# Returns

  - `cov::NamedTuple`: `(; mean_coverage, min_coverage, mean_n_scored, min_n_scored)`, which are ``\\bar{c}``, ``c_{\\min}``, ``\\bar{n}`` and ``n_{\\min}``. The two shares are `NaN` when the universe of every date is empty, and the two counts are `NaN` when the evaluation carries no date.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`forecast_summary_scored`](@ref)
  - [`forecast_coverage`](@ref)
"""
function forecast_summary_coverage(fe::ForecastEvaluationResult, w::Option{<:MatNum})
    u = forecast_ic_weights(fe.alpha, w)
    c = forecast_coverage(fe, w)
    n = forecast_summary_scored(fe, u)
    Tf = eltype(n)
    fc = c[isfinite.(c)]
    return (; mean_coverage = isempty(fc) ? Tf(NaN) : Tf(sum(fc) / length(fc)),
            min_coverage = isempty(fc) ? Tf(NaN) : Tf(minimum(fc)),
            mean_n_scored = isempty(n) ? Tf(NaN) : sum(n) / length(n),
            min_n_scored = isempty(n) ? Tf(NaN) : minimum(n))
end
"""
    forecast_summary_row(fe::ForecastEvaluationResult, w::Option{<:MatNum},
                         bins::Integer) -> NamedTuple

Return the thirty figures of one evaluation in a summary.

The row computes no statistic of its own. It reads the ten information-coefficient figures from [`forecast_ic_summary`](@ref), five figures of each of the two books from [`forecast_portfolio`](@ref), six calibration figures from [`forecast_calibration`](@ref), and the four figures of the sample from [`forecast_summary_coverage`](@ref).

Each book reports `ann_return`, `ann_volatility` and `sharpe` from the [`PerformanceSummaryResult`](@ref) it carries, with its hit rate and its mean turnover beside them. None of the three reads the order of the returns. The row leaves out the other figures of that summary, and a caller reads them from `forecast_portfolio(fe; kind = …).summary`. Two of them, `max_drawdown` and `calmar`, read the path, and [`forecast_portfolio`](@ref) summarises the compressed path, which joins the date before a gap to the date after it. When the gaps of two forecasts fall on different dates, their drawdowns are of two different paths.

The threshold is the one that the evaluation carries, and the row takes no keyword for it. [`forecast_portfolio`](@ref) and [`forecast_quantile_spread`](@ref) read `fe.min_count` from the Result, so a keyword that reached only the coefficients gives a row under two thresholds.

# Algorithm

 1. Summarise the coefficients of [`forecast_ic`](@ref) with [`forecast_ic_summary`](@ref), at the lag that [`forecast_ic_lags`](@ref) derives, giving `ic`.
 2. Score the rank book and the z-score book with [`forecast_portfolio`](@ref), giving `rk` and `zs`.
 3. Read the calibration at `bins` bins with [`forecast_calibration`](@ref), giving `cb`.
 4. Read the four coverage figures with [`forecast_summary_coverage`](@ref), giving `cv`.
 5. Collect the thirty figures into one named tuple.

# Arguments

  - `fe`: An evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `bins`: Number of quantile bins the calibration curve cuts.

# Validation

  - The rules of [`forecast_ic`](@ref), [`forecast_portfolio`](@ref), [`forecast_calibration`](@ref) and [`forecast_coverage`](@ref). [`forecast_portfolio`](@ref) refuses a sample in which no date reaches `fe.min_count`, and the row adds no check of its own.

# Returns

  - `row::NamedTuple`: The thirty columns that [`ForecastSummaryResult`](@ref) documents, as scalars.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`ForecastSummaryResult`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_portfolio`](@ref)
  - [`forecast_calibration`](@ref)
  - [`forecast_summary_coverage`](@ref)
"""
function forecast_summary_row(fe::ForecastEvaluationResult, w::Option{<:MatNum},
                              bins::Integer)
    ic = forecast_ic_summary(forecast_ic(fe, w); lags = forecast_ic_lags(fe))
    rk = forecast_portfolio(fe; kind = :rank)
    zs = forecast_portfolio(fe; kind = :zscore)
    cb = forecast_calibration(fe, w; bins = bins)
    cv = forecast_summary_coverage(fe, w)
    return (; spearman_mean_ic = ic.spearman.mean_ic, spearman_std_ic = ic.spearman.std_ic,
            spearman_ic_ir = ic.spearman.ic_ir, spearman_t_stat = ic.spearman.t_stat,
            spearman_hit_rate = ic.spearman.hit_rate, pearson_mean_ic = ic.pearson.mean_ic,
            pearson_std_ic = ic.pearson.std_ic, pearson_ic_ir = ic.pearson.ic_ir,
            pearson_t_stat = ic.pearson.t_stat, pearson_hit_rate = ic.pearson.hit_rate,
            rank_ann_return = rk.summary.ann_return,
            rank_ann_volatility = rk.summary.ann_volatility,
            rank_sharpe = rk.summary.sharpe, rank_hit_rate = rk.hit_rate,
            rank_mean_turnover = rk.mean_turnover,
            zscore_ann_return = zs.summary.ann_return,
            zscore_ann_volatility = zs.summary.ann_volatility,
            zscore_sharpe = zs.summary.sharpe, zscore_hit_rate = zs.hit_rate,
            zscore_mean_turnover = zs.mean_turnover, calibration_slope = cb.slope,
            mean_alpha = cb.mean_alpha, std_alpha = cb.std_alpha, mean_y = cb.mean_y,
            std_y = cb.std_y, n_bins = cb.n_bins, mean_coverage = cv.mean_coverage,
            min_coverage = cv.min_coverage, mean_n_scored = cv.mean_n_scored,
            min_n_scored = cv.min_n_scored)
end
"""
    forecast_summary_spreads(fes::AbstractVector{<:ForecastEvaluationResult},
                             quantiles::Nothing) -> Tuple
    forecast_summary_spreads(fes::AbstractVector{<:ForecastEvaluationResult},
                             quantiles) -> Tuple

Return the quantile-spread block of a summary, or five `nothing`s when the caller asks for no quantile.

Every other column of a summary holds one number per forecast. This block holds one number per forecast and per quantile, so it has a second axis. A caller who asks for no quantile computes none, and the method for `nothing` returns the five entries as `nothing`. [`FactorSummaryResult`](@ref) marks its absent columns in the same way.

# Algorithm

 1. Collect the tail fractions into the vector `q`. A single fraction becomes a vector of one.
 2. For each evaluation, cut the spread at every fraction of `q` with [`forecast_quantile_spread`](@ref), giving `s`.
 3. Write the annualised mean, the annualised volatility, their ratio and the hit rate of `s` into the row of the evaluation in `sm`, `sv`, `si` and `sh`.

# Arguments

  - `fes`: The evaluations to summarise.
  - `quantiles`: Tail fractions, each in `(0, 0.5]`, as a collection or as one number, or `nothing`.

# Validation

  - The rules of [`forecast_quantile_spread`](@ref).

# Returns

  - `block::Tuple`: `(quantiles, spread_ann_return, spread_ann_volatility, spread_sharpe, spread_hit_rate)`. `quantiles` is a vector, and the four matrices are `forecasts × quantiles`. All five entries are `nothing` when `quantiles` is `nothing`.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`ForecastSummaryResult`](@ref)
  - [`forecast_quantile_spread`](@ref)
"""
function forecast_summary_spreads(::AbstractVector{<:ForecastEvaluationResult}, ::Nothing)
    return nothing, nothing, nothing, nothing, nothing
end
function forecast_summary_spreads(fes::AbstractVector{<:ForecastEvaluationResult},
                                  quantiles)
    q = vec(collect(quantiles))
    n = length(fes)
    m = length(q)
    alpha::MatNum = first(fes).alpha
    y::MatNum = first(fes).y
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)))
    sm = Matrix{Tf}(undef, n, m)
    sv = Matrix{Tf}(undef, n, m)
    si = Matrix{Tf}(undef, n, m)
    sh = Matrix{Tf}(undef, n, m)
    # The four matrices are filled in a loop rather than built by a two-dimensional
    # comprehension. A comprehension over two ranges is a `Base.Generator` whose iterator
    # type JET cannot pin when the tail fractions arrive untyped, and reading a field of
    # that generator is what its analysis reports.
    for (i, k) in enumerate(eachindex(fes))
        fe::ForecastEvaluationResult = fes[k]
        s = forecast_quantile_spread(fe; quantiles = q)
        am::VecNum = s.ann_mean
        av::VecNum = s.ann_vol
        ai::VecNum = s.ann_ir
        ah::VecNum = s.hit_rate
        for j in 1:m
            sm[i, j] = am[j]
            sv[i, j] = av[j]
            si[i, j] = ai[j]
            sh[i, j] = ah[j]
        end
    end
    return q, sm, sv, si, sh
end
"""
$(DocStringExtensions.TYPEDEF)

The headline statistics of one or more Return Forecast evaluations, one entry per forecast.

`ForecastSummaryResult` is what [`forecast_evaluation_summary`](@ref) returns. It is the top of the evaluation hierarchy. It carries thirty columns that it reads from the verbs below it, on an axis whose entries are the forecasts, so a caller can tabulate it, plot it, test it, or read it without a plotting package.

# One class, and no comparison class beside it

A single evaluation is the length-1 case of this Result, and a comparison of two is the length-2 case. A comparison only sets the columns of several evaluations side by side, and a columnar Result already does that, so the library has no second class.

# One hit-rate denominator

The five hit rates of this Result count the dates that **scored**, and none counts every evaluation date. `spearman_hit_rate` and `pearson_hit_rate` come from [`exposure_ic_factor_summary`](@ref), and `rank_hit_rate`, `zscore_hit_rate` and the optional `spread_hit_rate` come from [`forecast_hit_rate`](@ref). Both read a date without a figure as a date at which the evaluation measured nothing, so each hit rate uses the same sample as the mean beside it. A date whose cross-section fell under `min_count` has no coefficient. The coverage columns `mean_coverage`, `min_coverage`, `mean_n_scored` and `min_n_scored` report how often that happened, and a hit rate that counted such a date as a miss would count the same gap twice.

# The five columns that can be absent

`quantiles`, `spread_ann_return`, `spread_ann_volatility`, `spread_sharpe` and `spread_hit_rate` have a second axis, the tail fraction, that no other column has. The summary computes them only when a caller asks for a quantile, and they are `nothing` otherwise. A consumer reads their absence by dispatch.

# What a caller reads one call down

Three parts of an evaluation are not columns here.

  - The other figures of the performance summary of each book. `forecast_portfolio(fe; kind = …).summary` carries them, and two of them, `max_drawdown` and `calmar`, read the compressed path of the book.
  - The holding-period and decay tables, whose axis is the forward window. [`forecast_holding_period`](@ref) and [`forecast_decay`](@ref) return them.
  - The factor correlations, whose axis is the factor. [`forecast_factor_correlation`](@ref) returns them, and [`exposure_ic_summary`](@ref) summarises them per factor.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastSummaryResult(
        names,
        spearman_mean_ic, spearman_std_ic, spearman_ic_ir, spearman_t_stat,
        spearman_hit_rate, pearson_mean_ic, pearson_std_ic, pearson_ic_ir,
        pearson_t_stat, pearson_hit_rate,
        rank_ann_return, rank_ann_volatility, rank_sharpe, rank_hit_rate,
        rank_mean_turnover, zscore_ann_return, zscore_ann_volatility, zscore_sharpe,
        zscore_hit_rate, zscore_mean_turnover,
        calibration_slope, mean_alpha, std_alpha, mean_y, std_y, n_bins,
        mean_coverage, min_coverage, mean_n_scored, min_n_scored,
        quantiles, spread_ann_return, spread_ann_volatility, spread_sharpe,
        spread_hit_rate, ppy
    ) -> ForecastSummaryResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a
Result, so [`forecast_evaluation_summary`](@ref) builds it and a caller reads it. It has no
keyword constructor, and it validates nothing of its own.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`FactorSummaryResult`](@ref)
"""
@concrete struct ForecastSummaryResult <: AbstractResult
    """
    Name of each forecast, one entry per forecast.
    """
    names
    """
    Mean Spearman information coefficient over the evaluation dates, one entry per forecast.
    """
    spearman_mean_ic
    """
    Standard deviation of the Spearman information coefficient, one entry per forecast.
    """
    spearman_std_ic
    """
    Ratio of the mean Spearman information coefficient to its standard deviation, one entry per forecast.
    """
    spearman_ic_ir
    """
    Date-level t-statistic of the mean Spearman information coefficient, one entry per forecast. Its standard error reads the overlap of the forward windows, so it is the familiar ratio only under a stride of the horizon.
    """
    spearman_t_stat
    """
    Share of the **scored** dates at which the Spearman information coefficient is positive, one entry per forecast.
    """
    spearman_hit_rate
    """
    Mean weighted Pearson information coefficient over the evaluation dates, one entry per forecast.
    """
    pearson_mean_ic
    """
    Standard deviation of the weighted Pearson information coefficient, one entry per forecast.
    """
    pearson_std_ic
    """
    Ratio of the mean weighted Pearson information coefficient to its standard deviation, one entry per forecast.
    """
    pearson_ic_ir
    """
    Date-level t-statistic of the mean weighted Pearson information coefficient, one entry per forecast. Its standard error reads the overlap of the forward windows, so it is the familiar ratio only under a stride of the horizon.
    """
    pearson_t_stat
    """
    Share of the **scored** dates at which the weighted Pearson information coefficient is positive, one entry per forecast.
    """
    pearson_hit_rate
    """
    Annualised mean return of the rank-weighted book, one entry per forecast.
    """
    rank_ann_return
    """
    Annualised volatility of the rank-weighted book, one entry per forecast.
    """
    rank_ann_volatility
    """
    Ratio of the annualised mean return of the rank-weighted book to its annualised volatility, one entry per forecast.
    """
    rank_sharpe
    """
    Share of the **scored** dates at which the rank-weighted book returned a profit, one entry per forecast.
    """
    rank_hit_rate
    """
    Mean turnover of the rank-weighted book over the dates it traded, one entry per forecast.
    """
    rank_mean_turnover
    """
    Annualised mean return of the z-score-weighted book, one entry per forecast.
    """
    zscore_ann_return
    """
    Annualised volatility of the z-score-weighted book, one entry per forecast.
    """
    zscore_ann_volatility
    """
    Ratio of the annualised mean return of the z-score-weighted book to its annualised volatility, one entry per forecast.
    """
    zscore_sharpe
    """
    Share of the **scored** dates at which the z-score-weighted book returned a profit, one entry per forecast.
    """
    zscore_hit_rate
    """
    Mean turnover of the z-score-weighted book over the dates it traded, one entry per forecast.
    """
    zscore_mean_turnover
    """
    Scale multiplier that maps the forecast onto the realised target, pooled over every scorable pair, one entry per forecast.
    """
    calibration_slope
    """
    Mean of the pooled forecasts the calibration reads, one entry per forecast.
    """
    mean_alpha
    """
    Standard deviation of the pooled forecasts the calibration reads, one entry per forecast.
    """
    std_alpha
    """
    Mean of the pooled targets the calibration reads, one entry per forecast.
    """
    mean_y
    """
    Standard deviation of the pooled targets the calibration reads, one entry per forecast.
    """
    std_y
    """
    Number of non-empty bins the calibration curve was cut into, one entry per forecast.
    """
    n_bins
    """
    Mean share of the universe the evaluation scored, over the dates whose universe is not empty, one entry per forecast. The universe is the estimation mask the evaluation carries in `umsk`, so a late lister lowers the count and not the share.
    """
    mean_coverage
    """
    Least share of the universe the evaluation scored at any date whose universe is not empty, one entry per forecast.
    """
    min_coverage
    """
    Mean number of assets the evaluation scored per evaluation date, one entry per forecast.
    """
    mean_n_scored
    """
    Least number of assets the evaluation scored at any of its dates, one entry per forecast.
    """
    min_n_scored
    """
    Tail fractions the quantile spreads were cut at, or `nothing` when the caller asked for none.
    """
    quantiles
    """
    Annualised mean of the top-minus-bottom spread, `forecasts × quantiles`, or `nothing` when the caller asked for no quantile.
    """
    spread_ann_return
    """
    Annualised volatility of the top-minus-bottom spread, `forecasts × quantiles`, or `nothing` when the caller asked for no quantile.
    """
    spread_ann_volatility
    """
    Ratio of the annualised mean of the top-minus-bottom spread to its annualised volatility, `forecasts × quantiles`, or `nothing` when the caller asked for no quantile.
    """
    spread_sharpe
    """
    Share of the **scored** dates at which the top-minus-bottom spread is positive, `forecasts × quantiles`, or `nothing` when the caller asked for no quantile.
    """
    spread_hit_rate
    """
    $(field_dict[:ps_ppy]) It is the `ppy` that every evaluation of the summary carries.
    """
    ppy
end
"""
    forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                w::Option{<:MatNum} = nothing; align::Bool = false,
                                names = nothing, bins::Integer = 10,
                                quantiles = nothing) -> ForecastSummaryResult
    forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                csfm::CrossSectionalFactorModel;
                                weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                kwargs...) -> ForecastSummaryResult
    forecast_evaluation_summary(fe::ForecastEvaluationResult,
                                w::Option{<:MatNum} = nothing;
                                kwargs...) -> ForecastSummaryResult
    forecast_evaluation_summary(fe::ForecastEvaluationResult,
                                csfm::CrossSectionalFactorModel;
                                kwargs...) -> ForecastSummaryResult

Summarise one or more Return Forecast evaluations as a [`ForecastSummaryResult`](@ref).

This verb is the top of the evaluation hierarchy, and it is also the comparison. It calls one verb per block, computes no statistic of its own, and collects each block onto the forecast axis, as [`factor_model_summary`](@ref) does. A single evaluation is the length-1 case, with the same type and the same columns as a comparison of four. A caller who starts with one member and later wants a table of four changes the argument and nothing else.

The vector method **refuses evaluations that are not comparable**, and names the field that differs, because a table invites a comparison that its rows must support. Two members that answer one question on two grids are the ordinary case, because a refit member warms up and a member that publishes its history does not. The verb refuses them on `dates` unless `align = true`, which first puts them on their shared grid through [`forecast_evaluation_align`](@ref).

A weighting reaches the summary in one of two forms, as it reaches every other block-aware statistic of the evaluation. The caller passes a bare weight history in the second position, or the cross-sectional factor model, whose [`AbstractOrthogonalityMetric`](@ref) resolves one. The Result carries no block, so the verb cannot read the metric from it.

# Algorithm

 1. Under `align = true`, put the evaluations on the grid they share with [`forecast_evaluation_align`](@ref).
 2. Refuse a set of evaluations that are not comparable, with [`forecast_summary_assert_comparable`](@ref).
 3. Resolve the names axis with [`forecast_summary_names`](@ref).
 4. Read the thirty core figures of each evaluation with [`forecast_summary_row`](@ref).
 5. Read the quantile-spread block with [`forecast_summary_spreads`](@ref), which returns `nothing` when the caller asks for no quantile.
 6. Collect the rows into columns and build a [`ForecastSummaryResult`](@ref), with the `ppy` that the evaluations agree on.

# Arguments

  - `fes`: The evaluations to summarise, at least one, from [`forecast_evaluation`](@ref).
  - `fe`: One evaluation. The verb summarises it as the length-1 case.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecasts, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `align`: Whether to put the evaluations on the grid they share before the summary reads them. Under `false` the verb refuses a set whose `dates` differ.
  - `names`: One name per evaluation, or `nothing` to number them.
  - `bins`: Number of quantile bins the calibration curve cuts.
  - `quantiles`: Tail fractions the quantile spreads are cut at, each in `(0, 0.5]`, as a collection or as one number, or `nothing` for none.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose weight history the weighted statistics read.

# Validation

  - The rules of [`forecast_evaluation_align`](@ref) under `align = true`.
  - The rules of [`forecast_summary_assert_comparable`](@ref), [`forecast_summary_names`](@ref), [`forecast_summary_row`](@ref) and [`forecast_summary_spreads`](@ref).

# Returns

  - `fs::ForecastSummaryResult`: The computed summary, one entry per forecast.

# Related

  - [`ForecastSummaryResult`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`forecast_evaluation_align`](@ref)
  - [`forecast_summary_row`](@ref)
  - [`forecast_summary_spreads`](@ref)
  - [`forecast_summary_assert_comparable`](@ref)
  - [`factor_model_summary`](@ref)
"""
function forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                     w::Option{<:MatNum} = nothing; align::Bool = false,
                                     names = nothing, bins::Integer = 10,
                                     quantiles = nothing)::ForecastSummaryResult
    afes = align ? forecast_evaluation_align(fes) : fes
    forecast_summary_assert_comparable(afes)
    nm = forecast_summary_names(names, length(afes))
    # The element is asserted rather than taken from the iterator: `afes` is an
    # `AbstractVector` of an abstract element type, so without the assertion every read
    # inside the row is analysed against an unknown carrier.
    rows = [forecast_summary_row(afes[i]::ForecastEvaluationResult, w, bins)
            for i in eachindex(afes)]
    q, sm, sv, si, sh = forecast_summary_spreads(afes, quantiles)
    return ForecastSummaryResult(nm, [r.spearman_mean_ic for r in rows],
                                 [r.spearman_std_ic for r in rows],
                                 [r.spearman_ic_ir for r in rows],
                                 [r.spearman_t_stat for r in rows],
                                 [r.spearman_hit_rate for r in rows],
                                 [r.pearson_mean_ic for r in rows],
                                 [r.pearson_std_ic for r in rows],
                                 [r.pearson_ic_ir for r in rows],
                                 [r.pearson_t_stat for r in rows],
                                 [r.pearson_hit_rate for r in rows],
                                 [r.rank_ann_return for r in rows],
                                 [r.rank_ann_volatility for r in rows],
                                 [r.rank_sharpe for r in rows],
                                 [r.rank_hit_rate for r in rows],
                                 [r.rank_mean_turnover for r in rows],
                                 [r.zscore_ann_return for r in rows],
                                 [r.zscore_ann_volatility for r in rows],
                                 [r.zscore_sharpe for r in rows],
                                 [r.zscore_hit_rate for r in rows],
                                 [r.zscore_mean_turnover for r in rows],
                                 [r.calibration_slope for r in rows],
                                 [r.mean_alpha for r in rows], [r.std_alpha for r in rows],
                                 [r.mean_y for r in rows], [r.std_y for r in rows],
                                 [r.n_bins for r in rows], [r.mean_coverage for r in rows],
                                 [r.min_coverage for r in rows],
                                 [r.mean_n_scored for r in rows],
                                 [r.min_n_scored for r in rows], q, sm, sv, si, sh,
                                 first(afes).ppy)
end
function forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                     csfm::CrossSectionalFactorModel;
                                     weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                     kwargs...)::ForecastSummaryResult
    return forecast_evaluation_summary(fes, cs_diagnostic_weights(weighting, csfm);
                                       kwargs...)
end
function forecast_evaluation_summary(fe::ForecastEvaluationResult,
                                     w::Option{<:MatNum} = nothing;
                                     kwargs...)::ForecastSummaryResult
    return forecast_evaluation_summary([fe], w; kwargs...)
end
function forecast_evaluation_summary(fe::ForecastEvaluationResult,
                                     csfm::CrossSectionalFactorModel;
                                     kwargs...)::ForecastSummaryResult
    return forecast_evaluation_summary([fe], csfm; kwargs...)
end

export ForecastSummaryResult, forecast_evaluation_align, forecast_evaluation_summary
