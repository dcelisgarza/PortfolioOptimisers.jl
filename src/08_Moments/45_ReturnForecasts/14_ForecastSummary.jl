"""
    forecast_summary_names(names::Nothing, n::Integer) -> Vector{String}
    forecast_summary_names(names::AbstractVector{<:AbstractString},
                           n::Integer) -> Vector{String}

Return the names axis of a forecast summary, one name per evaluation.

A summary is read as a table, and a row of it is worth little without a name: a caller who is told the second row scored better than the first still has to remember which member produced it. The names are supplied by the caller, because an evaluation carries the pairing and its parameters and no member ever carried a name of its own. The absent case is dispatched rather than branched, and it numbers the rows in the order they were handed over.

# Arguments

  - `names`: One name per evaluation, or `nothing` to number them.
  - `n`: Number of evaluations the summary carries.

# Validation

  - `length(names) == n`. Raises a `DimensionMismatch`.

# Returns

  - `names::Vector{String}`: One name per evaluation, in the order the evaluations were given.

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

Return whether two evaluations were scored against the same forward target.

Two members are comparable only where they answer for the same quantity, and the quantity is the [`AbstractForecastTarget`](@ref) the pairing was taken over. The test is the member and its fields, not the member alone: [`PanelFieldTarget`](@ref) names a field of the panel, so two of them are the same target where they name the same field and different targets where they do not.

The fields are walked in declaration order rather than compared by an equality this library does not define, which is what [`assert_mergeable_states`](@ref) does for the same reason.

# Arguments

  - `a`: The forward target of the first evaluation.
  - `b`: The forward target of the second evaluation.

# Returns

  - `same::Bool`: `true` where the two are the same member carrying the same fields.

# Related

  - [`forecast_summary_assert_comparable`](@ref)
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
    forecast_summary_assert_comparable(fes::AbstractVector{<:ForecastEvaluationResult})

Refuse a set of evaluations whose rows would not mean the same thing, naming the field that differs.

A summary puts one row per forecast beside the others and invites the reader to compare them, so the rows must be answers to one question. Seven of the fields a [`ForecastEvaluationResult`](@ref) carries change what a row means: the forward target changes the quantity scored, `horizon` and `lag` change the window it is scored over, `step` and `dates` change the sample, `min_count` changes which cross-sections entered it, and `ppy` changes the units the annualised columns are reported in. A difference in any of them is refused here rather than reported as a difference in skill.

The asset axis is checked with them, because two forecasts over different universes are two different questions however their parameters agree. `alpha` and `y` are not compared: a summary of two members of one panel is exactly the case where the forecasts differ, and that is what the summary is for.

The reference implementation checks none of this.

# Arguments

  - `fes`: The evaluations to summarise, at least one.

# Validation

  - `!isempty(fes)`. Raises an [`IsEmptyError`](@ref).
  - Every evaluation agrees with the first on `target`, `horizon`, `lag`, `step`, `min_count`, `ppy`, `dates` and the number of assets. Raises a [`ConflictingArgumentError`](@ref) naming the field.

# Returns

  - `nothing`.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`forecast_summary_same_target`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_summary_assert_comparable(fes::AbstractVector{<:ForecastEvaluationResult})
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
        @argcheck(a.dates == b.dates,
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on `dates`: $(length(b.dates)) date(s) against $(length(a.dates))"))
        @argcheck(size(a.alpha, 2) == size(b.alpha, 2),
                  ConflictingArgumentError("evaluation $(k) differs from evaluation 1 on the asset axis: $(size(b.alpha, 2)) against $(size(a.alpha, 2))"))
    end
    return nothing
end
"""
    forecast_summary_scored(fe::ForecastEvaluationResult, u::MatNum) -> Vector{<:Real}

Return the number of assets an evaluation scored, one entry per evaluation date.

This is the numerator [`forecast_coverage`](@ref) divides by the universe, and a summary reports it beside the share because the two say different things: a coverage of one half over forty assets is a cross-section a statistic can be believed on, and the same share over six assets is not.

An asset is counted where it carries a positive weight, a finite forecast and a finite target, which is the rule the coverage applies.

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
    dates::AbstractVector{<:Integer} = fe.dates
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    n = Vector{Tf}(undef, length(dates))
    for (j, t) in enumerate(dates)
        s = 0
        for i in axes(alpha, 2)
            s += u[t, i] > 0 && isfinite(alpha[t, i]) && isfinite(y[t, i])
        end
        n[j] = Tf(s)
    end
    return n
end
"""
    forecast_summary_coverage(fe::ForecastEvaluationResult,
                              w::Option{<:MatNum}) -> NamedTuple

Return the four coverage figures a summary carries for one evaluation.

The share and the count are each reported at their mean and at their minimum. The mean says what the evaluation ran on, and the minimum says the worst cross-section any statistic of it was taken over, which is the figure that says whether a coefficient is a measurement or an accident.

# Arguments

  - `fe`: An evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.

# Returns

  - `cov::NamedTuple`: `(; mean_coverage, min_coverage, mean_n_scored, min_n_scored)`. Every figure is `NaN` where the evaluation scored no date.

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

Return the thirty figures one evaluation contributes to a summary.

The row computes no statistic of its own. It reads [`forecast_ic_summary`](@ref) for the ten information-coefficient figures, [`forecast_portfolio`](@ref) for each of the two books, [`forecast_calibration`](@ref) for the scale of the forecast, and [`forecast_summary_coverage`](@ref) for the sample every one of them was taken over.

The two books report the first three figures of the [`PerformanceSummaryResult`](@ref) they carry, and their hit rate and mean turnover beside them. The drawdown family — `sortino`, `calmar`, `max_drawdown`, `cvar` and `sharpe_stderr` — is deliberately left out and read from `forecast_portfolio(fe; kind = …).summary` instead: [`forecast_portfolio`](@ref) states that its summary is of the **compressed** path, so those five figures join the date before a gap to the date after it, and two forecasts whose gaps fall on different dates would be set beside each other over two different paths. The three that are reported are two means and a ratio of them, which the compression does not move.

The threshold is the evaluation's own. It is not a keyword here, because [`forecast_portfolio`](@ref) and [`forecast_quantile_spread`](@ref) read `fe.min_count` off the Result, and one that reached only the coefficients would print a row taken under two thresholds.

# Arguments

  - `fe`: An evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `bins`: Number of quantile bins the calibration curve cuts.

# Validation

  - The rules of [`forecast_ic`](@ref), [`forecast_portfolio`](@ref), [`forecast_calibration`](@ref) and [`forecast_coverage`](@ref). A sample no date can score is refused by [`forecast_portfolio`](@ref) rather than by a guard this verb adds.

# Returns

  - `row::NamedTuple`: The thirty columns [`ForecastSummaryResult`](@ref) documents, as scalars.

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
    ic = forecast_ic_summary(forecast_ic(fe, w))
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

Return the quantile-spread block of a summary, or five `nothing`s where none was asked for.

Every other column of a summary is one number per forecast, and this block is one number per forecast **and quantile**, so it carries a second axis the rest of the Result does not. A caller who asks for no quantile pays for none and reads the five entries back as `nothing` by dispatch, which is the idiom [`FactorSummaryResult`](@ref) already holds for its absent columns.

# Arguments

  - `fes`: The evaluations to summarise.
  - `quantiles`: Tail fractions, each in `(0, 0.5]`, or `nothing`.

# Validation

  - The rules of [`forecast_quantile_spread`](@ref).

# Returns

  - `block::Tuple`: `(quantiles, spread_ann_return, spread_ann_volatility, spread_sharpe, spread_hit_rate)`. The four matrices are `forecasts × quantiles`, or all five entries are `nothing`.

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
    q = collect(quantiles)
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
        s = forecast_quantile_spread(fe; quantiles = quantiles)
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

`ForecastSummaryResult` is what [`forecast_evaluation_summary`](@ref) returns. It is the top of the evaluation hierarchy: it carries thirty columns read off the level-2 verbs, on an axis whose entries are the forecasts that were evaluated, so a summary can be tabulated, plotted, asserted on in a test, or read without a plotting package installed.

# One class, and no comparison class beside it

A single evaluation is the **length-1** case of this Result and a comparison is the length-2 case, so no second class ships. The reference implementation exports two, an evaluation and a comparison, and the comparison exists only to set the first one's columns side by side — which a columnar Result already is.

# The two hit-rate denominators

The five hit rates this Result carries are **not** taken against the same denominator, and the difference is deliberate rather than an oversight.

  - `spearman_hit_rate` and `pearson_hit_rate` come from [`exposure_ic_factor_summary`](@ref), which counts a positive coefficient against **every** evaluation date. A date whose cross-section fell under `min_count` carries no coefficient, and that is a date the forecast failed to rank, so it counts as a miss.
  - `rank_hit_rate`, `zscore_hit_rate` and the optional `spread_hit_rate` come from [`forecast_hit_rate`](@ref), which counts against the dates that **scored**. A date with no portfolio return traded nothing, so it is not a loss and must not be read as one, and the denominator is then the one the annualised mean beside it was taken over.

The two conventions live in different verbs, one of which is shared with the cross-sectional exposure diagnostics, so this Result names its columns apart and states the denominator of each rather than moving either.

# The five columns that can be absent

`quantiles`, `spread_ann_return`, `spread_ann_volatility`, `spread_sharpe` and `spread_hit_rate` carry a second axis — the tail fraction — that no other column has. They are computed only where a caller asks for a quantile, and read back as `nothing` otherwise. A consumer reads that by dispatch rather than by a test of its own.

# What is read one call down instead

Three parts of an evaluation are not columns here, because their axis is not the forecast.

  - The **drawdown family** of each book, which is of the compressed path; `forecast_portfolio(fe; kind = …).summary` carries it.
  - The **holding-period and decay tables**, whose axis is the forward window; [`forecast_holding_period`](@ref) and [`forecast_decay`](@ref) answer them.
  - The **factor correlations**, whose axis is the factor; [`forecast_factor_correlation`](@ref) answers them and [`exposure_ic_summary`](@ref) summarises them per factor.

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
Result, so [`forecast_evaluation_summary`](@ref) builds it and a caller reads it; there is no
keyword constructor, and the type validates nothing of its own.

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
    Date-level t-statistic of the mean Spearman information coefficient, one entry per forecast.
    """
    spearman_t_stat
    """
    Share of **every** evaluation date at which the Spearman information coefficient is positive, one entry per forecast.
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
    Date-level t-statistic of the mean weighted Pearson information coefficient, one entry per forecast.
    """
    pearson_t_stat
    """
    Share of **every** evaluation date at which the weighted Pearson information coefficient is positive, one entry per forecast.
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
    Mean share of the universe the evaluation scored, one entry per forecast.
    """
    mean_coverage
    """
    Least share of the universe the evaluation scored at any of its dates, one entry per forecast.
    """
    min_coverage
    """
    Mean number of assets the evaluation scored, one entry per forecast.
    """
    mean_n_scored
    """
    Least number of assets the evaluation scored at any of its dates, one entry per forecast.
    """
    min_n_scored
    """
    Tail fractions the quantile spreads were cut at, or `nothing` when none was asked for.
    """
    quantiles
    """
    Annualised mean of the top-minus-bottom spread, `forecasts × quantiles`, or `nothing` when no quantile was asked for.
    """
    spread_ann_return
    """
    Annualised volatility of the top-minus-bottom spread, `forecasts × quantiles`, or `nothing` when no quantile was asked for.
    """
    spread_ann_volatility
    """
    Ratio of the annualised mean of the top-minus-bottom spread to its annualised volatility, `forecasts × quantiles`, or `nothing` when no quantile was asked for.
    """
    spread_sharpe
    """
    Share of the **scored** dates at which the top-minus-bottom spread is positive, `forecasts × quantiles`, or `nothing` when no quantile was asked for.
    """
    spread_hit_rate
    """
    $(field_dict[:ps_ppy]) It defaults to `1`, which reports the statistics per period.
    """
    ppy
end
"""
    forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                w::Option{<:MatNum} = nothing; names = nothing,
                                bins::Integer = 10,
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

This is the top of the evaluation hierarchy, and it is also the comparison. It calls one level-2 verb per block and computes no statistic of its own beyond collecting each block onto the forecast axis, which is how [`factor_model_summary`](@ref) is built. A single evaluation is the length-1 case and has the same type and the same columns as a comparison of four, so a caller who starts with one member and later wants a table of four changes the argument and nothing else.

The vector method **refuses evaluations that are not comparable**, naming the field that differs, because a table invites a comparison its rows would not support. The reference implementation checks none of this.

A weighting reaches the summary the way it reaches every other block-aware statistic of this map: as a bare weight history positionally, or as the cross-sectional factor model whose [`AbstractOrthogonalityMetric`](@ref) resolves one. The Result carries no block, so the metric cannot be read off it.

# Algorithm

 1. Refuse a set of evaluations that are not comparable, with [`forecast_summary_assert_comparable`](@ref).
 2. Resolve the names axis with [`forecast_summary_names`](@ref).
 3. Read the thirty core figures of each evaluation with [`forecast_summary_row`](@ref).
 4. Read the quantile-spread block with [`forecast_summary_spreads`](@ref), which answers `nothing` where no quantile was asked for.
 5. Collect the rows into columns and build a [`ForecastSummaryResult`](@ref), carrying the `ppy` the evaluations agree on.

# Arguments

  - `fes`: The evaluations to summarise, at least one, from [`forecast_evaluation`](@ref).
  - `fe`: One evaluation. It is summarised as the length-1 case.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecasts, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `names`: One name per evaluation, or `nothing` to number them.
  - `bins`: Number of quantile bins the calibration curve cuts.
  - `quantiles`: Tail fractions the quantile spreads are cut at, each in `(0, 0.5]`, or `nothing` for none.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose weight history the weighted statistics read.

# Validation

  - The rules of [`forecast_summary_assert_comparable`](@ref), [`forecast_summary_names`](@ref), [`forecast_summary_row`](@ref) and [`forecast_summary_spreads`](@ref).

# Returns

  - `fs::ForecastSummaryResult`: The computed summary, one entry per forecast.

# Related

  - [`ForecastSummaryResult`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`forecast_summary_row`](@ref)
  - [`forecast_summary_spreads`](@ref)
  - [`forecast_summary_assert_comparable`](@ref)
  - [`factor_model_summary`](@ref)
"""
function forecast_evaluation_summary(fes::AbstractVector{<:ForecastEvaluationResult},
                                     w::Option{<:MatNum} = nothing; names = nothing,
                                     bins::Integer = 10,
                                     quantiles = nothing)::ForecastSummaryResult
    forecast_summary_assert_comparable(fes)
    nm = forecast_summary_names(names, length(fes))
    # The element is asserted rather than taken from the iterator: `fes` is an
    # `AbstractVector` of an abstract element type, so without the assertion every read
    # inside the row is analysed against an unknown carrier.
    rows = [forecast_summary_row(fes[i]::ForecastEvaluationResult, w, bins)
            for i in eachindex(fes)]
    q, sm, sv, si, sh = forecast_summary_spreads(fes, quantiles)
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
                                 first(fes).ppy)
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

export ForecastSummaryResult, forecast_evaluation_summary
