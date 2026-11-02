"""
$(DocStringExtensions.TYPEDEF)

The headline performance statistics of one or more realised return series.

[`performance_summary`](@ref) returns a `PerformanceSummaryResult`. It holds the eleven numbers that [`plot_performance_summary`](@ref) draws, the standard error of the Sharpe ratio, the effective number of assets, the extra risk measures of the caller, and the inputs of the statistics. A caller can put a summary in a table with `DataFrames.DataFrame(ps)`, compare it across runs, test it, or give it to a search, with no plotting package installed.

# Entries and shape

Each **entry** is one series. Three keys name it: `names` is the candidate, `paths` is the path of a population, and `members` is the member of a prediction whose optimiser gives many portfolios. A key that does not apply is `nothing`. Each entry also carries its own `n_periods`, `date0` and `date1`, so the entries can cover different rows. No figure pairs two series.

The shape follows the **type** of the input of [`performance_summary`](@ref), and never the count at run time. An input that holds exactly one series gives one scalar in each field of an entry. Every other input gives one vector entry for each series, in the order candidate, then path, then member. `ppy`, `alpha`, `compound`, `rf` and `measures` are one value for the whole summary.

The excess statistics, the turnover and the effective number of assets read inputs that a bare return series does not hold. Each is `NaN` when its input is absent. The three excess statistics need a `benchmark` series. A prediction result reads that series from its own rows when the returns data held a `B`. The turnover needs the weight path of a multi-period prediction result. The effective number of assets needs weights.

# Sign convention

`max_drawdown` and `cvar` are in **return space**, so both are not positive for a portfolio that has a loss. The [`ConditionalValueatRisk`](@ref) risk measure has the opposite sign, because it gives a loss as a positive number. Thus `ConditionalValueatRisk(; alpha = alpha)(ret) == -summary.cvar`. The values in `risk` keep the sign of their measure, and [`bigger_is_better`](@ref) gives the orientation of each.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PerformanceSummaryResult(
        names, paths, members, date0, date1, n_periods, ppy, alpha, compound, rf,
        ann_return, ann_volatility, sharpe, sharpe_stderr,
        sortino, calmar, max_drawdown, cvar,
        excess_ret, tracking_error, information_ratio, turnover, n_eff, measures, risk
    ) -> PerformanceSummaryResult

The arguments are the fields of the struct, in the order of the declaration. The type is a
Result, so [`performance_summary`](@ref) builds it and a caller reads it. The type has no
keyword constructor, and it does no validation of its own.

Breaking: the struct gained the fields `names`, `paths`, `members`, `date0`, `date1`, `rf`,
`n_eff`, `measures` and `risk`, so a positional call of the old sixteen arguments raises a
`MethodError`.

# Related

  - [`performance_summary`](@ref)
  - [`DataFrames.DataFrame(ps::PerformanceSummaryResult)`](@ref)
  - [`plot_performance_summary`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ForecastSummaryResult`](@ref)
"""
@concrete struct PerformanceSummaryResult <: AbstractResult
    """
    `names`: Name of the candidate of each entry, from the `names` keyword of a vector method. It is `nothing` for an input that holds one candidate.
    """
    names
    """
    `paths`: The `id` of the path of each entry in a population, or `nothing` outside a population. A single-fold prediction result in a population has no `id`, so its position in the population names it.
    """
    paths
    """
    `members`: Index of the member of each entry, when the optimiser of the input gives many portfolios, or `nothing`.
    """
    members
    """
    `date0`: First timestamp of the series of each entry, or `nothing` when the input carries no timestamps.
    """
    date0
    """
    `date1`: Last timestamp of the series of each entry, or `nothing` when the input carries no timestamps.
    """
    date1
    """
    $(field_dict[:ps_n_periods])
    """
    n_periods
    """
    $(field_dict[:ps_ppy])
    """
    ppy
    """
    $(field_dict[:ps_alpha])
    """
    alpha
    """
    $(field_dict[:ps_compound])
    """
    compound
    """
    `rf`: Risk-free rate for each period, at the frequency of the returns. Only `sharpe`, `sharpe_stderr` and `sortino` read it.
    """
    rf
    """
    $(field_dict[:ps_ann_return])
    """
    ann_return
    """
    $(field_dict[:ps_ann_volatility])
    """
    ann_volatility
    """
    $(field_dict[:ps_sharpe])
    """
    sharpe
    """
    $(field_dict[:ps_sharpe_stderr])
    """
    sharpe_stderr
    """
    $(field_dict[:ps_sortino])
    """
    sortino
    """
    $(field_dict[:ps_calmar])
    """
    calmar
    """
    $(field_dict[:ps_max_drawdown])
    """
    max_drawdown
    """
    $(field_dict[:ps_cvar])
    """
    cvar
    """
    `excess_ret`: Annualised arithmetic mean of `ret - benchmark`. It is `NaN` when no benchmark is given.
    """
    excess_ret
    """
    `tracking_error`: Annualised sample standard deviation of `ret - benchmark`. It is `NaN` when no benchmark is given.
    """
    tracking_error
    """
    `information_ratio`: `excess_ret / tracking_error`. It is `NaN` when no benchmark is given, or when the tracking error is zero or `NaN`.
    """
    information_ratio
    """
    `turnover`: Mean turnover for each rebalance of the held path of a multi-period prediction result. One rebalance trades `sum(abs, w_next - w_held)`, from the weights that a fold held after its last observation to the target of the next fold. It is `NaN` for a bare series, for a single fold and for a multi-period result of one fold, because none of them records a rebalance.
    """
    turnover
    """
    `n_eff`: Effective number of assets, [`number_effective_assets`](@ref) of the weights of the entry. A path takes the mean over its folds of the value of each fold target, and a weight path the mean over its rows. It is `NaN` for a bare series, which holds no weights.
    """
    n_eff
    """
    `measures`: The extra risk measures of the caller, a `NamedTuple` by name. It is empty when the caller gives none.
    """
    measures
    """
    `risk`: The value of each extra risk measure on the net series of each entry, a `NamedTuple` with the names of `measures`. Each value is `expected_risk_from_returns(r, ret)`, with the sign of its measure.
    """
    risk
end
"""
    performance_summary(ret::VecNum; ppy::Number = 1, alpha::Number = 0.05,
                        compound::Bool = false, rf::Number = 0, measures = (),
                        benchmark::Option{<:VecNum} = nothing) -> PerformanceSummaryResult
    performance_summary(w::Union{ArrNum, VecVecNum}, X::MatNum,
                        fees::Option{<:Fees} = nothing; kwargs...) -> PerformanceSummaryResult
    performance_summary(w::Union{ArrNum, VecVecNum}, rd::ReturnsResult,
                        fees::Option{<:Fees} = nothing; kwargs...) -> PerformanceSummaryResult
    performance_summary(res::OptimisationResult, rd::ReturnsResult;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(pred::PredRes_MultiPredRes; kwargs...) -> PerformanceSummaryResult
    performance_summary(ppred::PopulationPredictionResult;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(rets::AbstractVector{<:VecNum}; names = nothing,
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(ress::AbstractVector{<:OptimisationResult}, rd::ReturnsResult;
                        names = nothing, kwargs...) -> PerformanceSummaryResult
    performance_summary(preds::AbstractVector{<:AbstractPredictionResult};
                        names = nothing, kwargs...) -> PerformanceSummaryResult

Summarise one or more realised return series as a [`PerformanceSummaryResult`](@ref).

This verb is the one summary of a realised series. It summarises one series, the paths of a population, the members of a prediction whose optimiser gives many portfolios, and a set of candidates of any of these kinds. Each series is one **entry** of the Result, and three keys name it: the candidate, the path and the member.

# The shape follows the type of the input

An input that holds exactly one series gives scalar fields. These inputs are a bare series, weights with returns, one [`OptimisationResult`](@ref) with a [`ReturnsResult`](@ref), and a [`PredictionResult`](@ref) or a [`MultiPeriodPredictionResult`](@ref) that holds one series. Every other input gives one vector entry for each series:

  - Weights in a `VecVecNum`, and an [`OptimisationResult`](@ref) that holds one weight vector for each member, give one entry for each member.
  - A prediction that holds one series for each member gives one entry for each member. Each member reads its own series, its own benchmark series, its own turnover and its own weights.
  - A [`PopulationPredictionResult`](@ref) gives one entry for each path, keyed by the `id` of the path. Each figure of a path is a point estimate, and the column of a figure is its distribution over the paths. The summary does no reduction over the paths, because a reduction of a ratio is not the ratio of a reduction. A caller takes `median(ps.sharpe)` or `DataFrames.describe(DataFrame(ps))`.
  - A vector of inputs gives the entries of each input in turn, keyed by the name of the candidate. A vector of one input gives vectors of length one.

The entries can cover different rows, and the summary does not refuse them. No figure pairs two series, so `n_periods`, `date0` and `date1` show the difference. This is different from [`forecast_evaluation_summary`](@ref), whose figures pair the forecasts.

# Inputs of the figures

The methods that take weights and returns first make the net returns with [`calc_net_returns`](@ref). So the summary of a portfolio includes its fees.

Every method takes a `benchmark`, a second return series over the same periods, and gives the three excess statistics against it. When no benchmark is given, the three are `NaN`. A given `benchmark` applies to every entry, and its length is checked against each entry. On a method that takes a prediction, `benchmark` defaults to the benchmark of the rows that the prediction scored, which [`regret_benchmark`](@ref) reads, one series for each member. So a prediction over returns data with a `B` gives the three statistics with no keyword, and `benchmark = nothing` turns them off. The other methods hold no prediction, and they read no `B`.

Each fold collapsed a benchmark matrix with its own weights, as [`collapse_benchmark`](@ref) states. Against such a benchmark, `excess_ret` is the return of ``(\\boldsymbol{X} - \\boldsymbol{B}) \\boldsymbol{w}``. That is the active return against the same weights in the benchmark assets, and not the excess over one index.

A method that takes a prediction also gives the **turnover** of the held path, the mean trade at a rebalance of a multi-period result. At `test_size = 1`, a rebalance is a period. A bare series, a single fold and a result of one fold record no rebalance, so their turnover is `NaN`.

The **effective number of assets** `n_eff` reads the weights of the entry with [`number_effective_assets`](@ref). Weights, and one [`OptimisationResult`](@ref) on its investable view, give the value of their weights. A path gives the mean over its folds of the value of each fold target, and a weight path the mean over its rows. The mean weight vector of a path is not used, because when the folds hold different assets, its value is larger than the value of any fold. A bare series holds no weights, so its `n_eff` is `NaN`. The risk contribution of each asset is not a figure of the summary, because its axis is the asset. [`risk_contribution`](@ref) gives it.

# Extra risk measures

The keyword `measures` takes a vector of pairs `name::Symbol => r::AbstractBaseRiskMeasure`. The value of each measure on each entry is `expected_risk_from_returns(r, ret)` on the same net series as the headline figures, with the sign of the measure. `ps.risk.name` reads it, and [`bigger_is_better`](@ref) gives its orientation. A name, and not the type of a measure, keys each value, because two CVaR levels have one type.

A measure that cannot read a series is refused at entry, before any figure is computed. [`supports_precomputed_returns`](@ref) is `false` for such a measure, for example a [`Variance`](@ref), a [`StandardDeviation`](@ref) or a [`TurnoverRiskMeasure`](@ref). [`expected_risk`](@ref) on weights and a prior evaluates it. The refusal holds for every input, so an [`OptimisationResult`](@ref) and a prediction give the same value for the same series.

# The risk-free rate

`rf` is the risk-free rate for each period, at the frequency of the returns, as in [`probabilistic_sharpe_ratio`](@ref) and [`MeanReturnRiskRatio`](@ref). It enters `sharpe`, `sharpe_stderr` and `sortino`, and no other figure. At `rf = 0` every figure is the figure of a zero rate.

**The series `ret` must be finite, as the Precomputed-returns contract states.** No method checks that the series is finite, because every internal caller gives a finite series. The prediction methods read the output of [`calc_net_returns`](@ref) for the fold, and a check of a long series costs time on every call. One entry that is not finite makes the mean, the volatility and every ratio not finite. The tail figure then gives a **finite wrong number** and not a `NaN`, because `partialsort` puts a `NaN` after every real number. A caller with a gapped series must first remove the gaps with `x[isfinite.(x)]`. A caller with a gapped panel must score it through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), which filters the Held Gaps once.

The standard error of the Sharpe ratio reads every observation as independent. So on a series scored under a Weight Drift, it is less than the true standard error, as the `# Mathematical definition` states. A long-run variance estimator corrects for serial dependence, but it needs a bandwidth, and no bandwidth is correct for every sample. The library does not build one. It applies no guard and no threshold, and it gives the figure with no change.

# Mathematical definition

The five headline statistics:

```math
\\begin{align}
\\mathrm{ann\\_return} &= m p\\\\
\\mathrm{ann\\_volatility} &= s \\sqrt{p}\\\\
\\mathrm{sharpe} &= \\dfrac{(m - r_{f}) p}{\\mathrm{ann\\_volatility}}\\\\
\\mathrm{sortino} &= \\dfrac{(m - r_{f}) p}{\\sqrt{\\dfrac{p}{T} \\sum\\limits_{t=1}^{T} \\min(x_{t}, 0)^{2}}}\\\\
\\mathrm{calmar} &= \\dfrac{\\mathrm{ann\\_return}}{\\left\\lvert \\min_{t} d_{t} \\right\\rvert}\\,.
\\end{align}
```

The three excess statistics against a benchmark, and the turnover of the held path of a multi-period result:

```math
\\begin{align}
\\mathrm{excess\\_ret} &= m_e p\\\\
\\mathrm{tracking\\_error} &= s_e \\sqrt{p}\\\\
\\mathrm{information\\_ratio} &= \\dfrac{\\mathrm{excess\\_ret}}{\\mathrm{tracking\\_error}}\\\\
\\mathrm{turnover} &= \\dfrac{1}{F - 1} \\sum_{f=2}^{F} \\left\\lVert \\boldsymbol{w}_f - \\hat{\\boldsymbol{w}}_{f-1} \\right\\rVert_1\\,.
\\end{align}
```

The standard error of the Sharpe ratio corrects for the skewness and the excess kurtosis of the returns. Its variance is the asymptotic variance of Equation 2 of [mertens2002](@cite), divided by ``T - 1`` as in Section 2.5 of [sharpe_stderr](@cite), and annualised:

```math
\\begin{align}
\\mathrm{sharpe\\_stderr} &= \\sqrt{\\dfrac{p}{T - 1} \\left(1 - g_{1} \\dfrac{m - r_{f}}{s} + \\dfrac{g_{2} + 2}{4} \\dfrac{(m - r_{f})^{2}}{s^{2}}\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - ``p``: Number of periods per year, `ppy`.
  - ``m``, ``s``: Sample mean and corrected sample standard deviation of ``\\boldsymbol{x}``.
  - ``r_{f}``: Risk-free rate for each period, `rf`.
  - ``\\boldsymbol{d}``: Drawdown path of the cumulative wealth of ``\\boldsymbol{x}``, compounded or not as `compound` states, with ``d_{t} \\leq 0``.
  - ``m_e``, ``s_e``: Sample mean and corrected sample standard deviation of the excess series ``\\boldsymbol{e} = \\boldsymbol{x} - \\boldsymbol{b}``.
  - ``F``: Number of folds of the multi-period result.
  - ``\\boldsymbol{w}_f``: Weights at the start of fold ``f``, its target on a solved fold.
  - ``\\hat{\\boldsymbol{w}}_f``: Weights of fold ``f`` after its last observation. They are the drifted holding when the fold has a Held Weights record, and ``\\boldsymbol{w}_f`` otherwise.
  - ``g_{1}``, ``g_{2}``: Sample skewness and sample excess kurtosis of ``\\boldsymbol{x}``.

The turnover puts every weight vector on the union of the universes of the folds, by asset name. A weight is zero at an asset that its fold does not hold. Each term of the sum is the trade at one rebalance. [`calc_turnover`](@ref) reads that quantity at one step, and [`Turnover`](@ref) bounds it. The drift in a fold is not a trade, so it enters the turnover only through ``\\hat{\\boldsymbol{w}}_{f-1}``.

The downside deviation of the Sortino ratio divides by ``T``, and not by the number of periods with a loss. This is the convention of Sortino and Price. With it, the ratios of two series with different frequencies of loss stay comparable.

The kurtosis term is ``(g_{2} + 2)/4``. Section 2.5 of [sharpe_stderr](@cite) states it as ``(\\gamma_{4} - 1)/4`` on the **raw** fourth standardised moment ``\\gamma_{4}``, and `StatsBase.kurtosis` gives the **excess** moment ``g_{2} = \\gamma_{4} - 3``. The two forms are equal. Only this form becomes the naive ``\\sqrt{(1 + \\mathrm{SR}^{2}/2)/T}`` on a normal series. Over 200,000 samples of 250 normal returns at a Sharpe ratio per period of 0.5, the sample standard deviation of the Sharpe ratio is **0.06728323**. This expression at the population moments gives **0.06721661**, and the same expression without the ``+2`` gives **0.06337243**.

A return series that is not normal makes a Sharpe ratio less precise than the naive expression shows, and a negative skew makes it less precise again. A real portfolio has such a series, so the summary gives the standard error with the ratio.

The expression corrects for the third and fourth moments and **not** for serial dependence, so it reads every observation as independent of the others. A series scored under a Weight Drift is serially dependent through the weights that it held, because a position that grew has a larger weight in the next observation. So on such a series, this figure is less than the true standard error.

# Algorithm

Every method checks its keywords with [`summary_options`](@ref) first, so no figure is computed for a call that is not valid. Then it makes one entry for each series of the input. A method that takes weights first makes the net returns with [`calc_net_returns`](@ref). A method that takes a prediction reads each member with [`prediction_entries`](@ref), and a population reads each path with [`population_entries`](@ref). [`summarise_returns`](@ref) does the steps below on each entry, and [`summarise_entries`](@ref) gives scalar fields for one entry, or stacks the entries with [`stack_summaries`](@ref).

 1. Check `alpha`, `ppy`, `rf` and `measures`, as `# Validation` states.
 2. Take `T`, the number of periods. Take `m`, the sample mean of `ret`, and clamp it to the least and the greatest entry of `ret`. Take `s`, the corrected sample standard deviation of `ret` about `m`. The clamp changes nothing in exact arithmetic. The rounded mean of a constant `ret` can differ from its common value, and the clamp makes `m` equal to that value, so `s` is exactly zero.
 3. Annualise `m` and `s`, to get `ann_ret` and `ann_vol`.
 4. Annualise `m - rf`, to get `ann_ex`. Divide `ann_ex` by `ann_vol`, to get `sharpe`. When `ann_vol` is not positive, `sharpe` is `NaN`.
 5. Set every positive entry of `ret` to zero. Square the entries, average them over all `T` periods, annualise, and take the square root, to get the downside deviation `ddev`. Divide `ann_ex` by `ddev`, to get `sortino`. When `ddev` is not positive, `sortino` is `NaN`.
 6. Build the cumulative wealth series with [`cumulative_returns`](@ref), and take its drawdown path with [`drawdowns`](@ref). Read `max_dd`, the minimum of that path. Divide `ann_ret` by the absolute value of `max_dd`, to get `calmar`. When `max_dd` is not negative, `calmar` is `NaN`.
 7. Compute [`ConditionalValueatRisk`](@ref) at `alpha` on `ret`, and change its sign, to get `cvar_val` in return space.
 8. Make the Sharpe ratio per period `sr_p`, from `m - rf` and `s`. Correct its variance `var_sr` for the sample skewness and the sample excess kurtosis of `ret`. Annualise the square root, to get `sharpe_se`. When `var_sr` is not positive, `sharpe_se` is `NaN`.
 9. Compute the three excess statistics with [`excess_statistics`](@ref). With a `benchmark`, the function checks its length and takes `e = ret - benchmark`. It annualises the mean of `e`, clamped as in step 2, and the standard deviation of `e` about that mean, to get `excess_ret` and `tracking_error`. It divides them, to get `information_ratio`, which is `NaN` when `tracking_error` is not positive. With no benchmark, all three are `NaN`.
10. Take the turnover that the entry carries, from [`held_path_turnover`](@ref). A `nothing` from that function, and every input that holds no prediction, gives a `NaN`.
11. Take the effective number of assets that the entry carries. An entry with no weights gives a `NaN`.
12. Evaluate each extra measure on `ret` with [`expected_risk_from_returns`](@ref), to get `risk`.
13. Read `date0` and `date1` from the timestamps of the entry with [`series_dates`](@ref). Put the keys, the inputs and the statistics into a [`PerformanceSummaryResult`](@ref).

# Arguments

  - `ret`: Vector of periodic portfolio returns.
  - `w`: Portfolio weights, a weight path `observations × assets`, or one weight vector for each member.
  - $(arg_dict[:X])
  - `fees`: Optional transaction fees.
  - `rd`: [`ReturnsResult`](@ref) that holds the asset returns. Its `ts` gives `date0` and `date1`.
  - `res`: An [`OptimisationResult`](@ref). [`result_investable_view`](@ref) puts its weights and fees, and the `rd` of the caller, on the investable universe of the result.
  - `pred`: A prediction result, from [`predict`](@ref) or [`cross_val_predict`](@ref).
  - `ppred`: A population of paths.
  - `rets`, `ress`, `preds`: A vector of candidates. Each element of `preds` can be a single prediction, a multi-period prediction or a population.
  - `names`: One name for each element of the input vector, or `nothing` for `"Portfolio 1"`, `"Portfolio 2"`, ….
  - $(arg_dict[:ps_ppy]) It defaults to `1`, which gives the statistics per period at any frequency of the data.
  - $(arg_dict[:ps_alpha])
  - $(arg_dict[:ps_compound])
  - `rf`: Risk-free rate for each period, at the frequency of the returns. It defaults to `0`.
  - `measures`: A vector or a tuple of pairs `name::Symbol => r::AbstractBaseRiskMeasure`, the extra risk measures. It defaults to none.
  - `benchmark`: Benchmark return series over the same periods as each entry, or `nothing` for no excess statistics. It defaults to `nothing`, and on a prediction to the benchmark of its rows.

Every keyword other than `names` is one value for the whole call, so the entries always agree on it.

# Validation

  - `0 < alpha < 1`, `ppy > 0` and `isfinite(rf)`. Otherwise the function raises a `DomainError`.
  - Each entry of `measures` is a `Symbol => AbstractBaseRiskMeasure` pair, no name repeats, and no name is a column of the table, [`performance_summary_columns`](@ref). Otherwise the function raises an `ArgumentError`.
  - Each measure in `measures` reads a series. Otherwise the function raises an `ArgumentError` that names each refused measure by its name and its type.
  - `length(benchmark) == length(ret)` for each entry when a benchmark is given. Otherwise the function raises a `DimensionMismatch`.
  - A vector input and a population are not empty. Otherwise the function raises an [`IsEmptyError`](@ref).
  - `length(names)` is the length of the input vector. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - `summary::PerformanceSummaryResult`: The computed summary, with scalar fields for an input that holds one series, and one vector entry for each series otherwise.

# Related

  - [`PerformanceSummaryResult`](@ref)
  - [`DataFrames.DataFrame(ps::PerformanceSummaryResult)`](@ref)
  - [`plot_performance_summary`](@ref)
  - [`expected_risk_from_returns`](@ref)
  - [`number_effective_assets`](@ref)
  - [`forecast_evaluation_summary`](@ref)
  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
  - [`drawdowns`](@ref)
  - [`calc_turnover`](@ref)
  - [`held_path_turnover`](@ref)
  - [`log_wealth_regret`](@ref)

# References

  - $(ref_dict[:mertens2002]) Section 1, Equation 2.
  - $(ref_dict[:sharpe_stderr]) Section 2.5.
"""
function performance_summary(ret::VecNum; kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, ret)
end
"""
    excess_statistics(ret::VecNum, benchmark::Nothing, ann::Number, m::Number)
    excess_statistics(ret::VecNum, benchmark::VecNum, ann::Number, m::Number)

Compute the three excess statistics of [`performance_summary`](@ref) against a benchmark series, or give three `NaN` values when no benchmark is given.

The closed forms are the `excess_ret`, `tracking_error` and `information_ratio` rows of the `# Mathematical definition` of [`performance_summary`](@ref).

# Arguments

  - `ret`: Periodic portfolio returns.
  - `benchmark`: Benchmark returns over the same periods, or `nothing`.
  - `ann`: Annualisation factor.
  - `m`: Sample mean of `ret`. With no benchmark, each `NaN` has the number type that its statistic has with a benchmark. The function derives that type from `m` and `ann`.

# Validation

  - `length(benchmark) == length(ret)`. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - `(excess_ret, tracking_error, information_ratio)`: Annualised mean and standard deviation of `ret - benchmark`, and their ratio. The ratio is `NaN` when the tracking error is not positive, that is when the tracking error is zero or `NaN`.

# Related

  - [`performance_summary`](@ref)
  - [`PerformanceSummaryResult`](@ref)
"""
function excess_statistics(::VecNum, ::Nothing, ann::Number, m::Number)
    # Each `NaN` takes the number type the statistic takes with a benchmark.
    excess_ret = oftype(m * ann, NaN)
    tracking_error = oftype(m * sqrt(ann), NaN)
    return excess_ret, tracking_error, oftype(one(excess_ret) / one(tracking_error), NaN)
end
function excess_statistics(ret::VecNum, benchmark::VecNum, ann::Number, ::Number)
    @argcheck(length(benchmark) == length(ret),
              DimensionMismatch("`length(benchmark) == length(ret)` must hold.\nlength(benchmark) => $(length(benchmark))\nlength(ret) => $(length(ret))"))
    e = ret .- benchmark
    # The clamp moves nothing in exact arithmetic. On a constant excess series it makes the
    # mean the common value, so the tracking error is exactly zero and the guard fires.
    me = clamp(mean(e), extrema(e)...)
    excess_ret = me * ann
    tracking_error = std(e; mean = me) * sqrt(ann)
    information_ratio = if tracking_error > zero(tracking_error)
        excess_ret / tracking_error
    else
        oftype(one(excess_ret) / one(tracking_error), NaN)
    end
    return excess_ret, tracking_error, information_ratio
end
"""
    held_path_turnover(pred::PredictionResult, j::Integer = 1)
    held_path_turnover(pred::MultiPeriodPredictionResult, j::Integer = 1)

Compute the mean turnover for each rebalance of the held path of member `j` of a prediction result, as [`performance_summary`](@ref) gives it.

A fold trades from the weights that the previous fold held after its last observation, and not from the target of that fold. Under a Held Weights record the two are different, so the turnover of a drifted path is not the turnover of its targets. The closed form is the `turnover` row of the `# Mathematical definition` of [`performance_summary`](@ref). A prediction whose optimiser gives many portfolios holds one path for each member, and `j` names the member.

# Algorithm

 1. Stack the weights of member `j` at the start of every fold, to get `W0` with one row for each fold. [`fold_target`](@ref) reads the weights, and [`stacked_fold_weights`](@ref) puts them on the union of the universes of the folds, by asset name.
 2. Stack the weights of member `j` of every fold after its last observation on the same axis, to get `W1`. [`fold_held`](@ref) reads these weights.
 3. Take `F`, the number of folds. When `F < 2`, the result records no rebalance, and the answer is `nothing`.
 4. Average the trade from row `f - 1` of `W1` to row `f` of `W0` over the `F - 1` rebalances, to get the turnover.

# Arguments

  - `pred`: The prediction result.
  - `j`: The index of the member.

# Returns

  - `tn::Option{<:Number}`: The mean turnover for each rebalance. It is `nothing` when the result records no rebalance, that is for a single prediction result and for a multi-period result of one fold. The summary writes `nothing` as `NaN`.

# Related

  - [`performance_summary`](@ref)
  - [`stacked_fold_weights`](@ref)
  - [`calc_turnover`](@ref)
  - [`Turnover`](@ref)
"""
function held_path_turnover(::PredictionResult, ::Integer = 1)
    return nothing
end
function held_path_turnover(pred::MultiPeriodPredictionResult, j::Integer = 1)
    W0 = stacked_fold_weights(pred, f -> fold_target(f, j))
    W1 = stacked_fold_weights(pred, f -> fold_held(f, j))
    F = size(W0, 1)
    if F < 2
        return nothing
    end
    return sum(sum(abs, view(W0, f, :) - view(W1, f - 1, :)) for f in 2:F) / (F - 1)
end
"""
    summarise_returns(e::NamedTuple, opts::NamedTuple) -> PerformanceSummaryResult

Compute the [`PerformanceSummaryResult`](@ref) of one entry, with scalar fields.

This function is the one site that computes the figures of a summary. It does the steps of [`performance_summary`](@ref) on the series of one entry. Every `performance_summary` method calls it once for each entry. An entry that holds no weight path gives `nothing` for the turnover, and an entry that holds no weights gives `nothing` for the effective number of assets. This function writes each `nothing` as `NaN`.

# Arguments

  - `e`: One entry, from [`summary_entry`](@ref).
  - `opts`: The checked keywords of the call, from [`summary_options`](@ref).

# Validation

  - `length(e.benchmark) == length(e.ret)` when the entry holds a benchmark. [`excess_statistics`](@ref) checks it, and otherwise raises a `DimensionMismatch`.

# Returns

  - `summary::PerformanceSummaryResult`: The summary of the entry, with scalar fields.

# Related

  - [`performance_summary`](@ref)
  - [`summarise_entries`](@ref)
  - [`excess_statistics`](@ref)
  - [`held_path_turnover`](@ref)
"""
function summarise_returns(e::NamedTuple, opts::NamedTuple)::PerformanceSummaryResult
    ret = e.ret
    alpha = opts.alpha
    compound = opts.compound
    T = length(ret)
    ann = opts.ppy
    # The clamp moves nothing in exact arithmetic. On a constant series it makes the mean the
    # common value, so `s` is exactly zero and every guard below fires.
    m = clamp(mean(ret), extrema(ret)...)
    s = std(ret; mean = m)
    ann_ret = m * ann
    ann_vol = s * sqrt(ann)
    # The excess over the risk-free rate. At `rf = 0` it is `m` exactly, so no figure moves.
    ex = m - opts.rf
    ann_ex = ex * ann
    # A guarded statistic's `NaN` takes the number type its unguarded branch lands in, read
    # off `one` so no division by zero runs, and never the `Float64` of a bare literal.
    sharpe = if ann_vol > zero(ann_vol)
        ann_ex / ann_vol
    else
        oftype(one(ann_ex) / one(ann_vol), NaN)
    end
    neg_ret = min.(ret, zero(eltype(ret)))
    ddev = sqrt(mean(abs2, neg_ret) * ann)
    sortino = ddev > zero(ddev) ? ann_ex / ddev : oftype(one(ann_ex) / one(ddev), NaN)
    cret = cumulative_returns(ret, compound)
    dd_series = drawdowns(cret, compound; cX = true)
    max_dd = minimum(dd_series)
    calmar = if max_dd < zero(max_dd)
        ann_ret / abs(max_dd)
    else
        oftype(one(ann_ret) / one(max_dd), NaN)
    end
    cvar_val = -ConditionalValueatRisk(; alpha = alpha)(ret)
    # The per-period Sharpe ratio, corrected by the third and fourth standardised
    # moments. `T - 1` matches the corrected `std` used for `sharpe` itself.
    sr_p = s > zero(s) ? ex / s : oftype(one(ex) / one(s), NaN)
    var_sr = sharpe_ratio_variance_factor(sr_p, StatsBase.skewness(ret),
                                          StatsBase.kurtosis(ret)) / (T - 1)
    sharpe_se = if var_sr > zero(var_sr)
        sqrt(var_sr * ann)
    else
        oftype(sqrt(one(var_sr) * one(ann)), NaN)
    end
    excess_ret, tracking_error, information_ratio = excess_statistics(ret, e.benchmark, ann,
                                                                      m)
    risk = map(r -> expected_risk_from_returns(r, ret), opts.measures)
    date0, date1 = series_dates(e.ts)
    return PerformanceSummaryResult(e.name, e.path, e.member, date0, date1, T, ann, alpha,
                                    compound, opts.rf, ann_ret, ann_vol, sharpe, sharpe_se,
                                    sortino, calmar, max_dd, cvar_val, excess_ret,
                                    tracking_error, information_ratio,
                                    something(e.turnover, oftype(ann_ret, NaN)),
                                    something(e.n_eff, oftype(ann_ret, NaN)), opts.measures,
                                    risk)
end
"""
    performance_summary_columns()

The columns of the table of a [`PerformanceSummaryResult`](@ref), each with the field that it reads.

[`DataFrames.DataFrame(ps::PerformanceSummaryResult)`](@ref) writes the columns in this order: the three keys, the two dates and the number of periods, the twelve released figures in the order of the struct, and the effective number of assets. The column of each extra measure comes after them. [`summary_measures`](@ref) refuses a measure whose name is one of these columns.

# Returns

  - `cols::NamedTuple`: The field of each column, keyed by the name of the column.

# Related

  - [`DataFrames.DataFrame(ps::PerformanceSummaryResult)`](@ref)
  - [`summary_measures`](@ref)
"""
function performance_summary_columns()
    return (; name = :names, path = :paths, member = :members, date0 = :date0,
            date1 = :date1, n_periods = :n_periods, ann_return = :ann_return,
            ann_volatility = :ann_volatility, sharpe = :sharpe,
            sharpe_stderr = :sharpe_stderr, sortino = :sortino, calmar = :calmar,
            max_drawdown = :max_drawdown, cvar = :cvar, excess_ret = :excess_ret,
            tracking_error = :tracking_error, information_ratio = :information_ratio,
            turnover = :turnover, n_eff = :n_eff)
end
"""
    summary_measures(measures::Tuple{})
    summary_measures(measures::Union{Tuple, AbstractVector})

Check the extra risk measures of a [`performance_summary`](@ref) call, and key them by name.

The check runs before any figure is computed, so a call with a measure that cannot read a series computes nothing. It collects every such measure, and the error names all of them.

# Algorithm

 1. For each entry, check that it is a pair `name::Symbol => r::AbstractBaseRiskMeasure`, that its name is new, and that its name is not a column of [`performance_summary_columns`](@ref).
 2. Collect each measure whose [`supports_precomputed_returns`](@ref) is `false`. When there is one, raise an error that names each by its name and its type.
 3. Key the measures by their names, in the order of `measures`.

# Arguments

  - `measures`: A vector or a tuple of pairs `name::Symbol => r::AbstractBaseRiskMeasure`.

# Validation

  - Each entry is a pair of a `Symbol` and an `AbstractBaseRiskMeasure`, no name repeats, and no name is a column of the table. Otherwise the function raises an `ArgumentError`.
  - Each measure reads a series. Otherwise the function raises an `ArgumentError`.

# Returns

  - `ms::NamedTuple`: The measures, keyed by name. It is empty when `measures` is empty.

# Related

  - [`performance_summary`](@ref)
  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function summary_measures(::Tuple{})
    return (;)
end
function summary_measures(measures::Union{Tuple, AbstractVector})
    cols = keys(performance_summary_columns())
    ks = Symbol[]
    for p in measures
        @argcheck(p isa Pair && first(p) isa Symbol && last(p) isa AbstractBaseRiskMeasure,
                  ArgumentError("each entry of `measures` must be a pair `name::Symbol => r::AbstractBaseRiskMeasure`, and one entry is $(repr(p))"))
        k = first(p)
        @argcheck(!(k in ks),
                  ArgumentError("the name `$(k)` names more than one entry of `measures`"))
        @argcheck(!(k in cols),
                  ArgumentError("the name `$(k)` of a measure is a column of the summary table, so it cannot name a measure"))
        push!(ks, k)
    end
    refused = ["`$(first(p))` (`$(typeof(last(p)))`)"
               for p in measures if !supports_precomputed_returns(last(p))]
    @argcheck(isempty(refused),
              ArgumentError("a measure in `measures` cannot be evaluated on a return series, because it needs portfolio weights or per-asset data: $(join(refused, ", ")). Evaluate it with `expected_risk` on weights and a prior."))
    return NamedTuple{Tuple(ks)}(Tuple(last(p) for p in measures))
end
"""
    summary_options(; ppy::Number = 1, alpha::Number = 0.05, compound::Bool = false,
                    rf::Number = 0, measures = (),
                    benchmark::Option{<:VecNum} = nothing) -> NamedTuple

Check the keywords of a [`performance_summary`](@ref) call, and give the values that every entry shares.

Every method of the verb calls this function first, so a call that is not valid computes no figure. The function checks the type of `benchmark` and does not keep it, because [`summary_benchmark`](@ref) reads whether the caller gave one.

# Arguments

  - `ppy`, `alpha`, `compound`, `rf`, `measures`, `benchmark`: As [`performance_summary`](@ref) states them.

# Validation

  - `0 < alpha < 1`, `ppy > 0` and `isfinite(rf)`. Otherwise the function raises a `DomainError`.
  - The rules of [`summary_measures`](@ref).

# Returns

  - `opts::NamedTuple`: `ppy`, `alpha`, `compound`, `rf`, and `measures` keyed by name.

# Related

  - [`performance_summary`](@ref)
  - [`summary_measures`](@ref)
  - [`summary_benchmark`](@ref)
"""
function summary_options(; ppy::Number = 1, alpha::Number = 0.05, compound::Bool = false,
                         rf::Number = 0, measures::Union{Tuple, AbstractVector} = (),
                         benchmark::Option{<:VecNum} = nothing)
    assert_unit_interval(alpha, :alpha)
    @argcheck(ppy > zero(ppy), DomainError(ppy, "ppy must be positive"))
    @argcheck(isfinite(rf), DomainError(rf, "rf must be finite"))
    return (; ppy = ppy, alpha = alpha, compound = compound, rf = rf,
            measures = summary_measures(measures))
end
"""
    summary_benchmark(kwargs)

Read the `benchmark` keyword of a [`performance_summary`](@ref) call, and state whether the caller gave it.

A prediction reads the benchmark of its own rows when the caller gives none, and `benchmark = nothing` turns the excess statistics off. So `nothing` cannot also mean that the caller gave no keyword. This function wraps a given value in `Some`, and [`row_benchmark`](@ref) unwraps it.

# Arguments

  - `kwargs`: The keywords of the call.

# Returns

  - `bm::Option{<:Some}`: `Some(benchmark)` when the caller gave the keyword, and `nothing` otherwise.

# Related

  - [`row_benchmark`](@ref)
  - [`summary_options`](@ref)
"""
function summary_benchmark(kwargs)
    return haskey(kwargs, :benchmark) ? Some(kwargs[:benchmark]) : nothing
end
"""
    row_benchmark(bm::Nothing, own)
    row_benchmark(bm::Some, own)

The benchmark series of one entry: the benchmark of the caller when the caller gave one, and the benchmark of its own rows otherwise.

# Arguments

  - `bm`: The benchmark of the caller, from [`summary_benchmark`](@ref).
  - `own`: The benchmark series of the rows of the entry, or `nothing` when the input holds no benchmark.

# Returns

  - `b::Option{<:VecNum}`: The benchmark series of the entry, or `nothing`.

# Related

  - [`summary_benchmark`](@ref)
  - [`regret_benchmark`](@ref)
"""
function row_benchmark(::Nothing, own)
    return own
end
function row_benchmark(bm::Some, ::Any)
    return something(bm)
end
"""
    summary_entry(ret::VecNum, benchmark::Option{<:VecNum}, turnover::Option{<:Number},
                  n_eff::Option{<:Number}, ts::Option{<:AbstractVector};
                  path = nothing, member = nothing) -> NamedTuple

One entry of a [`PerformanceSummaryResult`](@ref): a series, with the inputs of its figures and its keys.

[`summarise_returns`](@ref) reads an entry. The name of the candidate is `nothing` here, and [`entry_list`](@ref) sets it.

# Arguments

  - `ret`: The net return series.
  - `benchmark`: The benchmark series of the entry, or `nothing`.
  - `turnover`: The mean turnover of the held path, or `nothing`.
  - `n_eff`: The effective number of assets, or `nothing` when the entry holds no weights.
  - `ts`: The timestamps of the series, or `nothing`.
  - `path`: The key of the path in a population, or `nothing`.
  - `member`: The index of the member, or `nothing`.

# Returns

  - `e::NamedTuple`: The entry.

# Related

  - [`summarise_returns`](@ref)
  - [`input_entries`](@ref)
"""
function summary_entry(ret::VecNum, benchmark::Option{<:VecNum}, turnover::Option{<:Number},
                       n_eff::Option{<:Number}, ts::Option{<:AbstractVector};
                       path = nothing, member = nothing)
    return (; ret = ret, benchmark = benchmark, turnover = turnover, n_eff = n_eff, ts = ts,
            name = nothing, path = path, member = member)
end
"""
    series_dates(ts::Nothing)
    series_dates(ts::AbstractVector)

The first and the last timestamp of the series of an entry, or two `nothing` values when the series carries no timestamps.

# Arguments

  - `ts`: The timestamps of the series, or `nothing`.

# Returns

  - `(date0, date1)`: The first and the last timestamp, or `(nothing, nothing)`.

# Related

  - [`summarise_returns`](@ref)
  - [`brinson_attribution`](@ref)
"""
function series_dates(::Nothing)
    return nothing, nothing
end
function series_dates(ts::AbstractVector)
    return first(ts), last(ts)
end
"""
    weights_n_eff(w::VecNum)
    weights_n_eff(w::MatNum)

The effective number of assets of a weight vector, or the mean over the rows of a weight path `observations × assets`, as [`performance_summary`](@ref) gives it.

# Arguments

  - `w`: A weight vector, or a weight path.

# Returns

  - `neff::Number`: The effective number of assets.

# Related

  - [`number_effective_assets`](@ref)
  - [`prediction_n_eff`](@ref)
"""
function weights_n_eff(w::VecNum)
    return number_effective_assets(w)
end
function weights_n_eff(w::MatNum)
    return mean(number_effective_assets, eachrow(w))
end
"""
    prediction_n_eff(pred::PredictionResult, j::Integer)
    prediction_n_eff(pred::MultiPeriodPredictionResult, j::Integer)

The effective number of assets of member `j` of a prediction, as [`performance_summary`](@ref) gives it.

A single prediction gives the value of its target. A multi-period prediction gives the mean over its folds of the value of each fold target, and not the value of the mean target. When the folds hold different assets, the mean target spreads over the union of their universes, and its value is larger than the value of any fold.

# Arguments

  - `pred`: The prediction result.
  - `j`: The index of the member.

# Returns

  - `neff::Number`: The effective number of assets.

# Related

  - [`number_effective_assets`](@ref)
  - [`fold_target`](@ref)
  - [`weights_n_eff`](@ref)
"""
function prediction_n_eff(pred::PredictionResult, j::Integer)
    return number_effective_assets(fold_target(pred, j))
end
function prediction_n_eff(pred::MultiPeriodPredictionResult, j::Integer)
    return mean(f -> number_effective_assets(fold_target(f, j)), pred.pred)
end
"""
    prediction_entries(pred::PredRes_MultiPredRes, bm::Option{<:Some}, path = nothing)

The entries of a prediction: one entry when it holds one series, and one entry for each member when its optimiser gives many portfolios.

The function does not read the first member of a prediction alone, as [`regret_series`](@ref) gives the series. Each member reads its own series, its own benchmark series through [`member_vector`](@ref), its own turnover through [`held_path_turnover`](@ref) and its own weights through [`prediction_n_eff`](@ref).

# Arguments

  - `pred`: The prediction result.
  - `bm`: The benchmark of the caller, from [`summary_benchmark`](@ref).
  - `path`: The key of the path in a population, or `nothing`.

# Returns

  - `e::NamedTuple`: One entry, for a prediction that holds one series.
  - `es::Vector`: One entry for each member, for a prediction that holds many series.

# Related

  - [`summary_entry`](@ref)
  - [`population_entries`](@ref)
  - [`regret_series`](@ref)
  - [`regret_benchmark`](@ref)
"""
function prediction_entries(pred::PredRes_MultiPredRes, bm::Option{<:Some}, path = nothing)
    X, ts = regret_series(pred)
    return prediction_entries(X, ts, regret_benchmark(pred), pred, bm, path)
end
function prediction_entries(X::VecNum, ts::Option{<:AbstractVector}, B::Option{<:VecNum},
                            pred::PredRes_MultiPredRes, bm::Option{<:Some}, path)
    return summary_entry(X, row_benchmark(bm, B), held_path_turnover(pred),
                         prediction_n_eff(pred, 1), ts; path = path)
end
function prediction_entries(X::VecVecNum, ts::Option{<:AbstractVector},
                            B::Option{<:VecNum_VecVecNum}, pred::PredRes_MultiPredRes,
                            bm::Option{<:Some}, path)
    return [summary_entry(X[j], row_benchmark(bm, member_vector(B, j)),
                          held_path_turnover(pred, j), prediction_n_eff(pred, j), ts;
                          path = path, member = j) for j in eachindex(X)]
end
"""
    population_entries(ppred::PopulationPredictionResult, bm::Option{<:Some}) -> Vector

The entries of a population: the entries of each path in turn, each keyed by its path.

A multi-period path is keyed by its `id`. A single-fold prediction result in a population has no `id`, so its position in the population keys it, as [`population_ids`](@ref) numbers the paths that lack an `id`.

# Algorithm

 1. Check that the population holds a path.
 2. Read the entries of each path with [`prediction_entries`](@ref), keyed by [`population_path_key`](@ref): the `id` of the path or its position.
 3. Join the entries of the paths in the order of the population.

# Arguments

  - `ppred`: The population.
  - `bm`: The benchmark of the caller, from [`summary_benchmark`](@ref).

# Validation

  - The population holds at least one path. Otherwise the function raises an [`IsEmptyError`](@ref).

# Returns

  - `es::Vector`: The entries, in the order path, then member.

# Related

  - [`prediction_entries`](@ref)
  - [`population_path_key`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function population_entries(ppred::PopulationPredictionResult, bm::Option{<:Some})
    paths = ppred.pred
    @argcheck(!isempty(paths),
              IsEmptyError("the population holds no path, so there is nothing to summarise"))
    # The loop reads each path by its index, not through `enumerate`, which gives JET a false
    # report on an untyped element (issue #1341).
    es = map(eachindex(paths)) do k
        p = paths[k]
        return entry_list(prediction_entries(p, bm, population_path_key(p, k)), nothing)
    end
    return reduce(vcat, es)
end
"""
    population_path_key(p::MultiPeriodPredictionResult, k::Integer)
    population_path_key(p::PredictionResult, k::Integer)

The key of path `k` of a population, as the `paths` field of a [`PerformanceSummaryResult`](@ref) holds it.

A multi-period path is keyed by its `id`. A single-fold prediction result has no `id`, so its position `k` in the population keys it, as [`population_ids`](@ref) numbers the paths that lack an `id`. [`portfolio_report`](@ref) heads the column of each path with the same key.

# Arguments

  - `p`: The path.
  - `k`: The position of the path in the population.

# Returns

  - `key`: The `id` of a multi-period path, or `k`.

# Related

  - [`population_entries`](@ref)
  - [`population_ids`](@ref)
"""
function population_path_key(p::MultiPeriodPredictionResult, ::Integer)
    return p.id
end
function population_path_key(::PredictionResult, k::Integer)
    return k
end
"""
    input_entries(bm::Option{<:Some}, ret::VecNum)
    input_entries(bm::Option{<:Some}, w::Union{ArrNum, VecVecNum}, X::MatNum,
                  fees::Option{<:Fees} = nothing, ts::Option{<:AbstractVector} = nothing)
    input_entries(bm::Option{<:Some}, w::Union{ArrNum, VecVecNum}, rd::ReturnsResult,
                  fees::Option{<:Fees} = nothing)
    input_entries(bm::Option{<:Some}, res::OptimisationResult, rd::ReturnsResult)
    input_entries(bm::Option{<:Some}, pred::PredRes_MultiPredRes)
    input_entries(bm::Option{<:Some}, ppred::PopulationPredictionResult)

The entries of one input of [`performance_summary`](@ref): one entry for an input that holds one series, and a vector of entries otherwise.

A bare series holds no weights. Weights give their net returns through [`calc_net_returns`](@ref) and their effective number of assets through [`weights_n_eff`](@ref), and a `VecVecNum` gives one entry for each member. An [`OptimisationResult`](@ref) reads its weights, its returns and its fee on its investable universe through [`result_investable_view`](@ref).

# Arguments

  - `bm`: The benchmark of the caller, from [`summary_benchmark`](@ref).
  - `ret`, `w`, `X`, `fees`, `rd`, `res`, `pred`, `ppred`: As [`performance_summary`](@ref) states them.
  - `ts`: The timestamps of the rows of `X`, or `nothing`.

# Returns

  - `e::NamedTuple`: One entry, for an input that holds one series.
  - `es::Vector`: One entry for each series otherwise.

# Related

  - [`summary_entry`](@ref)
  - [`prediction_entries`](@ref)
  - [`population_entries`](@ref)
  - [`summarise_entries`](@ref)
"""
function input_entries(bm::Option{<:Some}, ret::VecNum)
    return summary_entry(ret, row_benchmark(bm, nothing), nothing, nothing, nothing)
end
function input_entries(bm::Option{<:Some}, w::ArrNum, X::MatNum,
                       fees::Option{<:Fees} = nothing,
                       ts::Option{<:AbstractVector} = nothing)
    return summary_entry(calc_net_returns(w, X, fees), row_benchmark(bm, nothing), nothing,
                         weights_n_eff(w), ts)
end
function input_entries(bm::Option{<:Some}, w::VecVecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing,
                       ts::Option{<:AbstractVector} = nothing)
    b = row_benchmark(bm, nothing)
    return [summary_entry(calc_net_returns(w[j], X, fees), b, nothing,
                          number_effective_assets(w[j]), ts; member = j)
            for j in eachindex(w)]
end
function input_entries(bm::Option{<:Some}, w::Union{ArrNum, VecVecNum}, rd::ReturnsResult,
                       fees::Option{<:Fees} = nothing)
    return input_entries(bm, w, rd.X, fees, rd.ts)
end
function input_entries(bm::Option{<:Some}, res::OptimisationResult, rd::ReturnsResult)
    _, w, X, fees = result_investable_view(res, rd.X)
    return input_entries(bm, w, X, fees, rd.ts)
end
function input_entries(bm::Option{<:Some}, pred::PredRes_MultiPredRes)
    return prediction_entries(pred, bm)
end
function input_entries(bm::Option{<:Some}, ppred::PopulationPredictionResult)
    return population_entries(ppred, bm)
end
"""
    entry_list(e::NamedTuple, name)
    entry_list(es::Vector{<:NamedTuple}, name)

The entries of one input as a vector, each with the name of its candidate.

# Arguments

  - `e`, `es`: One entry, or a vector of entries.
  - `name`: The name of the candidate, or `nothing`.

# Returns

  - `es::Vector`: The entries, with `name` set.

# Related

  - [`candidate_summary`](@ref)
  - [`population_entries`](@ref)
"""
function entry_list(e::NamedTuple, name)
    return [merge(e, (; name = name))]
end
function entry_list(es::Vector{<:NamedTuple}, name)
    return [merge(es[i], (; name = name)) for i in eachindex(es)]
end
"""
    summarise_entries(e::NamedTuple, opts::NamedTuple)
    summarise_entries(es::AbstractVector, opts::NamedTuple)

Summarise the entries of a [`performance_summary`](@ref) call. One entry gives scalar fields, and a vector of entries gives one vector entry for each, also when the vector holds one entry.

# Arguments

  - `e`, `es`: One entry, or a vector of entries.
  - `opts`: The checked keywords of the call, from [`summary_options`](@ref).

# Returns

  - `summary::PerformanceSummaryResult`: The summary.

# Related

  - [`summarise_returns`](@ref)
  - [`stack_summaries`](@ref)
"""
function summarise_entries(e::NamedTuple, opts::NamedTuple)
    return summarise_returns(e, opts)
end
function summarise_entries(es::AbstractVector, opts::NamedTuple)
    return stack_summaries([summarise_returns(e, opts) for e in es], opts)
end
"""
    stack_summaries(rs::AbstractVector{<:PerformanceSummaryResult},
                    opts::NamedTuple) -> PerformanceSummaryResult

Stack the summaries of single entries into one summary with one vector entry for each.

# Algorithm

 1. Collect each field of an entry into a vector, in the order of `rs`.
 2. Collect the value of each extra measure into a vector, keyed by the name of the measure.
 3. Take `ppy`, `alpha`, `compound`, `rf` and `measures` once, from `opts`.

# Arguments

  - `rs`: The summary of each entry, each with scalar fields.
  - `opts`: The checked keywords of the call, from [`summary_options`](@ref).

# Returns

  - `summary::PerformanceSummaryResult`: The summary with one vector entry for each element of `rs`.

# Related

  - [`summarise_entries`](@ref)
  - [`summarise_returns`](@ref)
"""
function stack_summaries(rs::AbstractVector{<:PerformanceSummaryResult}, opts::NamedTuple)
    # The field travels as a `Val`, so its name is a constant and each column is inferred.
    col(::Val{F}) where {F} = [getfield(r, F) for r in rs]
    ks = keys(opts.measures)
    risk = NamedTuple{ks}(Tuple([r.risk[k] for r in rs] for k in ks))
    return PerformanceSummaryResult(col(Val(:names)), col(Val(:paths)), col(Val(:members)),
                                    col(Val(:date0)), col(Val(:date1)),
                                    col(Val(:n_periods)), opts.ppy, opts.alpha,
                                    opts.compound, opts.rf, col(Val(:ann_return)),
                                    col(Val(:ann_volatility)), col(Val(:sharpe)),
                                    col(Val(:sharpe_stderr)), col(Val(:sortino)),
                                    col(Val(:calmar)), col(Val(:max_drawdown)),
                                    col(Val(:cvar)), col(Val(:excess_ret)),
                                    col(Val(:tracking_error)), col(Val(:information_ratio)),
                                    col(Val(:turnover)), col(Val(:n_eff)), opts.measures,
                                    risk)
end
"""
    summarise_input(kwargs, args...) -> PerformanceSummaryResult

Summarise one input of [`performance_summary`](@ref): check the keywords, make the entries of the input, and summarise them.

# Arguments

  - `kwargs`: The keywords of the call.
  - `args...`: The positional arguments of the call.

# Returns

  - `summary::PerformanceSummaryResult`: The summary.

# Related

  - [`summary_options`](@ref)
  - [`input_entries`](@ref)
  - [`summarise_entries`](@ref)
"""
function summarise_input(kwargs, args...)
    opts = summary_options(; kwargs...)
    return summarise_entries(input_entries(summary_benchmark(kwargs), args...), opts)
end
"""
    candidate_summary(kwargs, names, xs::AbstractVector, args...) -> PerformanceSummaryResult

Summarise a vector of candidates of [`performance_summary`](@ref), one candidate after the other.

# Algorithm

 1. Check that `xs` is not empty, and check the keywords with [`summary_options`](@ref).
 2. Resolve one name for each candidate with [`forecast_summary_names`](@ref), with the default `"Portfolio k"`.
 3. Make the entries of each candidate with [`input_entries`](@ref), each with the name of its candidate, and join them in the order of `xs`.
 4. Summarise the entries with [`summarise_entries`](@ref), which gives one vector entry for each.

# Arguments

  - `kwargs`: The keywords of the call.
  - `names`: One name for each candidate, or `nothing`.
  - `xs`: The candidates.
  - `args...`: The arguments that each candidate shares, such as the `rd` of a vector of [`OptimisationResult`](@ref).

# Validation

  - `!isempty(xs)`. Otherwise the function raises an [`IsEmptyError`](@ref).
  - The rules of [`forecast_summary_names`](@ref).

# Returns

  - `summary::PerformanceSummaryResult`: The summary, with one vector entry for each series.

# Related

  - [`performance_summary`](@ref)
  - [`input_entries`](@ref)
  - [`entry_list`](@ref)
"""
function candidate_summary(kwargs, names, xs::AbstractVector, args...)
    @argcheck(!isempty(xs),
              IsEmptyError("the summary needs at least one candidate, and the vector is empty"))
    opts = summary_options(; kwargs...)
    bm = summary_benchmark(kwargs)
    # A 1-based copy pairs each candidate with its name by one index, with no `enumerate`.
    cs = collect(xs)
    nm = forecast_summary_names(names, length(cs), "Portfolio")
    es = reduce(vcat,
                [entry_list(input_entries(bm, cs[k], args...), nm[k])
                 for k in eachindex(cs)])
    return summarise_entries(es, opts)
end
function performance_summary(w::Union{ArrNum, VecVecNum}, X::MatNum,
                             fees::Option{<:Fees} = nothing;
                             kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, w, X, fees)
end
function performance_summary(w::Union{ArrNum, VecVecNum}, rd::ReturnsResult,
                             fees::Option{<:Fees} = nothing;
                             kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, w, rd, fees)
end
function performance_summary(res::OptimisationResult, rd::ReturnsResult;
                             kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, res, rd)
end
function performance_summary(pred::PredRes_MultiPredRes;
                             kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, pred)
end
function performance_summary(ppred::PopulationPredictionResult;
                             kwargs...)::PerformanceSummaryResult
    return summarise_input(kwargs, ppred)
end
function performance_summary(rets::AbstractVector{<:VecNum}; names = nothing,
                             kwargs...)::PerformanceSummaryResult
    return candidate_summary(kwargs, names, rets)
end
function performance_summary(ress::AbstractVector{<:OptimisationResult}, rd::ReturnsResult;
                             names = nothing, kwargs...)::PerformanceSummaryResult
    return candidate_summary(kwargs, names, ress, rd)
end
function performance_summary(preds::AbstractVector{<:AbstractPredictionResult};
                             names = nothing, kwargs...)::PerformanceSummaryResult
    return candidate_summary(kwargs, names, preds)
end
"""
    DataFrames.DataFrame(ps::PerformanceSummaryResult) -> DataFrame

Write a [`PerformanceSummaryResult`](@ref) as a table, with one row for each entry.

A summary of one series gives one row. The columns are the columns of [`performance_summary_columns`](@ref), in that order: the three keys `:name`, `:path` and `:member`, then `:date0`, `:date1` and `:n_periods`, then the twelve released figures in the order of the struct, then `:n_eff`. One column for each extra measure follows, named by its key, in the order of `measures`. `ppy`, `alpha`, `compound` and `rf` are one value for the whole summary, so they are table-level metadata, and `DataFrames.metadata(df, "rf")` reads one.

# Algorithm

 1. Read the shape of the summary from `n_periods`: one integer for one entry, a vector otherwise.
 2. Write each column of [`performance_summary_columns`](@ref) from its field, as a vector of one entry for a summary of one series.
 3. Write the column of each extra measure from `risk`.
 4. Write `ppy`, `alpha`, `compound` and `rf` as metadata of the table.

# Arguments

  - `ps`: The summary.

# Returns

  - `df::DataFrame`: The table, one row for each entry.

# Examples

```jldoctest
julia> using DataFrames

julia> ps = performance_summary([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01]]; names = [\"a\", \"b\"],
                                measures = [:mdd => MaximumDrawdown()]);

julia> df = DataFrame(ps);

julia> (size(df), names(df)[1:3], names(df)[end])
((2, 20), ["name", "path", "member"], "mdd")

julia> metadata(df, \"ppy\")
1
```

# Related

  - [`performance_summary`](@ref)
  - [`PerformanceSummaryResult`](@ref)
  - [`performance_summary_columns`](@ref)
"""
function DataFrames.DataFrame(ps::PerformanceSummaryResult)
    one_entry = isa(ps.n_periods, Integer)
    column = x -> one_entry ? [x] : collect(x)
    df = DataFrames.DataFrame()
    for (c, f) in pairs(performance_summary_columns())
        df[!, c] = column(getfield(ps, f))
    end
    for (k, v) in pairs(ps.risk)
        df[!, k] = column(v)
    end
    for k in (:ppy, :alpha, :compound, :rf)
        DataFrames.metadata!(df, String(k), getfield(ps, k); style = :note)
    end
    return df
end

export PerformanceSummaryResult, performance_summary
