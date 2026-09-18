"""
$(DocStringExtensions.TYPEDEF)

The log-wealth regret of a strategy against a comparator over one sequence of rows, with the test of its per-row difference.

`LogWealthRegretResult` is what [`log_wealth_regret`](@ref) returns. The regret is the gap in log terminal wealth between the comparator and the strategy, positive when the comparator wins; the per-row difference series carries a Newey–West test of equal expected log growth, in the shape of [`CovarianceForecastComparisonResult`](@ref). Negative regret is expected on many sequences and is not a defect: a causal strategy that reads the market's structure can beat a constant portfolio, and a Hindsight Comparator is a ceiling only over the class it was chosen from.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogWealthRegretResult(
        regret, regret_per_period, difference, variance, z, p, lags, n_periods,
        wealth_a, wealth_b
    ) -> LogWealthRegretResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`log_wealth_regret`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`log_wealth_regret`](@ref)
  - [`CovarianceForecastComparisonResult`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
"""
@concrete struct LogWealthRegretResult <: AbstractResult
    """
    Log-wealth regret, ``\\sum_t \\log(1 + r_{b,t}) - \\sum_t \\log(1 + r_{a,t})``: the comparator's log terminal wealth less the strategy's, positive when the comparator wins.
    """
    regret
    """
    The regret divided by the number of periods, which is the mean of `difference`.
    """
    regret_per_period
    """
    Per-row series ``\\log(1 + r_{b,t}) - \\log(1 + r_{a,t})``, the comparator's log growth less the strategy's at every period.
    """
    difference
    """
    Newey–West long-run variance of `difference`, at `lags` lags.
    """
    variance
    """
    Diebold–Mariano–West statistic of `difference`, asymptotically standard normal under equal expected log growth when the comparator did not read the rows.
    """
    z
    """
    Two-sided p-value of the statistic against the standard normal. It is `NaN` when the two series coincide, because their difference has no variance.
    """
    p
    """
    Number of lags of the Bartlett kernel.
    """
    lags
    """
    Number of periods compared.
    """
    n_periods
    """
    Terminal wealth of the strategy from a unit start, ``\\prod_t (1 + r_{a,t})``.
    """
    wealth_a
    """
    Terminal wealth of the comparator from a unit start, ``\\prod_t (1 + r_{b,t})``.
    """
    wealth_b
end
"""
    regret_series(pred::PredictionResult)
    regret_series(pred::MultiPeriodPredictionResult)

The realised return series and the timestamps a prediction result was scored on, as [`log_wealth_regret`](@ref) reads them.

The series is the one [`performance_summary`](@ref) reads: the portfolio returns [`predict`](@ref) stored, net of the fee the fold settled, drifted where a Weight Drift ran; a population of paths reads its first path. A multi-period result reads its stacked returns and timestamps.

# Arguments

  - `pred`: The prediction result.

# Returns

  - `ret::VecNum`: The realised return series.
  - `ts::Option{<:AbstractVector}`: The timestamps of the rows, or `nothing` when the fold carried none.

# Related

  - [`log_wealth_regret`](@ref)
  - [`performance_summary`](@ref)
"""
function regret_series(pred::PredictionResult)
    rd = pred.rd
    return isa(rd.X, VecVecNum) ? first(rd.X) : rd.X, rd.ts
end
function regret_series(pred::MultiPeriodPredictionResult)
    mrd = pred.mrd
    return isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X, mrd.ts
end
"""
    log_wealth_regret(a::PredRes_MultiPredRes, b::PredRes_MultiPredRes; lags::Integer = 0) -> LogWealthRegretResult

The log-wealth regret of strategy `a` against comparator `b` over one and the same sequence of rows.

Regret is defined over one sequence, so the verb refuses unless the two prediction results carry the same timestamps. Each series is read exactly as it was scored — net of the fee the fold settled, drifted where a Weight Drift ran — so the comparator's fee policy is the caller's, and a fee-free comparator against a fee-paying strategy measures the fee as regret. The comparator is any prediction result over the rows: the same estimator run causally through `cross_val_predict` with the strategy's `cv`, or a Hindsight Comparator, an estimator fit on the rows it is scored on and predicted in sample, `predict(optimise(est, rd_test), rd_test)`. The best constant rebalanced portfolio in hindsight is [`BestConstantRebalancedPortfolio`](@ref), or [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) when it is bounded; the best stock in hindsight is a [`ScoreSelector`](@ref) under [`RankRule`](@ref)`(; best = 1)` and [`MeanReturn`](@ref)`(; flag = true)` composed with [`EqualWeighted`](@ref) in a [`Pipeline`](@ref).

# Mathematical definition

```math
\\begin{align}
\\delta_t &= \\log\\left(1 + r_{b,t}\\right) - \\log\\left(1 + r_{a,t}\\right)\\,, \\\\
R &= \\sum_{t=1}^{T} \\delta_t\\,, \\quad \\bar{\\delta} = \\frac{R}{T}\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,, \\\\
z &= \\frac{\\sqrt{T}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,.
\\end{align}
```

Where:

  - ``r_{a,t}``, ``r_{b,t}``: Realised return of the strategy and of the comparator at period ``t``, as scored.
  - ``\\delta_t``: Per-row difference in log growth.
  - ``R``: Log-wealth regret, positive when the comparator wins.
  - ``\\bar{\\delta}``: Regret per period.
  - ``T``: Number of periods.
  - ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with ``\\ell`` lags.
  - ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``.
  - ``\\ell``: Number of lags, ``0`` by default because the rows of a walk-forward at `test_size = 1` do not overlap.
  - ``z``: Diebold–Mariano–West statistic.

Under equal expected log growth, ``z`` is asymptotically standard normal, and the two-sided `p` reads it against that law. The test is exact only for a comparator that did not read the rows it is scored on; against a Hindsight Comparator it is optimistic by construction, because the comparator was chosen on the very sequence the difference is tested over, and the `p` then overstates the evidence that the comparator is better. Two identical series have ``\\bar{\\delta} = 0`` and ``\\hat{\\omega}^2 = 0``, so their statistic and `p` are `NaN`. Negative regret is expected on many sequences and is not a defect.

No scorer and no summary column ship for regret. A hyperparameter search ranks on [`MeanReturn`](@ref)`(; flag = true)`, which is log wealth per period and orders candidates as regret against any fixed comparator would; the performance summary reads one series and holds no comparator.

# Arguments

  - `a`: The strategy's prediction result.
  - `b`: The comparator's prediction result.
  - `lags`: Number of lags of the Bartlett kernel.

# Validation

  - `a` and `b` carry the same timestamps and the same number of rows. An `ArgumentError` is thrown otherwise: two runs over different rows compare nothing.
  - `0 <= lags < n_periods`. A `DomainError` is thrown otherwise.

# Returns

  - `reg::LogWealthRegretResult`: The regret, its per-period form, the per-row difference and its test, and the two terminal wealths.

# Related

  - [`LogWealthRegretResult`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`newey_west_variance`](@ref)
  - [`performance_summary`](@ref)
  - [`cross_val_predict`](@ref)

# References

  - $(ref_dict[:lihoi2014])
"""
function log_wealth_regret(a::PredRes_MultiPredRes, b::PredRes_MultiPredRes;
                           lags::Integer = 0)
    ra, tsa = regret_series(a)
    rb, tsb = regret_series(b)
    @argcheck(isequal(tsa, tsb) && length(ra) == length(rb),
              ArgumentError("the two prediction results do not share their rows: regret is defined over one sequence, so both must be scored over the same rows, with the same timestamps. Run both through `cross_val_predict` with the same `rd` and `cv`, or fit the comparator on the rows the strategy was scored on and predict it over them."))
    T = length(ra)
    @argcheck(0 <= lags < T, DomainError(lags, "`lags` must lie in [0, n_periods)"))
    la = log1p.(ra)
    lb = log1p.(rb)
    d = lb .- la
    regret = sum(lb) - sum(la)
    md = Statistics.mean(d)
    v = newey_west_variance(d, lags)
    z = sqrt(T) * md / sqrt(v)
    p = 2 * Distributions.ccdf(Distributions.Normal(), abs(z))
    return LogWealthRegretResult(regret, md, d, v, z, p, lags, T, exp(sum(la)),
                                 exp(sum(lb)))
end

export log_wealth_regret, LogWealthRegretResult
