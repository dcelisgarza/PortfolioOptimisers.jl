"""
$(DocStringExtensions.TYPEDEF)

The headline statistics of one or more covariance forecast evaluations, one entry per evaluation.

`CovarianceForecastSummaryResult` is what [`covariance_forecast_summary`](@ref) returns. Each column has one entry per evaluation, so one evaluation is the length-1 case and two forecasts side by side are the length-2 case. So, like [`ForecastSummaryResult`](@ref) for the return forecasts, it needs no separate comparison type. The summary reads every column from the per-step diagnostics of a [`CovarianceForecastEvaluationResult`](@ref). The bands are Gaussian references, and the summary does not correct them for heavy tails.

# Mathematical definition

```math
\\begin{align}
\\bar{m} &= \\frac{\\sum_{t=1}^{M} \\nu_t\\, m_t}{\\sum_{t=1}^{M} \\nu_t}\\,, \\quad \\nu_t = \\operatorname{dof}(N_t, h_t)\\,, \\\\
\\bar{d} &= \\frac{\\sum_{t=1}^{M} \\nu'_t\\, \\bar{d}_t}{\\sum_{t=1}^{M} \\nu'_t}\\,, \\quad \\nu'_t = \\operatorname{dof}'(h_t)\\,, \\\\
\\bar{m} &\\in 1 \\pm z_{\\alpha/2} \\sqrt{\\frac{2}{\\sum_{t=1}^{M} \\nu_t}}\\,, \\quad \\bar{d} \\in 1 \\pm z_{\\alpha/2} \\sqrt{\\frac{2}{\\sum_{t=1}^{M} \\nu'_t}}\\,, \\\\
B &= \\sqrt{\\frac{1}{M - 1} \\sum_{t=1}^{M} \\left(b_t - \\bar{b}\\right)^2}\\,, \\\\
e_q &= \\frac{1}{M} \\sum_{t=1}^{M} \\mathbb{1}\\left[\\nu_t\\, m_t > \\chi^2_{\\nu_t}(q)\\right]\\,.
\\end{align}
```

Where:

  - ``\\bar{m}``: Mahalanobis ratio over the walk-forward.
  - ``m_t``: Mahalanobis ratio of step ``t``.
  - ``\\nu_t``: Degrees of freedom of the Mahalanobis statistic of step ``t`` under a Gaussian null, ``N_t h_t`` for the realised covariance and ``N_t`` for the horizon return.
  - ``N_t``: Number of active assets at step ``t``.
  - $(math_dict[:h_step])
  - $(math_dict[:M_steps])
  - ``\\bar{d}``: Diagonal ratio over the walk-forward.
  - ``\\bar{d}_t``: Mean of the diagonal ratio over the active assets of step ``t``.
  - ``\\nu'_t``: Degrees of freedom of the ratio of one asset at step ``t`` under a Gaussian null, ``h_t`` for the realised covariance and ``1`` for the horizon return.
  - ``z_{\\alpha/2}``: Upper ``\\alpha / 2`` quantile of the standard normal distribution.
  - ``B``: Bias statistic of a test portfolio, the sample standard deviation of its standardised returns.
  - ``b_t``: Standardised return of the test portfolio at step ``t``.
  - ``\\bar{b}``: Mean of the standardised returns of the test portfolio.
  - ``e_q``: Exceedance rate at level ``q``, the share of steps whose Mahalanobis statistic exceeds the chi-squared quantile of that level.
  - ``\\chi^2_{\\nu}(q)``: Quantile at level ``q`` of the chi-squared distribution with ``\\nu`` degrees of freedom.

Under a Gaussian null, with the forecast correct and the whitened returns independent, ``\\bar{m} \\sum_t \\nu_t \\sim \\chi^2_{\\sum_t \\nu_t}``. So ``\\mathbb{E}[\\bar{m}] = 1``, and the band holds with probability ``1 - \\alpha``. The ratio of sums weights a step by its degrees of freedom, and it is the plain mean when every step has the same ``N_t`` and ``h_t``. The band on ``\\bar{d}`` is the band of the ratio of one asset. A mean over assets has at most the variance of the ratio of one asset, so the band is conservative, and it is exact only when the ratios of the assets are perfectly correlated.

Let ``\\kappa`` be the fourth moment of one whitened coordinate of the realised quantity, with the Gaussian value three. Under the realised covariance the coordinate is that of one return, and under the horizon return it is that of the horizon return divided by ``\\sqrt{h_t}``. With independent coordinates, the variance of ``\\bar{m}`` is ``(\\kappa - 1) / \\sum_t \\nu_t``, so the band widens by ``\\sqrt{(\\kappa - 1) / 2}``. A horizon return sums ``h_t`` returns, so for independent and identically distributed returns its excess of ``\\kappa`` over three is that of one return divided by ``h_t``.

The median of ``\\chi^2_{\\nu} / \\nu`` lies below one, near ``(1 - 2 / (9 \\nu))^3 \\approx 1 - 2 / (3 \\nu)``, so compare a median with that value and not with one. ``B = 1`` under a calibrated forecast, and ``B > 1`` when the forecast under-predicts the risk of the portfolio. ``B`` needs two steps. ``e_q`` is ``1 - q`` under the Gaussian null, and it rises with heavy tails as well as with a misspecified forecast.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastSummaryResult(
        names, mahalanobis_mean, mahalanobis_median, mahalanobis_p5, mahalanobis_p95,
        mahalanobis_band_lo, mahalanobis_band_hi, diagonal_mean, diagonal_median,
        diagonal_p5, diagonal_p95, diagonal_band_lo, diagonal_band_hi, bias_statistic,
        bias_p5, bias_p25, bias_p75, bias_p95, qlike_mean, frobenius_mean,
        portfolio_qlike_mean, levels, exceedance, n_steps, n_portfolios, alpha
    ) -> CovarianceForecastSummaryResult

Arguments correspond to the struct's fields, in the order they are declared. [`covariance_forecast_summary`](@ref) builds the type, and a caller reads it. The type has no keyword constructor, and it validates nothing.

# Related

  - [`covariance_forecast_summary`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`ForecastSummaryResult`](@ref)
"""
@concrete struct CovarianceForecastSummaryResult <: AbstractResult
    """
    Name of each evaluation, one entry per evaluation.
    """
    names
    """
    Mahalanobis ratio over the walk-forward, the mean of the per-step ratios weighted by their degrees of freedom. One entry per evaluation. The target is one.
    """
    mahalanobis_mean
    """
    Median of the per-step Mahalanobis ratio. One entry per evaluation. Under a Gaussian null its target lies below one.
    """
    mahalanobis_median
    """
    Fifth percentile of the per-step Mahalanobis ratio. One entry per evaluation.
    """
    mahalanobis_p5
    """
    Ninety-fifth percentile of the per-step Mahalanobis ratio. One entry per evaluation.
    """
    mahalanobis_p95
    """
    Lower end of the Gaussian band on the Mahalanobis ratio at level `alpha`. One entry per evaluation.
    """
    mahalanobis_band_lo
    """
    Upper end of the Gaussian band on the Mahalanobis ratio at level `alpha`. One entry per evaluation.
    """
    mahalanobis_band_hi
    """
    Diagonal ratio over the walk-forward, the mean over the steps of the per-step mean over the active assets, weighted by the degrees of freedom of the steps. One entry per evaluation. The target is one.
    """
    diagonal_mean
    """
    Median of the per-step diagonal ratio, which is the mean over the active assets of the step. One entry per evaluation.
    """
    diagonal_median
    """
    Fifth percentile of the per-step diagonal ratio. One entry per evaluation.
    """
    diagonal_p5
    """
    Ninety-fifth percentile of the per-step diagonal ratio. One entry per evaluation.
    """
    diagonal_p95
    """
    Lower end of the Gaussian band on the diagonal ratio of one asset at level `alpha`. One entry per evaluation.
    """
    diagonal_band_lo
    """
    Upper end of the Gaussian band on the diagonal ratio of one asset at level `alpha`. One entry per evaluation.
    """
    diagonal_band_hi
    """
    Bias statistic, the median over the test portfolios of the sample standard deviation of the standardised returns of each portfolio. One entry per evaluation. The target is one. It is `NaN` for an evaluation with one step, and for an evaluation with a `NaN` standardised return.
    """
    bias_statistic
    """
    Fifth percentile of the bias statistic over the test portfolios. One entry per evaluation, `NaN` where `bias_statistic` is.
    """
    bias_p5
    """
    Twenty-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation, `NaN` where `bias_statistic` is.
    """
    bias_p25
    """
    Seventy-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation, `NaN` where `bias_statistic` is.
    """
    bias_p75
    """
    Ninety-fifth percentile of the bias statistic over the test portfolios. One entry per evaluation, `NaN` where `bias_statistic` is.
    """
    bias_p95
    """
    Mean QLIKE loss over the steps. One entry per evaluation. Only a difference between two evaluations says which forecast is better.
    """
    qlike_mean
    """
    Mean Frobenius loss over the steps. One entry per evaluation. Only a difference between two evaluations says which forecast is better.
    """
    frobenius_mean
    """
    Median over the test portfolios of the mean portfolio QLIKE loss of each portfolio. One entry per evaluation.
    """
    portfolio_qlike_mean
    """
    Confidence levels of the exceedance rates, as the caller gave them.
    """
    levels
    """
    Exceedance rate of the per-step Mahalanobis statistic against the chi-squared quantile of each level, `evaluations × levels`. The target is one less the level.
    """
    exceedance
    """
    Number of steps of each evaluation. One entry per evaluation.
    """
    n_steps
    """
    Number of test portfolios of each evaluation. One entry per evaluation.
    """
    n_portfolios
    """
    Level of the Gaussian bands.
    """
    alpha
end
"""
    covariance_forecast_summary(cfers::AbstractVector{<:CovarianceForecastEvaluationResult};
                                names = nothing, alpha::Real = 0.05,
                                levels = (0.95, 0.99)) -> CovarianceForecastSummaryResult
    covariance_forecast_summary(cfer::CovarianceForecastEvaluationResult; kwargs...)

Summarise one or more covariance forecast evaluations, one entry per evaluation.

For each evaluation, the summary gives the mean, the median and the tail percentiles of the two calibration ratios, the Gaussian band on each mean, the bias statistic of the test portfolios with its percentiles over the portfolios, the mean of each loss, and the exceedance rate of the Mahalanobis statistic at each level. The method for one Result is the length-1 case.

The mean of a ratio is a ratio of sums, which weights each step by its degrees of freedom under the target of the evaluation ([`target_dof`](@ref), [`target_step_dof`](@ref)). A date walk-forward whose folds differ in length weights each step by its length. A listing or a delisting also changes the weight of a step of the Mahalanobis ratio, because its degrees of freedom count the active assets. So the Mahalanobis mean is the plain mean only when every step has the same horizon and the same active count, and the diagonal mean is the plain mean when every step has the same horizon.

# Algorithm

 1. For each evaluation, find the degrees of freedom of each step for the Mahalanobis ratio through [`target_dof`](@ref), and for the ratio of one asset through [`target_step_dof`](@ref).
 2. Reduce the diagonal ratio of each step to its mean over the active assets.
 3. For each ratio, take the weighted mean, the median, and the fifth and ninety-fifth percentiles through [`summary_quantile`](@ref).
 4. Find the half-width of the Gaussian band of each mean at `alpha`, from the sum of the degrees of freedom.
 5. Take the sample standard deviation of the standardised returns of each test portfolio. Take the median and the percentiles of these over the portfolios, through [`summary_quantile`](@ref).
 6. Take the mean of each loss over the steps, and the median over the portfolios of the mean portfolio QLIKE.
 7. For each level, find the share of the steps whose Mahalanobis statistic exceeds the chi-squared quantile of that level.

# Arguments

  - `cfers`: The evaluations to summarise.
  - `names`: A name per evaluation, or `nothing` for `"forecast_1"`, `"forecast_2"`, ….
  - `alpha`: Level of the Gaussian bands, `1 - alpha` coverage.
  - `levels`: Confidence levels of the exceedance rates.

# Validation

  - `cfers` is not empty. An `IsEmptyError` is thrown otherwise.
  - `names` has one entry per evaluation when given. A `DimensionMismatch` is thrown otherwise.
  - `0 < alpha < 1`, and every level lies in `(0, 1)`. A `DomainError` is thrown otherwise.

# Returns

  - `summary::CovarianceForecastSummaryResult`: The columnar summary. The bias columns of an evaluation with one step are `NaN`, because a sample standard deviation needs two steps.

# Related

  - [`CovarianceForecastSummaryResult`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`target_dof`](@ref)
  - [`target_step_dof`](@ref)
"""
function covariance_forecast_summary(cfers::AbstractVector{<:CovarianceForecastEvaluationResult};
                                     names::Option{<:AbstractVector} = nothing,
                                     alpha::Real = 0.05, levels = (0.95, 0.99))
    @argcheck(!isempty(cfers), IsEmptyError("`cfers` cannot be empty"))
    @argcheck(zero(alpha) < alpha < one(alpha),
              DomainError(alpha, "`alpha` must lie in (0, 1)"))
    @argcheck(all(q -> zero(q) < q < one(q), levels),
              DomainError(levels, "every level must lie in (0, 1)"))
    n = length(cfers)
    nms = isnothing(names) ? ["forecast_$(i)" for i in 1:n] : names
    @argcheck(length(nms) == n,
              DimensionMismatch("`names` has $(length(nms)) entries and `cfers` $(n) evaluations."))
    z = Distributions.cquantile(Distributions.Normal(), alpha / 2)
    Tf = promote_type(eltype(cfers[1].mahalanobis_ratio), typeof(z))
    mm = Vector{Tf}(undef, n)
    mmed = similar(mm)
    m5 = similar(mm)
    m95 = similar(mm)
    mlo = similar(mm)
    mhi = similar(mm)
    dm = similar(mm)
    dmed = similar(mm)
    d5 = similar(mm)
    d95 = similar(mm)
    dlo = similar(mm)
    dhi = similar(mm)
    bs = similar(mm)
    b5 = similar(mm)
    b25 = similar(mm)
    b75 = similar(mm)
    b95 = similar(mm)
    ql = similar(mm)
    fr = similar(mm)
    pql = similar(mm)
    ex = Matrix{Tf}(undef, n, length(levels))
    ns = Vector{Int}(undef, n)
    np = Vector{Int}(undef, n)
    dbar = Vector{Tf}(undef, 0)
    for (i, cfer) in enumerate(cfers)
        m = cfer.mahalanobis_ratio
        dof = target_dof.(Ref(cfer.target), cfer.n_valid, cfer.horizon)
        sdof = target_step_dof.(Ref(cfer.target), cfer.horizon)
        map!(t -> Statistics.mean(filter(isfinite, view(cfer.diagonal_ratio, t, :))),
             resize!(dbar, size(cfer.diagonal_ratio, 1)), axes(cfer.diagonal_ratio, 1))
        mm[i] = LinearAlgebra.dot(dof, m) / sum(dof)
        mmed[i] = Statistics.median(m)
        m5[i] = summary_quantile(m, 0.05)
        m95[i] = summary_quantile(m, 0.95)
        hw = z * sqrt(2 / sum(dof))
        mlo[i] = 1 - hw
        mhi[i] = 1 + hw
        dm[i] = LinearAlgebra.dot(sdof, dbar) / sum(sdof)
        dmed[i] = Statistics.median(dbar)
        d5[i] = summary_quantile(dbar, 0.05)
        d95[i] = summary_quantile(dbar, 0.95)
        hwd = z * sqrt(2 / sum(sdof))
        dlo[i] = 1 - hwd
        dhi[i] = 1 + hwd
        b = vec(Statistics.std(cfer.standardised_return; dims = 1))
        bs[i] = Statistics.median(b)
        b5[i] = summary_quantile(b, 0.05)
        b25[i] = summary_quantile(b, 0.25)
        b75[i] = summary_quantile(b, 0.75)
        b95[i] = summary_quantile(b, 0.95)
        ql[i] = Statistics.mean(cfer.qlike)
        fr[i] = Statistics.mean(cfer.frobenius)
        pql[i] = Statistics.median(vec(Statistics.mean(cfer.portfolio_qlike; dims = 1)))
        stat = dof .* m
        for (j, q) in enumerate(levels)
            thr = Distributions.quantile.(Distributions.Chisq.(dof), q)
            ex[i, j] = Statistics.mean(k -> stat[k] > thr[k], eachindex(stat, thr))
        end
        ns[i] = length(m)
        np[i] = size(cfer.standardised_return, 2)
    end
    return CovarianceForecastSummaryResult(nms, mm, mmed, m5, m95, mlo, mhi, dm, dmed, d5,
                                           d95, dlo, dhi, bs, b5, b25, b75, b95, ql, fr,
                                           pql, levels, ex, ns, np, alpha)
end
function covariance_forecast_summary(cfer::CovarianceForecastEvaluationResult; kwargs...)
    return covariance_forecast_summary([cfer]; kwargs...)
end
"""
    summary_quantile(x::VecNum, p::Number)

Return the `p` quantile of a column of the summary, or `NaN` when the column holds a `NaN`.

`Statistics.quantile` throws on a `NaN`, and `Statistics.median` returns `NaN`. The summary reads a `NaN` as an undefined entry, so a percentile of a column that holds one is `NaN`, as the median of that column is. The bias statistic of a one-step evaluation is such a column, because the sample standard deviation of one step is `NaN`.

# Arguments

  - `x`: The column.
  - `p`: The probability of the quantile.

# Returns

  - `q::Number`: The quantile of `x` at `p`, or `NaN`.

# Related

  - [`covariance_forecast_summary`](@ref)
"""
function summary_quantile(x::VecNum, p::Number)
    # The `NaN` answered is an element of `x`, so it carries the element type of the data
    # and no type is chosen here.
    i = findfirst(isnan, x)
    return isnothing(i) ? Statistics.quantile(x, p) : x[i]
end
"""
$(DocStringExtensions.TYPEDEF)

The Diebold–Mariano–West comparison of two covariance forecasts, one row per loss.

`CovarianceForecastComparisonResult` is what [`covariance_forecast_compare`](@ref) returns. Each column has one entry per loss compared. The losses are the whole-matrix QLIKE, the Frobenius loss, and the portfolio QLIKE of each test portfolio. A positive mean difference says that the first forecast lost more.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastComparisonResult(
        names, mean_difference, variance, z, p, lags, n_steps
    ) -> CovarianceForecastComparisonResult

Arguments correspond to the struct's fields, in the order they are declared. [`covariance_forecast_compare`](@ref) builds the type, and a caller reads it. The type has no keyword constructor, and it validates nothing.

# Related

  - [`covariance_forecast_compare`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
"""
@concrete struct CovarianceForecastComparisonResult <: AbstractResult
    """
    Name of each loss compared: `"qlike"`, `"frobenius"`, then `"portfolio_qlike_k"` for each test portfolio `k`.
    """
    names
    """
    Mean over the steps of the loss of the first forecast less that of the second. One entry per loss.
    """
    mean_difference
    """
    Newey–West long-run variance of the per-step loss difference, at `lags` lags. One entry per loss.
    """
    variance
    """
    Diebold–Mariano–West statistic, asymptotically standard normal under equal expected loss. One entry per loss.
    """
    z
    """
    Two-sided p-value of the statistic against the standard normal. One entry per loss.
    """
    p
    """
    Number of lags of the Bartlett kernel.
    """
    lags
    """
    Number of steps compared.
    """
    n_steps
end
"""
    covariance_forecast_compare(a::CovarianceForecastEvaluationResult, b::CovarianceForecastEvaluationResult;
                                lags::Integer = min(maximum(a.horizon), length(a.qlike)) - 1)
        -> CovarianceForecastComparisonResult

Test whether two covariance forecasts differ in expected loss, one test per loss.

The level of a loss on a proxy is not a calibration reading, so two mean losses side by side do not say which forecast is better. The comparison tests the difference of the losses instead. For each loss the per-step difference is a series, and the test compares its mean with zero through a long-run variance. The test windows of a walk-forward do not overlap, so no two steps share a test row. But the forecasts of two steps can share training rows, so their losses can be correlated, and the lags absorb a serial correlation of the difference up to their number.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= L^{A}_t - L^{B}_t\\,, \\\\
\\bar{\\delta} &= \\frac{1}{M} \\sum_{t=1}^{M} \\delta_t\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,, \\\\
\\mathrm{DMW} &= \\frac{\\sqrt{M}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,, \\\\
p &= 2 \\left(1 - \\Phi\\left(\\lvert \\mathrm{DMW} \\rvert\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``\\delta_t``: Difference of the losses of forecasts ``A`` and ``B`` at step ``t``.
  - ``L^{A}_t``, ``L^{B}_t``: Loss of forecast ``A`` and of forecast ``B`` at step ``t``.
  - ``\\bar{\\delta}``: Mean loss difference over the walk-forward.
  - ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with ``\\ell`` lags ([`newey_west_variance`](@ref)).
  - ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``.
  - ``\\ell``: Number of lags.
  - $(math_dict[:M_steps])
  - ``\\mathrm{DMW}``: Diebold–Mariano–West statistic.
  - ``p``: Two-sided p-value of the statistic.
  - ``\\Phi``: Distribution function of the standard normal.

Under equal expected loss, ``\\mathrm{DMW}`` is asymptotically standard normal. A positive value says that forecast ``A`` lost more than forecast ``B``. Two identical forecasts have ``\\bar{\\delta} = 0`` and ``\\hat{\\omega}^2 = 0``, so their statistic is not a number.

# Arguments

  - `a`: The first evaluation.
  - `b`: The second evaluation.
  - `lags`: Number of lags of the Bartlett kernel. The default is the largest horizon less one, the rule of Diebold and Mariano for a forecast over ``h`` steps, capped at ``M - 1``.

# Validation

  - `a.dates == b.dates` and `a.horizon == b.horizon`. An `ArgumentError` is thrown otherwise, because a per-step difference needs the same steps on both sides.
  - `a` and `b` carry the same number of test portfolios. A `DimensionMismatch` is thrown otherwise.
  - `0 <= lags < n_steps`. A `DomainError` is thrown otherwise.

# Returns

  - `cmp::CovarianceForecastComparisonResult`: One row per loss.

# Related

  - [`CovarianceForecastComparisonResult`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_summary`](@ref)
  - [`newey_west_variance`](@ref)

# References

  - $(ref_dict[:dieboldmariano1995])
  - $(ref_dict[:west1996])
"""
function covariance_forecast_compare(a::CovarianceForecastEvaluationResult,
                                     b::CovarianceForecastEvaluationResult;
                                     lags::Integer = min(maximum(a.horizon),
                                                         length(a.qlike)) - 1)
    @argcheck(isequal(a.dates, b.dates) && a.horizon == b.horizon,
              ArgumentError("the two evaluations do not share their steps: a comparison reads a per-step loss difference, so both must be run over one walk-forward, with the same dates and the same horizon at every step. Run both forecasts through `covariance_forecast_evaluation` with the same `rd` and `cv`."))
    P = size(a.portfolio_qlike, 2)
    @argcheck(size(b.portfolio_qlike, 2) == P,
              DimensionMismatch("the first evaluation carries $(P) test portfolio(s) and the second $(size(b.portfolio_qlike, 2)); hand both the same `w`."))
    M = length(a.qlike)
    @argcheck(0 <= lags < M, DomainError(lags, "`lags` must lie in [0, n_steps)"))
    names = vcat(["qlike", "frobenius"], ["portfolio_qlike_$(k)" for k in 1:P])
    series = vcat([a.qlike .- b.qlike, a.frobenius .- b.frobenius],
                  [view(a.portfolio_qlike, :, k) .- view(b.portfolio_qlike, :, k)
                   for k in 1:P])
    md = [Statistics.mean(s) for s in series]
    v = [newey_west_variance(s, lags) for s in series]
    z = sqrt(M) .* md ./ sqrt.(v)
    p = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    return CovarianceForecastComparisonResult(names, md, v, z, p, lags, M)
end
"""
    newey_west_variance(d::VecNum, lags::Integer)

Return the Bartlett-kernel long-run variance of a series, at `lags` lags.

# Mathematical definition

```math
\\begin{align}
\\hat{\\gamma}_k &= \\frac{1}{M} \\sum_{t=k+1}^{M} \\left(d_t - \\bar{d}\\right) \\left(d_{t-k} - \\bar{d}\\right)\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,.
\\end{align}
```

Where:

  - ``d_t``: Entry ``t`` of the series.
  - ``\\bar{d}``: Mean of the series.
  - ``M``: Length of the series.
  - ``\\hat{\\gamma}_k``: Sample autocovariance of the series at lag ``k``.
  - ``\\ell``: Number of lags.
  - ``\\hat{\\omega}^2``: Long-run variance of the series.

The Bartlett weights make ``\\hat{\\omega}^2`` non-negative for every series and every ``\\ell``. At ``\\ell = 0`` it is the sample variance with the divisor ``M``.

# Arguments

  - `d`: The series.
  - `lags`: Number of lags, ``\\ell``.

# Returns

  - `omega2::Number`: The long-run variance.

# Related

  - [`covariance_forecast_compare`](@ref)

# References

  - $(ref_dict[:neweywest1987])
"""
function newey_west_variance(d::VecNum, lags::Integer)
    M = length(d)
    e = d .- Statistics.mean(d)
    gamma(k) = LinearAlgebra.dot(view(e, (k + 1):M), view(e, 1:(M - k))) / M
    omega2 = gamma(0)
    for k in 1:lags
        omega2 += 2 * (1 - k / (lags + 1)) * gamma(k)
    end
    return omega2
end
"""
    covariance_forecast_portfolio(cfer::CovarianceForecastEvaluationResult, rd::ReturnsResult,
                                  w::Option{<:VecNum_VecVecNum}) -> NamedTuple

Score the stored forecasts of an evaluation again, on a new test portfolio.

With `store_forecasts = true` the Result holds the forecast of every step and the location that the forecast was centred on, so this verb scores a new portfolio without a new run of the loop. The own `w` of the evaluation reproduces its columns.

# Algorithm

 1. For each step, read the stored forecast and location, and the test rows `rd.X[cfer.test_idx[i], :]`.
 2. Score them on `w` with the target of the evaluation, through [`covariance_forecast_step`](@ref).
 3. Put the standardised returns and the portfolio QLIKE losses of the step in its row of each matrix.

# Arguments

  - `cfer`: The evaluation, run with `store_forecasts = true`.
  - $(arg_dict[:rd]) It must be the carrier that the evaluation ran on, because each step reads its rows through `cfer.test_idx`.
  - `w`: The test portfolios on the full universe, as [`covariance_forecast_evaluation`](@ref) takes them.

# Validation

  - `cfer.sigma` is not `nothing`. An `ArgumentError` is thrown otherwise, which names `store_forecasts`.
  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.

# Returns

  - `proj::NamedTuple`: `standardised_return` and `portfolio_qlike`, each `steps × portfolios`.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_step`](@ref)
"""
function covariance_forecast_portfolio(cfer::CovarianceForecastEvaluationResult,
                                       rd::ReturnsResult, w::Option{<:VecNum_VecVecNum})
    @argcheck(!isnothing(cfer.sigma),
              ArgumentError("the evaluation holds no forecasts to re-project: run `covariance_forecast_evaluation` with `store_forecasts = true`, which keeps every step's forecast and the location it was centred on."))
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    steps = [covariance_forecast_step(cfer.sigma[i], view(rd.X, cfer.test_idx[i], :),
                                      cfer.location[i], w, cfer.target)
             for i in eachindex(cfer.sigma)]
    P = length(steps[1].standardised_return)
    b = Matrix{eltype(steps[1].standardised_return)}(undef, length(steps), P)
    pq = Matrix{eltype(steps[1].portfolio_qlike)}(undef, length(steps), P)
    for (i, s) in enumerate(steps)
        b[i, :] .= s.standardised_return
        pq[i, :] .= s.portfolio_qlike
    end
    return (; standardised_return = b, portfolio_qlike = pq)
end

export covariance_forecast_summary, CovarianceForecastSummaryResult,
       covariance_forecast_compare, CovarianceForecastComparisonResult,
       covariance_forecast_portfolio
