"""
    forecast_ic_weights(alpha::MatNum, w::Nothing)
    forecast_ic_weights(alpha::MatNum, w::MatNum)

Return the cross-sectional weight history the Pearson information coefficient reads, after a check against the forecast.

It is the counterpart of [`exposure_weights`](@ref) on the observation-by-asset axis. An absent weight history means equal weights. The function resolves the absent case once, into a history of ones, so the kernel reads one matrix and carries no branch.

A weight of zero excludes the asset from the observation, and so does a weight that is not finite, because [`cs_weighted_correlation`](@ref) reads only a finite positive weight. [`forecast_coverage`](@ref) reads the same rule. The function does not refuse an infinite weight, because [`InverseIdiosyncraticVarianceMetric`](@ref) gives one to an asset whose idiosyncratic variance is zero.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.

# Validation

  - `size(w) == size(alpha)`, when `w` is present. Raises a `DimensionMismatch`.
  - No finite entry of `w` is negative. Raises a `DomainError`.

# Returns

  - `u::MatNum`: Weight history `observations × assets`. It is a history of ones when `w` is `nothing`, and `w` itself otherwise.

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_coverage`](@ref)
  - [`exposure_weights`](@ref)
"""
function forecast_ic_weights(alpha::MatNum, ::Nothing)
    return ones(real(eltype(alpha)), size(alpha, 1), size(alpha, 2))
end
function forecast_ic_weights(alpha::MatNum, w::MatNum)
    @argcheck(size(w, 1) == size(alpha, 1) && size(w, 2) == size(alpha, 2),
              DimensionMismatch("w ($(size(w, 1))×$(size(w, 2))) must match alpha ($(size(alpha, 1))×$(size(alpha, 2)))"))
    @argcheck(all(x -> !isfinite(x) || x >= zero(x), w),
              DomainError(w, "w cannot carry a negative weight"))
    return w
end
"""
    forecast_ic(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing;
                min_count::Integer = fe.min_count) -> Matrix{<:Real}
    forecast_ic(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                min_count::Integer = fe.min_count) -> Matrix{<:Real}

Return the information coefficients of a Return Forecast, one row per evaluation date.

The information coefficient is the cross-sectional correlation between the forecast known at an observation and the forward target it is answerable for. A correlation does not change when the forecast is shifted or scaled by a positive number, so neither coefficient scores the size of the forecast.

The function answers two coefficients, one per column, because each scores a different claim and a caller compares them. The Spearman coefficient correlates the ranks. It reads no weights, and one asset with an extreme forecast moves it no more than one ordinary asset does. The Pearson coefficient correlates the levels under the cross-sectional weights. It falls below the Spearman one when the forecast orders the assets well and spaces them badly.

The Spearman column ranks a tie by the rule in `fe.ties`, as [`cs_ranks`](@ref) states. Under the default `:average`, equal values share the mean of their ranks, so the order of the assets does not move the coefficient. At a date where the forecast gives every asset the same value, both columns are then `NaN`. Under `:ordinal`, equal values take consecutive ranks in the order of the asset axis. That order then sets part of the Spearman coefficient, and a constant forecast can score `1` or `-1`.

# Mathematical definition

```math
\\mathrm{IC}^{\\mathrm{S}}_{j} = \\rho^{\\mathrm{S}} \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\boldsymbol{y}_{t_{j}} \\right)
\\qquad
\\mathrm{IC}^{\\mathrm{P}}_{j} = \\rho \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\boldsymbol{y}_{t_{j}}, \\boldsymbol{u}_{t_{j}} \\right)
```

Where:

  - $(math_dict[:alpha_t_fc])
  - ``\\boldsymbol{y}_{t}``: Cross-section of the forward target at observation ``t``.
  - $(math_dict[:u_t_cs])
  - $(math_dict[:rho_rank_cs])
  - $(math_dict[:rho_w_cs])
  - $(math_dict[:t_j_eval])

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref).
 2. At each evaluation date, correlate the forecast against the target with [`cs_spearman_correlation`](@ref) under `fe.ties` for the first column, and with [`cs_weighted_correlation`](@ref) for the second.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The rank column reads no weights.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the weight history the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history of the Pearson column, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default reads equal weights.
  - `min_count`: Least number of assets a cross-section needs before the function reports a coefficient of it. It defaults to the threshold the evaluation carries. A caller overrides it to read the same pairing at a second threshold.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.
  - The rules of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `ic::Matrix{<:Real}`: `dates × 2`, in the promotion of the element types of `fe.alpha`, `fe.y` and the weight history. Row `j` scores the evaluation date `fe.dates[j]`. The first column carries the Spearman coefficient and the second the Pearson one. The Spearman column is `NaN` at a date where fewer than `min_count` assets, or fewer than two, carry both a finite forecast and a finite target. The Pearson column is `NaN` where fewer than `min_count` of those assets also carry a finite positive weight. It is also `NaN` where the forecast or the target is constant over them, which [`cs_weighted_correlation`](@ref) reads as a denominator of at most `1e-12`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> forecast_ic(forecast_evaluation(alpha, y))
3×2 Matrix{Float64}:
  1.0   0.940484
  0.4   0.100906
 -0.4  -0.271052
```

# Related

  - [`forecast_ic_summary`](@ref)
  - [`forecast_coverage`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`cs_weighted_correlation`](@ref)
  - [`cs_ranks`](@ref)
  - [`exposure_ic`](@ref)
"""
function forecast_ic(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing;
                     min_count::Integer = fe.min_count)
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    ties::Symbol = fe.ties
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    ic = Matrix{Tf}(undef, length(dates), 2)
    for (j, t) in enumerate(dates)
        a = view(alpha, t, :)
        b = view(y, t, :)
        ic[j, 1] = Tf(cs_spearman_correlation(a, b; min_count = min_count, ties = ties))
        ic[j, 2] = Tf(cs_weighted_correlation(a, b, view(u, t, :); min_count = min_count))
    end
    return ic
end
function forecast_ic(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                     weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                     min_count::Integer = fe.min_count)
    return forecast_ic(fe, cs_diagnostic_weights(weighting, csfm); min_count = min_count)
end
"""
    forecast_ic_lags(horizon::Integer, step::Integer) -> Int
    forecast_ic_lags(fe::ForecastEvaluationResult) -> Int

Return the number of consecutive evaluation dates whose forward windows overlap a given one.

A window of `horizon` observations, scored every `step` observations, spans `cld(horizon, step)` dates. The dates on each side of a given one that read some of the same returns are one fewer than that. It is the number of autocovariances the t-statistic of [`exposure_ic_factor_summary`](@ref) reads. The function derives it from the window and the stride and does not ask the caller for it, because the overlap is a fact of the grid. Under the default stride of [`forecast_evaluation`](@ref), `step = horizon`, it is `0`.

# Mathematical definition

```math
L = \\left\\lceil \\frac{h}{s} \\right\\rceil - 1
```

Where:

  - ``h``: The forward window, in observations.
  - $(math_dict[:s_eval_stride])

# Arguments

  - `horizon`: Forward window, in observations.
  - `step`: Number of observations between two evaluation dates.
  - `fe`: An evaluation. The function reads its `horizon` and its `step`.

# Validation

  - `horizon >= 1` and `step >= 1`. Raise a `DomainError`.

# Returns

  - `L::Int`: The number of overlapping dates on each side.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_ic_lags(5, 5)
0

julia> PortfolioOptimisers.forecast_ic_lags(5, 1)
4

julia> PortfolioOptimisers.forecast_ic_lags(5, 2)
2
```

# Related

  - [`forecast_ic_summary`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`forecast_evaluation_dates`](@ref)
"""
function forecast_ic_lags(horizon::Integer, step::Integer)
    @argcheck(horizon >= one(horizon), DomainError(horizon, "horizon must be >= 1"))
    @argcheck(step >= one(step), DomainError(step, "step must be >= 1"))
    return cld(Int(horizon), Int(step)) - 1
end
function forecast_ic_lags(fe::ForecastEvaluationResult)
    return forecast_ic_lags(fe.horizon, fe.step)
end
"""
    forecast_ic_summary(ic::MatNum; lags::Integer = 0)

Return the summary of the two information coefficient series of an evaluation, by name.

The summary gives five numbers for each series. The mean is the average score, and the standard deviation is how much the score moves. Their ratio is the score per unit of movement. The t-statistic says whether the mean is far enough from zero to believe, over the dates that carried a score. The hit rate is the share of those dates at which the score was positive.

The two series carry names and not positions, because a column of [`forecast_ic`](@ref) carries no name, and a read of the wrong column raises no error. [`exposure_ic_factor_summary`](@ref) summarises each series. The exposure diagnostics call the same function, so a coefficient of a forecast and a coefficient of a factor exposure have the same summary.

# The t-statistic reads the overlap of the windows

Two evaluation dates closer together than the forward window score some of the same returns. When the forecast keeps its ordering from one date to the next, the coefficients of the two dates are correlated. A t-statistic that treats them as independent then overstates the evidence, by up to the root of the number of dates a window spans for a forecast that barely changes. A forecast that draws a new ordering at every date gives coefficients with almost no correlation, and the plain statistic does not overstate its evidence. The standard error of the mean therefore reads the long-run variance of the series over `lags` autocovariances, as [`exposure_ic_factor_summary`](@ref) states.

`lags` is the number of dates on each side of a given one whose windows overlap it. [`forecast_ic_lags`](@ref) derives it from the `horizon` and the `step` of the evaluation. It is `0` under the default stride, where the windows are disjoint and the statistic is ``\\mathrm{IR} \\sqrt{n}``. It is `horizon - 1` at a stride of one. [`forecast_evaluation_summary`](@ref), [`forecast_holding_period`](@ref) and [`forecast_decay`](@ref) derive it. A caller who summarises a series by hand passes it.

# Mathematical definition

```math
t_{k} = \\frac{\\overline{\\mathrm{IC}}_{k}}{\\sigma_{k}} \\sqrt{\\left| \\mathcal{T}_{k} \\right|}
```

Where:

  - ``\\overline{\\mathrm{IC}}_{k}``: Mean of series ``k`` over its finite dates.
  - ``\\sigma_{k}``: Its long-run standard deviation over `lags` autocovariances, from [`exposure_ic_factor_summary`](@ref), which is its standard deviation at `lags = 0`.
  - ``\\mathcal{T}_{k}``: The dates at which series ``k`` is finite.

# Algorithm

 1. Check that `ic` is not empty and carries two columns.
 2. Summarise the first column with [`exposure_ic_factor_summary`](@ref) at `lags`, into `spearman`.
 3. Summarise the second column the same way, into `pearson`.

# Arguments

  - `ic`: Information coefficient series `dates × 2`, from [`forecast_ic`](@ref).
  - `lags`: Number of autocovariances the t-statistic's standard error reads, from [`forecast_ic_lags`](@ref).

# Validation

  - `!isempty(ic)`. Raises an [`IsEmptyError`](@ref).
  - `size(ic, 2) == 2`. Raises a `DimensionMismatch`.
  - `lags >= 0`. Raises a `DomainError`.

# Returns

  - `summary::NamedTuple`: `(; spearman, pearson)`, each `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> fe = forecast_evaluation(alpha, y);

julia> forecast_ic_summary(forecast_ic(fe); lags = PortfolioOptimisers.forecast_ic_lags(fe)).spearman
(mean_ic = 0.3333333333333333, std_ic = 0.7023769168568493, ic_ir = 0.4745789978762494, t_stat = 0.8219949365267862, hit_rate = 0.6666666666666666)
```

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_ic_lags`](@ref)
  - [`forecast_coverage`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`exposure_ic_summary`](@ref)
"""
function forecast_ic_summary(ic::MatNum; lags::Integer = 0)
    @argcheck(!isempty(ic), IsEmptyError("ic cannot be empty"))
    @argcheck(size(ic, 2) == 2,
              DimensionMismatch("ic must carry the two columns forecast_ic answers, the Spearman coefficient and the Pearson one, and it carries $(size(ic, 2))"))
    return (; spearman = exposure_ic_factor_summary(ic, 1; lags = lags),
            pearson = exposure_ic_factor_summary(ic, 2; lags = lags))
end
"""
    forecast_coverage(fe::ForecastEvaluationResult,
                      w::Option{<:MatNum} = nothing) -> Vector{<:Real}
    forecast_coverage(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                      weighting::AbstractOrthogonalityMetric = IdentityMetric()) -> Vector{<:Real}

Return the share of the universe an evaluation scored, one entry per evaluation date.

The evaluation scores an asset at a date when the asset carries both a finite forecast and a finite target there. The coverage is the share of the universe that the evaluation scored. Read every other statistic of the evaluation beside it. At a date where the coverage collapses, the information coefficient reads a small number of assets. A run of such dates shows that the forecast lost its Descriptors, not its skill.

The universe of a date is the estimation universe that the evaluation carries in `umsk`, narrowed to the assets of finite positive weight. These are the assets the Pearson coefficient of [`forecast_ic`](@ref) can read. On a point-in-time panel the estimation mask moves with the listings. An asset that has not listed yet, or that has delisted, is outside the universe and is not a missed asset, so a late lister is not a lost Descriptor. The share is therefore `1` at a date where the evaluation scored every asset the panel admits, however few. [`forecast_summary_scored`](@ref) gives the count.

The coverage does not apply the threshold of [`forecast_ic`](@ref). A date under the threshold carries no coefficient, and the coverage says why. A coverage silenced at the same threshold would give no answer at the dates that need one most.

# Mathematical definition

```math
c_{j} = \\frac{\\left| \\left\\{ i \\in \\mathcal{U}_{t_{j}} : \\alpha_{t_{j} i} \\text{ and } y_{t_{j} i} \\text{ are finite} \\right\\} \\right|}{\\left| \\mathcal{U}_{t_{j}} \\right|}
\\qquad
\\mathcal{U}_{t} = \\left\\{ i : m_{ti} \\text{ and } 0 < u_{ti} < \\infty \\right\\}
```

Where:

  - $(math_dict[:alpha_ti_fc])
  - $(math_dict[:y_ti_fwd])
  - $(math_dict[:m_ti_univ])
  - $(math_dict[:u_ti_cs])
  - $(math_dict[:U_t_univ])
  - $(math_dict[:t_j_eval])

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref).
 2. At each evaluation date, count the assets in `fe.umsk` of finite positive weight, into `ne`.
 3. Count the assets of those that carry a finite pair, into `nc`.
 4. Divide `nc` by `ne`, or answer `NaN` when `ne` is zero.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref). Its `umsk` is the universe of the share.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for every asset of the universe.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the weight history the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history that narrows the universe, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - The rules of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `c::Vector{<:Real}`: One entry per evaluation date, between `0` and `1`, in the promotion of the element types of `fe.alpha`, `fe.y` and the weight history. A date whose universe is empty carries `NaN`, because it holds nothing to cover.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 NaN; 1.0 NaN 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> forecast_coverage(forecast_evaluation(alpha, y; min_count = 2))
3-element Vector{Float64}:
 0.75
 0.5
 0.75
```

A universe that excludes the cells the panel does not admit, as the estimation mask of a point-in-time panel does, leaves the share at `1` where the evaluation scored every admitted asset:

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 NaN; 1.0 NaN 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> umsk = trues(4, 4);
       umsk[2, 2] = umsk[2, 4] = umsk[3, 2] = false;

julia> forecast_coverage(forecast_evaluation(alpha, y; umsk = umsk, min_count = 2))
3-element Vector{Float64}:
 0.75
 1.0
 1.0
```

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_summary_scored`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`exposure_coverage`](@ref)
"""
function forecast_coverage(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing)
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    umsk::AbstractMatrix{Bool} = fe.umsk
    dates::AbstractVector{<:Integer} = fe.dates
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    c = Vector{Tf}(undef, length(dates))
    for (j, t) in enumerate(dates)
        ne = 0
        nc = 0
        for i in axes(alpha, 2)
            if umsk[t, i] && 0 < u[t, i] < Inf
                ne += 1
                nc += isfinite(alpha[t, i]) && isfinite(y[t, i])
            end
        end
        c[j] = ne > 0 ? Tf(nc) / Tf(ne) : Tf(NaN)
    end
    return c
end
function forecast_coverage(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                           weighting::AbstractOrthogonalityMetric = IdentityMetric())
    return forecast_coverage(fe, cs_diagnostic_weights(weighting, csfm))
end

export forecast_ic, forecast_ic_summary, forecast_coverage
