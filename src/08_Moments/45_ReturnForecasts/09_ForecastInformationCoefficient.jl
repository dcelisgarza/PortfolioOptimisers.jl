"""
    forecast_ic_weights(alpha::MatNum, w::Nothing)
    forecast_ic_weights(alpha::MatNum, w::MatNum)

Return the cross-sectional weight history the Pearson information coefficient reads, checking it against the forecast.

It is the counterpart of [`exposure_weights`](@ref) on the observation-by-asset axis: an absent weight history means equal weights, and the absent case is resolved once, into a history of ones, so the kernel reads one matrix and carries no branch. A weight of zero excludes the asset from the observation, which is what [`cs_weighted_correlation`](@ref) already reads a non-positive weight as.

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

The information coefficient is the cross-sectional correlation between the forecast known at an observation and the forward target it is answerable for. It is the first thing to read off an evaluation, because it scores the forecast's **ordering** of the cross-section and asks nothing of its magnitude.

Both coefficients are answered, in two columns, because they score two different claims and a caller compares them. The **Spearman** coefficient correlates the ranks, so one asset with an extreme forecast moves it no more than one ordinary asset does, and it reads no weights. The **Pearson** coefficient correlates the levels under the cross-sectional weights, so it falls where the Spearman one holds when the forecast orders the assets well and spaces them badly.

# Mathematical definition

```math
\\mathrm{IC}^{\\mathrm{S}}_{j} = \\rho^{\\mathrm{S}} \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\boldsymbol{y}_{t_{j}} \\right)
\\qquad
\\mathrm{IC}^{\\mathrm{P}}_{j} = \\rho \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\boldsymbol{y}_{t_{j}}, \\boldsymbol{u}_{t_{j}} \\right)
```

Where:

  - ``\\boldsymbol{\\alpha}_{t}``: Cross-section of the Return Forecast at observation ``t``.
  - ``\\boldsymbol{y}_{t}``: Cross-section of the forward target at observation ``t``.
  - ``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``.
  - ``\\rho^{\\mathrm{S}}``: The rank correlation of two cross-sections.
  - ``\\rho``: The weighted correlation of two cross-sections.
  - ``t_{j}``: The ``j``-th evaluation date.

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref).
 2. At each evaluation date, correlate the forecast against the target with [`cs_spearman_correlation`](@ref) for the first column and [`cs_weighted_correlation`](@ref) for the second.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The rank column reads no weights.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the weight history the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the Pearson column is taken under, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default reads equal weights.
  - `min_count`: Least number of assets a cross-section needs before a coefficient of it is reported. It defaults to the threshold the evaluation carries, and a caller overrides it to read the same pairing at a second threshold.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.
  - The rules of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `ic::Matrix{<:Real}`: `dates × 2`. Row `j` scores the evaluation date `fe.dates[j]`, the first column carries the Spearman coefficient and the second the Pearson one. A date at which fewer than `min_count` assets carry both a finite forecast and a finite target carries `NaN`.

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
  - [`exposure_ic`](@ref)
"""
function forecast_ic(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing;
                     min_count::Integer = fe.min_count)
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    ic = Matrix{Tf}(undef, length(dates), 2)
    for (j, t) in enumerate(dates)
        a = view(alpha, t, :)
        b = view(y, t, :)
        ic[j, 1] = Tf(cs_spearman_correlation(a, b; min_count = min_count))
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
    forecast_ic_summary(ic::MatNum)

Return the summary of the two information coefficient series of an evaluation, named.

The mean states the average score, the standard deviation states how much the score moves, their ratio states the score per unit of movement, the t-statistic states whether the mean is far enough from zero to believe over the dates that carried a score, and the hit rate states how often the score was positive. The two series are named rather than indexed, because a column of [`forecast_ic`](@ref) carries no name of its own and reading the wrong one is silent.

Each series is summarised by [`exposure_ic_factor_summary`](@ref), which the exposure diagnostics already read, so a coefficient of a forecast and a coefficient of a factor exposure are summarised on the same terms.

# Mathematical definition

```math
t_{k} = \\mathrm{IR}_{k} \\sqrt{\\left| \\mathcal{T}_{k} \\right|}
```

Where:

  - ``\\mathrm{IR}_{k}``: Mean of series ``k`` over its finite dates, divided by their standard deviation.
  - ``\\mathcal{T}_{k}``: The dates at which series ``k`` is finite.

# Arguments

  - `ic`: Information coefficient series `dates × 2`, from [`forecast_ic`](@ref).

# Validation

  - `!isempty(ic)`. Raises an [`IsEmptyError`](@ref).
  - `size(ic, 2) == 2`. Raises a `DimensionMismatch`.

# Returns

  - `summary::NamedTuple`: `(; spearman, pearson)`, each `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> forecast_ic_summary(forecast_ic(forecast_evaluation(alpha, y))).spearman
(mean_ic = 0.3333333333333333, std_ic = 0.7023769168568493, ic_ir = 0.4745789978762494, t_stat = 0.8219949365267862, hit_rate = 0.6666666666666666)
```

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_coverage`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`exposure_ic_summary`](@ref)
"""
function forecast_ic_summary(ic::MatNum)
    @argcheck(!isempty(ic), IsEmptyError("ic cannot be empty"))
    @argcheck(size(ic, 2) == 2,
              DimensionMismatch("ic must carry the two columns forecast_ic answers, the Spearman coefficient and the Pearson one, and it carries $(size(ic, 2))"))
    return (; spearman = exposure_ic_factor_summary(ic, 1),
            pearson = exposure_ic_factor_summary(ic, 2))
end
"""
    forecast_coverage(fe::ForecastEvaluationResult,
                      w::Option{<:MatNum} = nothing) -> Vector{<:Real}
    forecast_coverage(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                      weighting::AbstractOrthogonalityMetric = IdentityMetric()) -> Vector{<:Real}

Return the share of the universe an evaluation scored, one entry per evaluation date.

An asset is scored at a date when it carries both a finite forecast and a finite target there, and the coverage is the share of the universe that did. It is the denominator every other statistic of the evaluation is read against: a date whose coverage has collapsed reports an information coefficient over a handful of assets, and a run of such dates says the forecast lost its Descriptors rather than its skill.

The threshold [`forecast_ic`](@ref) applies is deliberately **not** applied here. A date under it carries no coefficient, and the coverage is what says why, so silencing the coverage at the same threshold would answer nothing where the answer is most wanted.

# Mathematical definition

```math
c_{j} = \\frac{\\left| \\left\\{ i \\in \\mathcal{U}_{t_{j}} : \\alpha_{t_{j} i} \\text{ and } y_{t_{j} i} \\text{ are finite} \\right\\} \\right|}{\\left| \\mathcal{U}_{t_{j}} \\right|}
```

Where:

  - ``\\alpha_{ti}``: Return Forecast of asset ``i`` at observation ``t``.
  - ``y_{ti}``: Forward target of asset ``i`` at observation ``t``.
  - ``\\mathcal{U}_{t}``: Universe of observation ``t``, the assets of positive weight.
  - ``t_{j}``: The ``j``-th evaluation date.

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref).
 2. At each evaluation date, count the universe and the assets of it that carry a finite pair, and divide.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for every asset.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the weight history the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the universe is read off, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - The rules of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `c::Vector{<:Real}`: One entry per evaluation date, between `0` and `1`. A date whose universe is empty carries `NaN`, because there is nothing there to cover.

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

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`exposure_coverage`](@ref)
"""
function forecast_coverage(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing)
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    c = Vector{Tf}(undef, length(dates))
    for (j, t) in enumerate(dates)
        ne = 0
        nc = 0
        for i in axes(alpha, 2)
            if u[t, i] > 0
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
