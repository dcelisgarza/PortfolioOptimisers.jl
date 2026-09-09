"""
    forecast_factor_exposures(alpha::MatNum, B::Arr3Num)

Return the exposure history the factor correlation reads, checking it against the forecast.

The correlation scores one cross-section of the forecast against one cross-section of an exposure, so the two must be written on the same observations and the same assets. The check is stated once, here, because both methods of [`forecast_factor_correlation`](@ref) reach the same loop and a mismatched axis is otherwise caught as a silent `NaN` rather than as an error.

A block writes its exposure history on its own rows, and a Return Forecast history is cut to those rows by [`forecast_target_history`](@ref), so the two already agree when the evaluation was built on the block the exposures come from. A caller who passes a tensor of their own is told when it does not.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`.
  - `B`: Exposure history `observations × assets × factors`, unlagged.

# Validation

  - `!isempty(B)`. Raises an [`IsEmptyError`](@ref).
  - `size(B, 1) == size(alpha, 1)` and `size(B, 2) == size(alpha, 2)`. Raises a `DimensionMismatch`.

# Returns

  - `B::Arr3Num`: The history, unchanged.

# Related

  - [`forecast_factor_correlation`](@ref)
  - [`cs_diagnostic_exposures`](@ref)
  - [`forecast_ic_weights`](@ref)
"""
function forecast_factor_exposures(alpha::MatNum, B::Arr3Num)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    @argcheck(size(B, 1) == size(alpha, 1) && size(B, 2) == size(alpha, 2),
              DimensionMismatch("B ($(size(B, 1))×$(size(B, 2))×$(size(B, 3))) must match alpha ($(size(alpha, 1))×$(size(alpha, 2))) on its observation and asset axes"))
    return B
end
"""
    forecast_factor_correlation(fe::ForecastEvaluationResult, B::Arr3Num,
                                w::Option{<:MatNum} = nothing; rank::Bool = false,
                                min_count::Integer = fe.min_count) -> Matrix{<:Real}
    forecast_factor_correlation(fe::ForecastEvaluationResult,
                                csfm::CrossSectionalFactorModel; rank::Bool = false,
                                weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                min_count::Integer = fe.min_count) -> Matrix{<:Real}

Return the contemporaneous correlation of a Return Forecast against every factor exposure, one row per evaluation date.

A Return Forecast is meant to add alpha **over** the risk factors, not to restate them. This verb asks whether it does: it correlates the cross-section of the forecast against the cross-section of each exposure at the same observation, so a forecast that is a repackaged size tilt reads near `±1` on the size factor and a forecast that is neutral to the model reads near `0` on every one of them.

The correlation is contemporaneous rather than forward-looking, which is what separates it from [`forecast_ic`](@ref): the information coefficient asks whether the forecast predicts the future, and this asks what the forecast is made of. A high information coefficient earned by a high factor correlation is a factor premium wearing a forecast's clothes, and reading the two together is the only way to tell them apart.

The summary is the one [`exposure_ic_summary`](@ref) already answers over a `dates × factors` correlation series, so no summary of its own ships: a coefficient of a factor exposure and a correlation of a forecast against one are summarised on the same terms, through the same kernel.

# Mathematical definition

```math
c_{jk} = \\begin{cases}
\\rho^{\\mathrm{S}} \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\mathbf{B}_{t_{j} \\cdot k} \\right) & \\text{rank}\\,, \\\\
\\rho \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\mathbf{B}_{t_{j} \\cdot k}, \\boldsymbol{u}_{t_{j}} \\right) & \\text{otherwise}\\,.
\\end{cases}
```

Where:

  - ``\\boldsymbol{\\alpha}_{t}``: Cross-section of the Return Forecast at observation ``t``.
  - ``\\mathbf{B}_{t \\cdot k}``: Cross-section of factor ``k``'s exposure at observation ``t``.
  - ``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``.
  - ``\\rho^{\\mathrm{S}}``: The rank correlation of two cross-sections.
  - ``\\rho``: The weighted correlation of two cross-sections.
  - ``t_{j}``: The ``j``-th evaluation date.

# Algorithm

 1. Check the exposure history against the forecast with [`forecast_factor_exposures`](@ref), and resolve the weight history with [`forecast_ic_weights`](@ref).
 2. At each evaluation date and each factor, correlate the forecast against the exposure with [`cs_spearman_correlation`](@ref) when `rank`, and with [`cs_weighted_correlation`](@ref) otherwise.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref).
  - `B`: Exposure history `observations × assets × factors`, unlagged, on the axis of `fe.alpha`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The rank form reads no weights.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the exposure history, through [`cs_diagnostic_exposures`](@ref), and the weight history the metric names.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise. It is the spelling [`exposure_ic`](@ref) already uses, and it defaults to the weighted form because the levels are what a neutralisation acts on.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the weighted form is taken under, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default reads equal weights.
  - `min_count`: Least number of assets a cross-section needs before a correlation of it is reported. It defaults to the threshold the evaluation carries, and a caller overrides it to read the same pairing at a second threshold.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.
  - The rules of [`forecast_factor_exposures`](@ref), of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_exposures`](@ref) and [`cs_diagnostic_weights`](@ref).

# Returns

  - `c::Matrix{<:Real}`: `dates × factors`. Entry `(j, k)` scores the evaluation date `fe.dates[j]` against factor `k`. A date at which fewer than `min_count` assets carry both a finite forecast and a finite exposure of that factor carries `NaN` there.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> G = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0; 1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0];

julia> B = cat(alpha, G; dims = 3);

julia> forecast_factor_correlation(forecast_evaluation(alpha, y), B)
3×2 Matrix{Float64}:
 1.0  -0.466252
 1.0   0.565546
 1.0   0.169031
```

The forecast is the first exposure, so it correlates `1` with it at every date, and it is unrelated to the second, which wanders.

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`forecast_factor_exposures`](@ref)
  - [`exposure_ic_summary`](@ref)
  - [`cs_diagnostic_exposures`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`cs_weighted_correlation`](@ref)
"""
function forecast_factor_correlation(fe::ForecastEvaluationResult, B::Arr3Num,
                                     w::Option{<:MatNum} = nothing; rank::Bool = false,
                                     min_count::Integer = fe.min_count)
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    alpha::MatNum = fe.alpha
    dates::AbstractVector{<:Integer} = fe.dates
    Bf = forecast_factor_exposures(alpha, B)
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(Bf)), real(eltype(u)))
    K = size(Bf, 3)
    c = Matrix{Tf}(undef, length(dates), K)
    for k in 1:K, (j, t) in enumerate(dates)
        a = view(alpha, t, :)
        b = view(Bf, t, :, k)
        c[j, k] = if rank
            Tf(cs_spearman_correlation(a, b; min_count = min_count))
        else
            Tf(cs_weighted_correlation(a, b, view(u, t, :); min_count = min_count))
        end
    end
    return c
end
function forecast_factor_correlation(fe::ForecastEvaluationResult,
                                     csfm::CrossSectionalFactorModel; rank::Bool = false,
                                     weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                     min_count::Integer = fe.min_count)
    return forecast_factor_correlation(fe, cs_diagnostic_exposures(csfm),
                                       cs_diagnostic_weights(weighting, csfm); rank = rank,
                                       min_count = min_count)
end

export forecast_factor_correlation
