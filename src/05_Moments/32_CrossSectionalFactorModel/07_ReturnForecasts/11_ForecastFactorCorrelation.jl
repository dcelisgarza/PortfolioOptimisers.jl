"""
    forecast_factor_exposures(alpha::MatNum, B::Arr3Num)

Return the exposure history that the factor correlation reads, after a check of its axes against the forecast.

The correlation scores one cross-section of the forecast against one cross-section of an exposure, so the two histories must hold the same observations and the same assets. Both methods of [`forecast_factor_correlation`](@ref) reach this check, so it is stated once. Without it, a tensor with fewer observations or fewer assets than the forecast throws a `BoundsError` inside the loop, and a larger tensor is read by position, so an offset axis gives a wrong correlation and no error.

A block writes its exposure history on its own rows, and [`forecast_history`](@ref) returns a Return Forecast history on the same rows. The two therefore agree when the evaluation was built on the block that the exposures come from, and the check matters for a tensor that the caller builds.

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
  - [`forecast_history`](@ref)
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
                                w::Option{<:MatNum} = nothing;
                                dates::AbstractVector{<:Integer} = axes(fe.alpha, 1),
                                rank::Bool = false,
                                min_count::Integer = fe.min_count) -> Matrix{<:Real}
    forecast_factor_correlation(fe::ForecastEvaluationResult,
                                csfm::CrossSectionalFactorModel;
                                dates::AbstractVector{<:Integer} = axes(fe.alpha, 1),
                                rank::Bool = false,
                                weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                min_count::Integer = fe.min_count) -> Matrix{<:Real}

Return the contemporaneous correlation of a Return Forecast against every factor exposure, one row per observation.

A Return Forecast must add alpha over the risk factors. This verb measures how much of the forecast restates them instead. At each observation it correlates the cross-section of the forecast with the cross-section of each exposure. A forecast that is a size tilt under another name reads near `±1` on the size factor, and a forecast that is neutral to the model reads near `0` on every factor.

The correlation is contemporaneous, and this separates it from [`forecast_ic`](@ref). The information coefficient measures whether the forecast predicts the forward target, and this verb measures what the forecast is made of. When a forecast with a high information coefficient also correlates highly with a factor, part of its score is the premium of that factor, and the two verbs read together show it.

The statistic reads no forward window, so every observation at which the forecast and an exposure are both finite is a sample of it. The verb therefore reads the whole observation axis of `fe.alpha` by default, and not the evaluation grid `fe.dates`. The grid keeps the forward windows of [`forecast_ic`](@ref) apart, and this verb has no forward window. Under the default `step = horizon`, the grid holds about `1 / horizon` of the observations, so a summary over the grid carries a t-statistic about `sqrt(horizon)` times smaller for the same forecast. Pass `dates = fe.dates` to read the grid, or any other vector of rows.

A member that computes no history is refit along the grid by [`forecast_history`](@ref), and its history is `NaN` off the grid. Its correlation is then `NaN` on those rows. [`exposure_ic_summary`](@ref) drops a `NaN` row, so the summary of the whole axis equals the summary of the rows that the member wrote.

This verb has no summary of its own. [`exposure_ic_summary`](@ref) summarises any `dates × factors` correlation series, so the coefficients of an exposure and the correlations of a forecast with an exposure share one summary kernel.

# Mathematical definition

```math
c_{jk} = \\begin{cases}
\\rho^{\\mathrm{S}} \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\mathbf{B}_{t_{j} \\cdot k} \\right) & \\text{rank}\\,, \\\\
\\rho \\left( \\boldsymbol{\\alpha}_{t_{j}}, \\mathbf{B}_{t_{j} \\cdot k}, \\boldsymbol{u}_{t_{j}} \\right) & \\text{otherwise}\\,.
\\end{cases}
```

Where:

  - $(math_dict[:alpha_t_fc])
  - $(math_dict[:B_tk_cs])
  - $(math_dict[:u_t_cs])
  - $(math_dict[:rho_rank_cs])
  - $(math_dict[:rho_w_cs])
  - ``t_{j}``: The ``j``-th row of `dates`, every observation by default.

# Algorithm

 1. Check the exposure history against the forecast with [`forecast_factor_exposures`](@ref).
 2. Make the weight history with [`forecast_ic_weights`](@ref). An absent history becomes a history of ones.
 3. At each row of `dates` and each factor, compute ``c_{jk}`` with [`cs_spearman_correlation`](@ref) under `fe.ties` when `rank`, and with [`cs_weighted_correlation`](@ref) otherwise.

# Arguments

  - `fe`: The evaluation, from [`forecast_evaluation`](@ref).
  - `B`: Exposure history `observations × assets × factors`, unlagged, on the axis of `fe.alpha`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The rank form reads no weights.
  - `csfm`: The fitted factor-model block that the evaluation was built on. [`cs_diagnostic_exposures`](@ref) reads the exposure history from it, and [`cs_diagnostic_weights`](@ref) reads the weight history that `weighting` names.
  - `dates`: Row indices of `fe.alpha` at which the verb reads the correlation. The default is every observation. `fe.dates` reads the evaluation grid, which puts each correlation on the date of a coefficient of [`forecast_ic`](@ref).
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise. [`exposure_ic`](@ref) uses the same keyword. The default is the weighted form, because a neutralisation acts on the levels. The rank form ranks a tie by `fe.ties`, as [`cs_spearman_correlation`](@ref) states. Under the default `:average`, a factor with many equal exposures, such as an industry dummy, compares the forecasts of its blocks, and the order of the assets inside a block does not move the correlation. A constant exposure then reads `NaN`. Under `:ordinal`, two equal values take two ranks in the order of the asset axis, so the correlation depends on that order, and a constant exposure reads the rank correlation of the forecast with the asset order.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history of the weighted form, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default, [`IdentityMetric`](@ref), gives equal weights.
  - `min_count`: Least number of assets that must enter a cross-section for the verb to report its correlation. The default is the threshold of the evaluation. Pass another value to read the same pairing at a second threshold.

# Validation

  - `min_count >= 1`. Raises a `DomainError`.
  - `!isempty(dates)`. Raises an [`IsEmptyError`](@ref).
  - Every row of `dates` is a row of `fe.alpha`. Raises a `DomainError`.
  - The rules of [`forecast_factor_exposures`](@ref), of [`forecast_ic_weights`](@ref) and, for the block method, of [`cs_diagnostic_exposures`](@ref) and [`cs_diagnostic_weights`](@ref).

# Returns

  - `c::Matrix{<:Real}`: `length(dates) × factors`, which is `observations × factors` by default. Entry `(j, k)` scores the observation `dates[j]` against factor `k`. An asset enters entry `(j, k)` when its forecast and its exposure to factor `k` are finite and, in the weighted form, its weight is finite and positive. The entry is `NaN` when fewer than `min_count` assets enter, and, in the weighted form, when the forecast or the exposure is constant over the assets that enter.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> G = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0; 1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0];

julia> B = cat(alpha, G; dims = 3);

julia> fe = forecast_evaluation(alpha, y);

julia> forecast_factor_correlation(fe, B)
4×2 Matrix{Float64}:
 1.0  -0.466252
 1.0   0.565546
 1.0   0.169031
 1.0   0.534522

julia> forecast_factor_correlation(fe, B; dates = fe.dates)
3×2 Matrix{Float64}:
 1.0  -0.466252
 1.0   0.565546
 1.0   0.169031
```

The first exposure is the forecast itself, so the first column is `1` at every observation. The second exposure is a pattern of zeros and ones that changes with the observation, and its column moves between `-0.47` and `0.57`. The last observation has no forward target, so it is not on the evaluation grid. It carries a forecast and an exposure, so the default read scores it, and `dates = fe.dates` leaves it out.

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`forecast_factor_exposures`](@ref)
  - [`forecast_history`](@ref)
  - [`forecast_ic_weights`](@ref)
  - [`exposure_ic_summary`](@ref)
  - [`cs_diagnostic_exposures`](@ref)
  - [`cs_diagnostic_weights`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`cs_weighted_correlation`](@ref)
"""
function forecast_factor_correlation(fe::ForecastEvaluationResult, B::Arr3Num,
                                     w::Option{<:MatNum} = nothing;
                                     dates::AbstractVector{<:Integer} = axes(fe.alpha, 1),
                                     rank::Bool = false, min_count::Integer = fe.min_count)
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    @argcheck(!isempty(dates), IsEmptyError("dates cannot be empty"))
    alpha::MatNum = fe.alpha
    @argcheck(all(in(axes(alpha, 1)), dates),
              DomainError(dates,
                          "dates must be rows of fe.alpha, whose observation axis is $(axes(alpha, 1))"))
    ties::Symbol = fe.ties
    Bf = forecast_factor_exposures(alpha, B)
    u = forecast_ic_weights(alpha, w)
    Tf = promote_type(real(eltype(alpha)), real(eltype(Bf)), real(eltype(u)))
    K = size(Bf, 3)
    c = Matrix{Tf}(undef, length(dates), K)
    for k in 1:K, (j, t) in enumerate(dates)
        a = view(alpha, t, :)
        b = view(Bf, t, :, k)
        c[j, k] = if rank
            Tf(cs_spearman_correlation(a, b; min_count = min_count, ties = ties))
        else
            Tf(cs_weighted_correlation(a, b, view(u, t, :); min_count = min_count))
        end
    end
    return c
end
function forecast_factor_correlation(fe::ForecastEvaluationResult,
                                     csfm::CrossSectionalFactorModel;
                                     dates::AbstractVector{<:Integer} = axes(fe.alpha, 1),
                                     rank::Bool = false,
                                     weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                                     min_count::Integer = fe.min_count)
    return forecast_factor_correlation(fe, cs_diagnostic_exposures(csfm),
                                       cs_diagnostic_weights(weighting, csfm);
                                       dates = dates, rank = rank, min_count = min_count)
end

export forecast_factor_correlation
