"""
    ew_forecast_weights(unit::IdiosyncraticReturnUnit, w::MatNum, vs::MatNum) -> MatNum
    ew_forecast_weights(unit::IdiosyncraticSharpeUnit, w::MatNum, vs::MatNum) -> MatNum

Return the regression weights of an [`ExpWeightedReturnForecast`](@ref), in its Forecast Unit.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticReturnUnit`](@ref): the fit runs on the idiosyncratic return, whose noise scales with the idiosyncratic variance, so the estimation mask is divided by that variance and the fit is a generalised least squares.
 2. [`IdiosyncraticSharpeUnit`](@ref): the target already carries the division, so the estimation mask is the whole weight and the fit is an ordinary least squares.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `w`: Cross-sectional weights, `observations × assets`, the estimation mask read as numbers.
  - `vs`: Idiosyncratic variance history, `observations × assets`.

# Returns

  - `W::MatNum`: Regression weights, `observations × assets`.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`AbstractForecastUnit`](@ref)
  - [`return_forecast_weights`](@ref)
  - [`forecast_unit_target`](@ref)
"""
function ew_forecast_weights(::IdiosyncraticReturnUnit, w::MatNum, vs::MatNum)::MatNum
    return w ./ vs
end
function ew_forecast_weights(::IdiosyncraticSharpeUnit, w::MatNum, ::MatNum)::MatNum
    return w
end
"""
    ew_forecast_valid(S::Arr3Num, y::MatNum, vs::MatNum, w::MatNum) -> BitMatrix

Return the mask of the `(observation, asset)` pairs an [`ExpWeightedReturnForecast`](@ref) fits on.

A pair enters the fit when it carries a positive cross-sectional weight, a finite forward target, a finite idiosyncratic variance and a finite score for every Descriptor. The four conditions are read once and the answer is a mask, so the per-observation gather never re-derives them.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `y`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.

# Returns

  - `valid::BitMatrix`: Eligibility mask, `observations × assets`.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ew_forecast_design`](@ref)
  - [`descriptor_scores`](@ref)
"""
function ew_forecast_valid(S::Arr3Num, y::MatNum, vs::MatNum, w::MatNum)::BitMatrix
    valid = falses(size(w))
    for idx in CartesianIndices(valid)
        ok = w[idx] > zero(w[idx]) && isfinite(y[idx]) && isfinite(vs[idx])
        for k in axes(S, 3)
            ok &= isfinite(S[idx, k])
        end
        valid[idx] = ok
    end
    return valid
end
"""
    ew_forecast_design(S::Arr3Num, y::MatNum, W::MatNum, t::Integer,
                       valid::AbstractMatrix{Bool}) -> Tuple

Gather the design, the target and the weights of one observation of an [`ExpWeightedReturnForecast`](@ref).

The three arrays are materialised at the common floating point type rather than viewed, because the score array carries an open element type and a view of it names no numeric matrix.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `y`: The target in the Forecast Unit, `observations × assets`.
  - `W`: Regression weights, `observations × assets`.
  - `t`: Index of the observation.
  - `valid`: Eligibility mask, `observations × assets`.

# Returns

  - `(Sb, yb, wb)::Tuple`: The scores `valid assets × descriptors`, the target and the weights of the observation.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ew_forecast_valid`](@ref)
  - [`ew_forecast_accumulate!`](@ref)
"""
function ew_forecast_design(S::Arr3Num, y::MatNum, W::MatNum, t::Integer,
                            valid::AbstractMatrix{Bool})
    Tf = promote_type(float(real(eltype(S))), float(real(eltype(y))),
                      float(real(eltype(W))))
    idx = findall(view(valid, t, :))
    return Tf.(view(S, t, idx, :)), Tf.(view(y, t, idx)), Tf.(view(W, t, idx))
end
"""
    ew_forecast_accumulate!(A::AbstractMatrix{<:Real}, c::AbstractVector{<:Real},
                            Sb::MatNum, yb::VecNum, wb::VecNum, decay::Real,
                            normalise::Bool) -> nothing

Advance the exponentially weighted normal equations of an [`ExpWeightedReturnForecast`](@ref) by one observation, in place.

# Algorithm

 1. When `normalise` is set, divide the weights by their mean over the valid assets, so that an observation enters with the same aggregate weight whatever the level of its idiosyncratic variances. A mean that is not finite or not positive leaves the weights as they stand.
 2. Scale the design and the target by the square root of the weights, which is the weighted least squares of the observation written as an ordinary one.
 3. Advance `A` to `λ A + (1 - λ) Sw' Sw` and `c` to `λ c + (1 - λ) Sw' yw`.

# Arguments

  - `A`: Exponentially weighted normal matrix `descriptors × descriptors`, changed in place.
  - `c`: Exponentially weighted cross product, of length `descriptors`, changed in place.
  - `Sb`: The scores of the observation, `valid assets × descriptors`.
  - `yb`: The target of the observation, of length `valid assets`.
  - `wb`: The weights of the observation, of length `valid assets`.
  - `decay`: The decay factor.
  - `normalise`: Whether the weights of an observation are divided by their mean.

# Returns

  - `nothing`. `A` and `c` carry the advanced state.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ew_forecast_design`](@ref)
  - [`ew_forecast_solve`](@ref)
"""
function ew_forecast_accumulate!(A::AbstractMatrix{<:Real}, c::AbstractVector{<:Real},
                                 Sb::MatNum, yb::VecNum, wb::VecNum, decay::Real,
                                 normalise::Bool)::Nothing
    Tf = eltype(A)
    s = normalise ? sum(wb) / length(wb) : one(Tf)
    sw = sqrt.(wb ./ (isfinite(s) && s > zero(s) ? s : one(Tf)))
    Sw = Sb .* sw
    yw = yb .* sw
    A .= decay .* A .+ (one(Tf) - decay) .* (transpose(Sw) * Sw)
    c .= decay .* c .+ (one(Tf) - decay) .* (transpose(Sw) * yw)
    return nothing
end
"""
    ew_forecast_solve(A::MatNum, c::VecNum, ridge::Real, t::Integer) -> VecNum

Solve the ridge stabilised normal equations of an [`ExpWeightedReturnForecast`](@ref).

# Algorithm

 1. When `ridge` is positive, add `ridge` times the mean absolute diagonal entry of `A` to that diagonal, with the floor `eps` on the mean, so an empty normal matrix still gives a finite penalty.
 2. Solve through [`cross_sectional_solve`](@ref) under [`PseudoInverseFallback`](@ref), which takes the plain solve on a full rank matrix and the minimum-norm solution on a rank deficient one.

# Arguments

  - `A`: Exponentially weighted normal matrix `descriptors × descriptors`.
  - `c`: Exponentially weighted cross product, of length `descriptors`.
  - `ridge`: Relative ridge penalty.
  - `t`: Index of the observation.

# Returns

  - `coef::VecNum`: Descriptor coefficients, of length `descriptors`.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ew_forecast_accumulate!`](@ref)
  - [`cross_sectional_solve`](@ref)
  - [`PseudoInverseFallback`](@ref)
"""
function ew_forecast_solve(A::MatNum, c::VecNum, ridge::Real, t::Integer)::VecNum
    Tf = promote_type(float(real(eltype(A))), float(real(eltype(c))))
    K = size(A, 1)
    Ar = Matrix{Tf}(A)
    if ridge > zero(ridge)
        d = zero(Tf)
        for k in 1:K
            d += abs(Ar[k, k])
        end
        r = ridge * max(d / K, eps(Tf))
        for k in 1:K
            Ar[k, k] += r
        end
    end
    return cross_sectional_solve(PseudoInverseFallback(), Ar, c, t)
end
"""
    ew_forecast_history(S::Arr3Num, coefs::MatNum, gap::Integer) -> Matrix{<:Real}

Read the Return Forecast history of an [`ExpWeightedReturnForecast`](@ref) off its coefficient history.

The forecast at observation `t` uses the coefficients estimated from the targets that are known at `t`. A target of observation `s` matures `lag + horizon - 1` observations later, so the row `t` reads the coefficient row `t - gap`, and the first `gap` rows read none and stay `NaN`.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `coefs`: The coefficient history, `trainable observations × descriptors`.
  - `gap`: `lag + horizon - 1`, the number of observations a target takes to mature.

# Returns

  - `H::Matrix{<:Real}`: The Return Forecast history in the Forecast Unit, `observations × assets`.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ew_forecast_solve`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function ew_forecast_history(S::Arr3Num, coefs::MatNum, gap::Integer)::Matrix{<:Real}
    Tf = promote_type(float(real(eltype(S))), float(real(eltype(coefs))))
    T = size(S, 1)
    H = fill(Tf(NaN), T, size(S, 2))
    for t in (gap + 1):T, i in axes(S, 2)
        h = zero(Tf)
        for k in axes(S, 3)
            h += S[t, i, k] * coefs[t - gap, k]
        end
        H[t, i] = h
    end
    return H
end
"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast whose Descriptor weights are fitted by exponentially weighted least squares.

The member turns its Descriptors into scores with the recipe in `scores`, and regresses the forward mean idiosyncratic return of every observation on them across the assets. The normal equations of the observations are combined under an exponential decay rather than solved one at a time, so the coefficients carry the whole history and move slowly. No intercept is fitted, so the forecast absorbs no cross-sectional mean and suits a long-short caller.

The Forecast Unit chooses the pair of target and weights. In the return unit the target is the forward return and the weights are the inverse idiosyncratic variance, which is a generalised least squares. In the Sharpe unit the target is the forward return divided by the idiosyncratic volatility and the weights are the estimation mask, which is the same estimator written as an ordinary least squares, and the forecast is multiplied by the idiosyncratic volatility at the end.

# Mathematical definition

```math
\\begin{align}
\\mathbf{A}_{t} &= \\lambda \\mathbf{A}_{t-1} + (1 - \\lambda) \\mathbf{S}_{t}^{\\intercal} \\mathbf{W}_{t} \\mathbf{S}_{t}\\,, \\\\
\\boldsymbol{c}_{t} &= \\lambda \\boldsymbol{c}_{t-1} + (1 - \\lambda) \\mathbf{S}_{t}^{\\intercal} \\mathbf{W}_{t} \\boldsymbol{y}_{t}\\,, \\\\
\\boldsymbol{\\beta}_{t} &= \\left(\\mathbf{A}_{t} + \\rho_{t} \\mathbf{I}\\right)^{-1} \\boldsymbol{c}_{t}\\,, &
\\rho_{t} &= \\varrho \\, \\overline{\\lvert \\operatorname{diag} \\mathbf{A}_{t} \\rvert}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}_{t}``: Descriptor scores of the valid assets of observation ``t``.
  - ``\\mathbf{W}_{t}``: diagonal of the regression weights of those assets.
  - ``\\boldsymbol{y}_{t}``: forward target of those assets.
  - ``\\lambda``: the decay factor.
  - ``\\varrho``: the relative ridge penalty.
  - ``\\boldsymbol{\\beta}_{t}``: Descriptor coefficients after observation ``t``.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    ExpWeightedReturnForecast(; scores::DescriptorScores, half_life::Real = 20.0,
                              decay::Real = half_life_decay(half_life),
                              min_obs::Integer = half_life_min_obs(half_life),
                              ridge::Real = 1e-6, horizon::Integer = 1, lag::Integer = 1,
                              scale::Real = 1.0, normalise::Bool = true,
                              unit::AbstractForecastUnit = IdiosyncraticReturnUnit()) -> ExpWeightedReturnForecast

Every keyword but `half_life` corresponds to a field. `half_life` is not a field: it fixes the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it stands. `min_obs = 1` publishes a forecast from the first fitted observation.

## Validation

  - `0 < decay < 1`.
  - `min_obs >= 1`, `horizon >= 1` and `lag >= 1`.
  - `ridge >= 0` and is finite.
  - `scale > 0` and is finite.

# Examples

```jldoctest
julia> ds = DescriptorScores(; descriptors = [Passthrough(; field = \"a\")]);

julia> ExpWeightedReturnForecast(; scores = ds, half_life = 2)
ExpWeightedReturnForecast
     scores ┼ DescriptorScores
            │   descriptors ┼ 1-element Vector{Passthrough}
            │               │ Passthrough ⋯
            │    neutralise ┼ nothing
            │       outlier ┼ CrossSectionalWinsoriser
            │               │    low ┼ Float64: 0.01
            │               │   high ┴ Float64: 0.99
            │       scoring ┼ CrossSectionalStandardiser
            │               │   min_group_size ┼ Int64: 8
            │               │             atol ┴ Float64: 1.0e-12
            │         group ┴ nothing
      decay ┼ Float64: 0.7071067811865476
    min_obs ┼ Int64: 2
      ridge ┼ Float64: 1.0e-6
    horizon ┼ Int64: 1
        lag ┼ Int64: 1
      scale ┼ Float64: 1.0
  normalise ┼ Bool: true
       unit ┴ IdiosyncraticReturnUnit()
```

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`ExpWeightedReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`DescriptorScores`](@ref)
  - [`AbstractForecastUnit`](@ref)
  - [`TargetReturnForecast`](@ref)

# References

  - $(ref_dict[:grinoldkahn1999])
"""
@concrete struct ExpWeightedReturnForecast <: AbstractReturnForecastEstimator
    """
    $(field_dict[:rf_scores])
    """
    scores
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    Relative ridge penalty added to the diagonal of the exponentially weighted normal matrix before it is solved.
    """
    ridge
    """
    $(field_dict[:rf_horizon])
    """
    horizon
    """
    $(field_dict[:rf_lag])
    """
    lag
    """
    $(field_dict[:rf_scale])
    """
    scale
    """
    Whether the regression weights of an observation are divided by their mean over its valid assets, so that a calm regime does not dominate the state through the size of its inverse variances.
    """
    normalise
    """
    $(field_dict[:rf_unit])
    """
    unit
    function ExpWeightedReturnForecast(scores::DescriptorScores, decay::Real,
                                       min_obs::Integer, ridge::Real, horizon::Integer,
                                       lag::Integer, scale::Real, normalise::Bool,
                                       unit::AbstractForecastUnit)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_finite(ridge, :ridge)
        assert_nonneg(ridge, :ridge)
        assert_nonempty_gt0_finite_val(horizon, :horizon)
        assert_nonempty_gt0_finite_val(lag, :lag)
        assert_finite(scale, :scale)
        assert_gt0(scale, :scale)
        return new{typeof(scores), typeof(decay), typeof(min_obs), typeof(ridge),
                   typeof(horizon), typeof(lag), typeof(scale), typeof(normalise),
                   typeof(unit)}(scores, decay, min_obs, ridge, horizon, lag, scale,
                                 normalise, unit)
    end
end
function ExpWeightedReturnForecast(; scores::DescriptorScores, half_life::Real = 20.0,
                                   decay::Real = half_life_decay(half_life),
                                   min_obs::Integer = half_life_min_obs(half_life),
                                   ridge::Real = 1e-6, horizon::Integer = 1,
                                   lag::Integer = 1, scale::Real = 1.0,
                                   normalise::Bool = true,
                                   unit::AbstractForecastUnit = IdiosyncraticReturnUnit())::ExpWeightedReturnForecast
    return ExpWeightedReturnForecast(scores, decay, min_obs, ridge, horizon, lag, scale,
                                     normalise, unit)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`ExpWeightedReturnForecast`](@ref).

Beside the two reads [`AbstractReturnForecastResult`](@ref) states, it carries the whole fitted state of the recursion: the latest coefficients, the two exponentially weighted accumulators and the count of the observations that advanced them. An update seam resumes the fit from those four fields alone, so nothing of the fit is lost by storing the Result rather than the estimator.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`ExpWeightedReturnForecast`](@ref)
  - [`return_forecast`](@ref)
"""
@concrete struct ExpWeightedReturnForecastResult <: AbstractReturnForecastResult
    """
    $(field_dict[:rf_mu])
    """
    mu
    """
    $(field_dict[:rf_hist])
    """
    hist
    """
    Latest Descriptor coefficients, in the order the Descriptors are written in. They are `NaN` only when no observation advanced the recursion: `min_obs` gates the publication of a forecast, not the state the recursion carries.
    """
    coef
    """
    Exponentially weighted normal matrix `descriptors × descriptors`.
    """
    A
    """
    Exponentially weighted cross product of the scores and the target, of length `descriptors`.
    """
    c
    """
    Count of the observations that advanced the recursion.
    """
    n
    function ExpWeightedReturnForecastResult(mu::VecNum, hist::MatNum, coef::VecNum,
                                             A::MatNum, c::VecNum, n::Integer)
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        @argcheck(!isempty(hist), IsEmptyError("hist cannot be empty"))
        @argcheck(!isempty(coef), IsEmptyError("coef cannot be empty"))
        @argcheck(length(mu) == size(hist, 2),
                  DimensionMismatch("mu ($(length(mu))) must match hist ($(size(hist, 2)) columns)"))
        K = length(coef)
        @argcheck(size(A) == (K, K),
                  DimensionMismatch("A ($(size(A, 1))×$(size(A, 2))) must be square over the Descriptors ($K)"))
        @argcheck(length(c) == K,
                  DimensionMismatch("c ($(length(c))) must match coef ($K)"))
        assert_nonneg(n, :n)
        return new{typeof(mu), typeof(hist), typeof(coef), typeof(A), typeof(c), typeof(n)}(mu,
                                                                                            hist,
                                                                                            coef,
                                                                                            A,
                                                                                            c,
                                                                                            n)
    end
end
function ExpWeightedReturnForecastResult(; mu::VecNum, hist::MatNum, coef::VecNum,
                                         A::MatNum, c::VecNum,
                                         n::Integer)::ExpWeightedReturnForecastResult
    return ExpWeightedReturnForecastResult(mu, hist, coef, A, c, n)
end
"""
    return_forecast(rfe::ExpWeightedReturnForecast, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> ExpWeightedReturnForecastResult

Fit a Return Forecast by exponentially weighted least squares on the forward idiosyncratic return.

# Algorithm

 1. Compute the Descriptor scores through [`descriptor_scores`](@ref), and read the idiosyncratic returns and variances off the block.
 2. Take the forward mean target through [`forward_mean_returns`](@ref), and convert it to the Forecast Unit through [`forecast_unit_target`](@ref).
 3. Read the regression weights through [`ew_forecast_weights`](@ref) and the eligibility mask through [`ew_forecast_valid`](@ref).
 4. Over the observations whose target is known, which are all but the last `lag + horizon - 1`, advance the normal equations through [`ew_forecast_accumulate!`](@ref) and solve them through [`ew_forecast_solve`](@ref). An observation with no valid asset advances nothing and carries the previous coefficients forward.
 5. Write the coefficients of an observation into the history only after `min_obs` observations have advanced the recursion.
 6. Read the history through [`ew_forecast_history`](@ref), multiply by `scale`, and convert the whole history to return units through [`forecast_return_units`](@ref).
 7. Read `mu` off the last observation of that history, which is the latest scores under the latest coefficients.

# Arguments

  - `rfe`: Exponentially weighted Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. It must carry the cross-sectional fit and the idiosyncratic variance history, and its exposure history is read under a Neutralisation.

# Validation

  - The rules of [`descriptor_scores`](@ref), of [`forecast_idiosyncratic_returns`](@ref) and of [`forecast_idiosyncratic_variances`](@ref).

# Returns

  - `rf::ExpWeightedReturnForecastResult`: The fitted forecast, its history and the state of the recursion.

# Related

  - [`ExpWeightedReturnForecast`](@ref)
  - [`ExpWeightedReturnForecastResult`](@ref)
  - [`ew_forecast_accumulate!`](@ref)
  - [`ew_forecast_solve`](@ref)
  - [`ew_forecast_history`](@ref)
  - [`forecast_return_units`](@ref)
"""
function return_forecast(rfe::ExpWeightedReturnForecast, rd::ReturnsResult,
                         csfm::CrossSectionalFactorModel)::ExpWeightedReturnForecastResult
    S = descriptor_scores(rfe.scores, rd, csfm)
    vs = forecast_idiosyncratic_variances(csfm)
    fwd = forward_mean_returns(forecast_idiosyncratic_returns(csfm), rfe.horizon, rfe.lag)
    emsk = return_forecast_weights(rd)
    y = forecast_unit_target(rfe.unit, fwd, vs)
    W = ew_forecast_weights(rfe.unit, emsk, vs)
    valid = ew_forecast_valid(S, fwd, vs, emsk)
    T = size(emsk, 1)
    K = size(S, 3)
    gap = rfe.lag + rfe.horizon - 1
    Tf = promote_type(float(real(eltype(y))), float(real(eltype(W))))
    A = zeros(Tf, K, K)
    c = zeros(Tf, K)
    coefs = fill(Tf(NaN), max(T - gap, 0), K)
    coef = fill(Tf(NaN), K)
    n = 0
    for t in 1:(T - gap)
        if any(view(valid, t, :))
            Sb, yb, wb = ew_forecast_design(S, y, W, t, valid)
            ew_forecast_accumulate!(A, c, Sb, yb, wb, rfe.decay, rfe.normalise)
            n += 1
            coef = ew_forecast_solve(A, c, rfe.ridge, t)
        end
        if n >= rfe.min_obs
            coefs[t, :] = coef
        end
    end
    hist = forecast_return_units(rfe.unit, rfe.scale .* ew_forecast_history(S, coefs, gap),
                                 vs)
    return ExpWeightedReturnForecastResult(; mu = hist[end, :], hist = hist, coef = coef,
                                           A = A, c = c, n = n)
end

export ExpWeightedReturnForecast, ExpWeightedReturnForecastResult
