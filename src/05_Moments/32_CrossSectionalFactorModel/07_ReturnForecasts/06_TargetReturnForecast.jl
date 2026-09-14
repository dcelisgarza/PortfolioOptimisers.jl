"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast fitted by a regression target over every observation and asset at once.

The member turns its Descriptors into scores with the recipe in `scores`, and hands every `(observation, asset)` pair whose forward target has matured to a regression target as one sample. It is the member that admits a nonlinear combination of the Descriptors, because the combination is whatever the target fits, and the library's own [`LinearModel`](@ref) and [`GeneralisedLinearModel`](@ref) are two targets that work today.

The target of the fit is transformed cross-sectionally before the fit sees it, which is what keeps one extreme observation from setting the shape of the whole model. That transformation leaves the prediction in no particular unit, so the member calibrates it: one exponentially weighted scalar regression of the forward return on the prediction restores return units, and `cv` decides whether that regression reads in-sample or out-of-fold predictions.

The member computes no history. Its in-sample predictions are not a forecast, so `hist` on its Result is `nothing`.

`whole_history` states the window the fit trains over. Under `true` the block's idiosyncratic returns are placed into the rows they were fitted on, and every pair with a finite score, a finite target and a positive weight is one sample, so a signal row before the block whose forward window reaches into the block trains the model too. Under `false` the fit trains on the block's rows alone. The calibration reads the idiosyncratic variance at the signal row either way, so a row before the block enters the fit and not the calibration.

# Mathematical definition

```math
\\begin{align}
\\hat{y}_{t,i} &= \\operatorname{predict}\\left(\\mathcal{M}, \\boldsymbol{s}_{t,i}\\right)\\,, &
\\mathcal{M} &= \\operatorname{fit}\\left(\\texttt{tgt}, \\mathbf{S}, \\boldsymbol{y}\\right)\\,, \\\\
\\alpha_{i} &= \\kappa \\, \\hat{y}_{T,i}\\,, &
\\kappa &= \\frac{\\sum_{t} \\lambda^{-t} \\langle \\boldsymbol{u}_{t}, \\hat{\\boldsymbol{y}}_{t} \\odot \\boldsymbol{\\epsilon}_{t} \\rangle}{\\sum_{t} \\lambda^{-t} \\langle \\boldsymbol{u}_{t}, \\hat{\\boldsymbol{y}}_{t} \\odot \\hat{\\boldsymbol{y}}_{t} \\rangle}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{s}_{t,i}``: Descriptor scores of asset ``i`` at observation ``t``.
  - ``\\mathcal{M}``: the model the regression target fits on every valid sample.
  - ``\\boldsymbol{\\epsilon}_{t}``: forward mean idiosyncratic returns of observation ``t``.
  - ``\\boldsymbol{u}_{t}``: normalised calibration weights of observation ``t``.
  - ``\\kappa``: the calibration coefficient, written here without its ridge.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    TargetReturnForecast(; scores::DescriptorScores,
                         tgt::AbstractRegressionTarget = LinearModel(),
                         horizon::Integer = 1, lag::Integer = 1,
                         whole_history::Bool = true,
                         target_outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                         target_scoring::Option{<:AbstractCrossSectionalTransform} = nothing,
                         calibrate::Bool = true, scale::Real = 1.0,
                         half_life::Real = 20.0, decay::Real = half_life_decay(half_life),
                         min_obs::Integer = half_life_min_obs(half_life),
                         cv::Option{<:CrossValidationEstimator} = nothing,
                         unit::AbstractForecastUnit = IdiosyncraticReturnUnit()) -> TargetReturnForecast

Every keyword but `half_life` corresponds to a field. `half_life` is not a field: it fixes the defaults of `decay` and `min_obs` of the calibration, and a value passed for either of those is used as it stands. `min_obs = 1` calibrates from the first observation that states a slope.

## Validation

  - `0 < decay < 1`.
  - `min_obs >= 1`, `horizon >= 1` and `lag >= 1`.
  - `scale > 0` and is finite.

# Examples

```jldoctest
julia> ds = DescriptorScores(; descriptors = [Passthrough(; field = \"a\")]);

julia> TargetReturnForecast(; scores = ds, target_outlier = nothing, half_life = 2,
                            calibrate = false)
TargetReturnForecast
          scores ┼ DescriptorScores
                 │   descriptors ┼ 1-element Vector{Passthrough}
                 │               │ Passthrough ⋯
                 │    neutralise ┼ nothing
                 │           cre ┼ CrossSectionalLinearRegression
                 │               │         alg ┼ PseudoInverseFallback()
                 │               │   intercept ┴ Bool: false
                 │       outlier ┼ CrossSectionalWinsoriser
                 │               │    low ┼ Float64: 0.01
                 │               │   high ┴ Float64: 0.99
                 │       scoring ┼ CrossSectionalStandardiser
                 │               │   min_group_size ┼ Int64: 8
                 │               │             atol ┴ Float64: 1.0e-12
                 │         group ┴ nothing
             tgt ┼ LinearModel
                 │   kwargs ┴ @NamedTuple{}: NamedTuple()
         horizon ┼ Int64: 1
             lag ┼ Int64: 1
   whole_history ┼ Bool: true
  target_outlier ┼ nothing
  target_scoring ┼ nothing
       calibrate ┼ Bool: false
           scale ┼ Float64: 1.0
           decay ┼ Float64: 0.7071067811865476
         min_obs ┼ Int64: 2
              cv ┼ nothing
            unit ┴ IdiosyncraticReturnUnit()
```

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`TargetReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`DescriptorScores`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`CrossValidationEstimator`](@ref)
  - [`ExpWeightedReturnForecast`](@ref)
"""
@concrete struct TargetReturnForecast <: AbstractReturnForecastEstimator
    """
    $(field_dict[:rf_scores])
    """
    scores
    """
    $(field_dict[:retgt])
    """
    tgt
    """
    $(field_dict[:rf_horizon])
    """
    horizon
    """
    $(field_dict[:rf_lag])
    """
    lag
    """
    $(field_dict[:rf_whole_history])
    """
    whole_history
    """
    Cross-sectional transform applied to the target of the fit before it is scored, or `nothing` to skip the step.
    """
    target_outlier
    """
    Cross-sectional transform applied to the target of the fit after the outlier step, or `nothing` to skip the step. It is `nothing` by default, so the fit predicts the winsorised return itself.
    """
    target_scoring
    """
    Whether the prediction is calibrated back to return units by an exponentially weighted scalar regression of the forward return on it.
    """
    calibrate
    """
    $(field_dict[:rf_scale])
    """
    scale
    """
    $(field_dict[:decay]) It weighs the calibration alone.
    """
    decay
    """
    $(field_dict[:min_obs]) It counts the observations that advanced the calibration.
    """
    min_obs
    """
    Cross-validation estimator whose folds give the out of fold predictions the calibration reads, or `nothing` to calibrate on in-sample predictions.
    """
    cv
    """
    $(field_dict[:rf_unit])
    """
    unit
    function TargetReturnForecast(scores::DescriptorScores, tgt::AbstractRegressionTarget,
                                  horizon::Integer, lag::Integer, whole_history::Bool,
                                  target_outlier::Option{<:AbstractCrossSectionalTransform},
                                  target_scoring::Option{<:AbstractCrossSectionalTransform},
                                  calibrate::Bool, scale::Real, decay::Real,
                                  min_obs::Integer, cv::Option{<:CrossValidationEstimator},
                                  unit::AbstractForecastUnit)
        assert_nonempty_gt0_finite_val(horizon, :horizon)
        assert_nonempty_gt0_finite_val(lag, :lag)
        assert_finite(scale, :scale)
        assert_gt0(scale, :scale)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(scores), typeof(tgt), typeof(horizon), typeof(lag),
                   typeof(whole_history), typeof(target_outlier), typeof(target_scoring),
                   typeof(calibrate), typeof(scale), typeof(decay), typeof(min_obs),
                   typeof(cv), typeof(unit)}(scores, tgt, horizon, lag, whole_history,
                                             target_outlier, target_scoring, calibrate,
                                             scale, decay, min_obs, cv, unit)
    end
end
function TargetReturnForecast(; scores::DescriptorScores,
                              tgt::AbstractRegressionTarget = LinearModel(),
                              horizon::Integer = 1, lag::Integer = 1,
                              whole_history::Bool = true,
                              target_outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                              target_scoring::Option{<:AbstractCrossSectionalTransform} = nothing,
                              calibrate::Bool = true, scale::Real = 1.0,
                              half_life::Real = 20.0,
                              decay::Real = half_life_decay(half_life),
                              min_obs::Integer = half_life_min_obs(half_life),
                              cv::Option{<:CrossValidationEstimator} = nothing,
                              unit::AbstractForecastUnit = IdiosyncraticReturnUnit())::TargetReturnForecast
    return TargetReturnForecast(scores, tgt, horizon, lag, whole_history, target_outlier,
                                target_scoring, calibrate, scale, decay, min_obs, cv, unit)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`TargetReturnForecast`](@ref).

Beside the two reads [`AbstractReturnForecastResult`](@ref) states, it carries the fitted model and the calibration coefficient, so a reader inspects the combination the target found and the scale that put it back into return units. `hist` is `nothing`, because the member computes no history.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`TargetReturnForecast`](@ref)
  - [`return_forecast`](@ref)
"""
@concrete struct TargetReturnForecastResult <: AbstractReturnForecastResult
    """
    $(field_dict[:rf_mu])
    """
    mu
    """
    $(field_dict[:rf_hist])
    """
    hist
    """
    The model the regression target fitted, or `nothing` when no sample was valid.
    """
    model
    """
    The calibration coefficient, or `NaN` when the member did not calibrate or the calibration is in its warm-up.
    """
    calib
    function TargetReturnForecastResult(mu::VecNum, model, calib::Number)
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        return new{typeof(mu), Nothing, typeof(model), typeof(calib)}(mu, nothing, model,
                                                                      calib)
    end
end
function TargetReturnForecastResult(; mu::VecNum, model = nothing,
                                    calib::Number = NaN)::TargetReturnForecastResult
    return TargetReturnForecastResult(mu, model, calib)
end
"""
    target_forecast_variances(unit::IdiosyncraticReturnUnit,
                              csfm::CrossSectionalFactorModel,
                              calibrate::Bool) -> Option{<:MatNum}
    target_forecast_variances(unit::IdiosyncraticSharpeUnit,
                              csfm::CrossSectionalFactorModel, calibrate::Bool) -> MatNum

Return the idiosyncratic variance history a [`TargetReturnForecast`](@ref) reads, or `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticSharpeUnit`](@ref): the variances scale the target and the forecast, so they are read whatever the calibration does.
 2. [`IdiosyncraticReturnUnit`](@ref): the variances weigh the calibration alone, so they are read when `calibrate` is set and skipped when it is not.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `csfm`: The fitted factor-model block.
  - `calibrate`: Whether the member calibrates its prediction to return units.

# Validation

  - The rules of [`forecast_idiosyncratic_variances`](@ref), where the history is read.

# Returns

  - `vs::Option{<:MatNum}`: Idiosyncratic variances, or `nothing` when the member reads none.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`forecast_idiosyncratic_variances`](@ref)
  - [`AbstractForecastUnit`](@ref)
"""
function target_forecast_variances(::IdiosyncraticReturnUnit,
                                   csfm::CrossSectionalFactorModel,
                                   calibrate::Bool)::Option{<:MatNum}
    return calibrate ? forecast_idiosyncratic_variances(csfm) : nothing
end
function target_forecast_variances(::IdiosyncraticSharpeUnit,
                                   csfm::CrossSectionalFactorModel, ::Bool)::MatNum
    return forecast_idiosyncratic_variances(csfm)
end
"""
    target_forecast_samples(S::Arr3Num, y::MatNum, w::MatNum, nt::Integer) -> Tuple

Flatten the Descriptor scores and the target of a [`TargetReturnForecast`](@ref) into one sample per `(observation, asset)` pair.

The regression target of this member is not a cross-section: every pair of an observation whose forward target has matured is one sample, and the fit sees them all at once. The pairs are laid out observation by observation, so the flat index of the pair `(t, i)` is `(t - 1) * assets + i`.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `y`: The target in the Forecast Unit, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `nt`: Number of observations whose target has matured.

# Returns

  - `(Sf, yf, ok)::Tuple`: The design `nt · assets × descriptors`, the target of the same length, and the mask of the pairs that carry a positive weight, a finite target and a finite score for every Descriptor.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`descriptor_scores`](@ref)
"""
function target_forecast_samples(S::Arr3Num, y::MatNum, w::MatNum, nt::Integer)
    Tf = promote_type(real(eltype(S)), real(eltype(y)))
    N = size(S, 2)
    K = size(S, 3)
    Sf = Matrix{Tf}(undef, nt * N, K)
    yf = Vector{Tf}(undef, nt * N)
    ok = falses(nt * N)
    j = 0
    for t in 1:nt, i in 1:N
        j += 1
        good = w[t, i] > zero(w[t, i]) && isfinite(y[t, i])
        for k in 1:K
            Sf[j, k] = S[t, i, k]
            good &= isfinite(Sf[j, k])
        end
        yf[j] = y[t, i]
        ok[j] = good
    end
    return Sf, yf, ok
end
"""
    target_forecast_fit(rfe::TargetReturnForecast, Sf::MatNum, yf::VecNum,
                        ok::AbstractVector{Bool}) -> Option

Fit the regression target of a [`TargetReturnForecast`](@ref) on the valid samples.

A window that carries no valid sample fits nothing and answers `nothing`, which is the warm-up of this member: it forecasts `NaN` until one observation's target has matured.

# Arguments

  - `rfe`: Target Return Forecast Estimator. Its regression target is read as a field rather than passed as an abstract argument, which is what keeps the call site of the third-party fit concrete.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.

# Returns

  - `model::Option`: The fitted model, or `nothing` when no sample is valid.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
function target_forecast_fit(rfe::TargetReturnForecast, Sf::MatNum, yf::VecNum,
                             ok::AbstractVector{Bool})
    idx = findall(ok)
    return !isempty(idx) ? StatsAPI.fit(rfe.tgt, Sf[idx, :], yf[idx]) : nothing
end
"""
    target_forecast_uncalibrated(cv::Nothing, rfe::TargetReturnForecast, model,
                                 Sf::MatNum, yf::VecNum,
                                 ok::AbstractVector{Bool}) -> VecNum
    target_forecast_uncalibrated(cv::CrossValidationEstimator, rfe::TargetReturnForecast,
                                 model, Sf::MatNum, yf::VecNum,
                                 ok::AbstractVector{Bool}) -> VecNum

Predict the uncalibrated forecast of the samples a [`TargetReturnForecast`](@ref) trained on.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the fitted model predicts its own training samples, so the calibration runs in sample.
 2. A cross-validation estimator: the samples are split through [`Base.split`](@ref), a model is fitted on the training folds of each split and predicts its test fold, so the calibration runs out of fold and the slope is not read off predictions the model has already seen. A sample no split tests keeps its `NaN` and states nothing.

# Arguments

  - `cv`: Cross-validation estimator, or `nothing`. It is `rfe.cv`, passed apart so that the method Julia selects is the algorithm.
  - `rfe`: Target Return Forecast Estimator.
  - `model`: The model fitted on every valid sample.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.

# Validation

  - Under a cross-validation estimator, the valid samples number at least twice the split count [`n_splits`](@ref) reports, which is the smallest sample that leaves two per fold. Raises an `ArgumentError`.

# Returns

  - `p::VecNum`: The uncalibrated prediction of every sample, `NaN` where the sample is not valid.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`CrossValidationEstimator`](@ref)
"""
function target_forecast_uncalibrated(::Nothing, ::TargetReturnForecast, model, Sf::MatNum,
                                      ::VecNum, ok::AbstractVector{Bool})::VecNum
    Tf = real(eltype(Sf))
    p = fill(Tf(NaN), length(ok))
    idx = findall(ok)
    p[idx] = StatsAPI.predict(model, Sf[idx, :])
    return p
end
function target_forecast_uncalibrated(cv::CrossValidationEstimator,
                                      rfe::TargetReturnForecast, ::Any, Sf::MatNum,
                                      yf::VecNum, ok::AbstractVector{Bool})::VecNum
    Tf = real(eltype(Sf))
    idx = findall(ok)
    m = length(idx)
    rdx = ReturnsResult(; nx = ["sample"], X = zeros(Tf, m, 1))
    ns = n_splits(cv, rdx)
    @argcheck(m >= 2 * ns,
              ArgumentError("an out of fold calibration needs at least two samples per fold, so $(2 * ns) valid samples over $ns folds, got $m"))
    Sv = Sf[idx, :]
    yv = yf[idx]
    q = fill(Tf(NaN), m)
    fld = Base.split(cv, rdx)
    for k in eachindex(fld.train_idx)
        tr = fld.train_idx[k]
        te = fld.test_idx[k]
        q[te] = StatsAPI.predict(StatsAPI.fit(rfe.tgt, Sv[tr, :], yv[tr]), Sv[te, :])
    end
    p = fill(Tf(NaN), length(ok))
    p[idx] = q
    return p
end
"""
    target_forecast_scatter(p::VecNum, nt::Integer, N::Integer) -> Matrix{<:Real}

Read a flat sample vector of a [`TargetReturnForecast`](@ref) back as an `observations × assets` matrix.

It is the inverse of the layout [`target_forecast_samples`](@ref) writes, so the two functions state the flattening once each and no caller recomputes an index.

# Arguments

  - `p`: The flat vector, of length `nt · N`.
  - `nt`: Number of observations whose target has matured.
  - `N`: Number of assets.

# Returns

  - `P::Matrix{<:Real}`: The same values, `nt × N`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
"""
function target_forecast_scatter(p::VecNum, nt::Integer, N::Integer)::Matrix{<:Real}
    Tf = real(eltype(p))
    P = Matrix{Tf}(undef, nt, N)
    j = 0
    for t in 1:nt, i in 1:N
        j += 1
        P[t, i] = p[j]
    end
    return P
end
"""
    target_forecast_calibration_design(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                       t::Integer) -> Tuple

Gather the calibration sample of one observation of a [`TargetReturnForecast`](@ref).

An asset enters when it carries a positive cross-sectional weight, a finite idiosyncratic variance, a finite uncalibrated prediction and a finite forward return. Its weight is the cross-sectional weight divided by its idiosyncratic variance, which is the same generalised least squares weighting the exponentially weighted member uses in the return unit.

# Arguments

  - `P`: The uncalibrated prediction, `matured observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `t`: Index of the observation.

# Returns

  - `(a, y, wv)::Tuple`: The prediction, the forward return and the weight of every entering asset.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_calibration_design(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                            t::Integer)
    Tf = promote_type(real(eltype(P)), real(eltype(fwd)), real(eltype(vs)), real(eltype(w)))
    idx = Int[]
    for i in axes(P, 2)
        if w[t, i] > zero(w[t, i]) &&
           isfinite(vs[t, i]) &&
           isfinite(P[t, i]) &&
           isfinite(fwd[t, i])
            push!(idx, i)
        end
    end
    return Tf.(view(P, t, idx)), Tf.(view(fwd, t, idx)),
           Tf.(view(w, t, idx)) ./ Tf.(view(vs, t, idx))
end
"""
    target_forecast_calibration(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                decay::Real, min_obs::Integer) -> Real

Fit the scalar calibration coefficient of a [`TargetReturnForecast`](@ref).

The regression target of this member is winsorised, and may be standardised, so its prediction is not in return units. One exponentially weighted scalar regression of the forward return on that prediction restores the units, and its slope is the coefficient the member publishes with.

# Algorithm

 1. For each observation in turn, gather its calibration sample through [`target_forecast_calibration_design`](@ref). An observation with fewer than two entering assets states no slope and is skipped.
 2. Divide the weights by their mean over the entering assets, and take the weighted inner products of the prediction with itself and with the forward return. An observation whose products are not finite, or whose normal product is at or below `eps`, is skipped.
 3. Advance the two exponentially weighted accumulators by those products, and read the slope off them with a relative ridge of `1e-6` on the normal accumulator.
 4. Return the slope when `min_obs` observations have advanced the accumulators, and `NaN` before then.

# Arguments

  - `P`: The uncalibrated prediction, `matured observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `decay`: The decay factor.
  - `min_obs`: The warm-up, in observations that advanced the accumulators.

# Returns

  - `calib::Real`: The calibration coefficient, or `NaN` while the fit is in its warm-up.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration_design`](@ref)
  - [`return_forecast`](@ref)
"""
function target_forecast_calibration(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                     decay::Real, min_obs::Integer)::Real
    Tf = promote_type(real(eltype(P)), real(eltype(fwd)))
    an = zero(Tf)
    ac = zero(Tf)
    calib = Tf(NaN)
    n = 0
    for t in axes(P, 1)
        a, y, wv = target_forecast_calibration_design(P, fwd, vs, w, t)
        s = length(wv) < 2 ? zero(Tf) : sum(wv) / length(wv)
        if !(isfinite(s) && s > zero(s))
            continue
        end
        u = wv ./ s
        on = LinearAlgebra.dot(u, a .* a)
        oc = LinearAlgebra.dot(u, a .* y)
        if !(isfinite(on) && isfinite(oc) && on > eps(Tf))
            continue
        end
        an = decay * an + (one(Tf) - decay) * on
        ac = decay * ac + (one(Tf) - decay) * oc
        n += 1
        calib = ac / (an + Tf(1e-6) * max(abs(an), eps(Tf)))
    end
    return n >= min_obs ? calib : Tf(NaN)
end
"""
    target_forecast_latest(model::Nothing, S::Arr3Num) -> Matrix{<:Real}
    target_forecast_latest(model, S::Arr3Num) -> Matrix{<:Real}

Predict the uncalibrated forecast of the latest observation of a [`TargetReturnForecast`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: nothing was fitted, so every asset reads `NaN`.
 2. A fitted model: the assets whose scores are all finite are predicted together, and the rest read `NaN`.

# Arguments

  - `model`: The fitted model, or `nothing`.
  - `S`: The Descriptor scores, `observations × assets × descriptors`.

# Returns

  - `P::Matrix{<:Real}`: The uncalibrated forecast of the last observation, `1 × assets`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`return_forecast`](@ref)
"""
function target_forecast_latest(::Nothing, S::Arr3Num)::Matrix{<:Real}
    Tf = real(eltype(S))
    return fill(Tf(NaN), 1, size(S, 2))
end
function target_forecast_latest(model, S::Arr3Num)::Matrix{<:Real}
    Tf = real(eltype(S))
    A = Tf.(view(S, size(S, 1), :, :))
    P = fill(Tf(NaN), 1, size(A, 1))
    idx = Int[]
    for i in axes(A, 1)
        ok = true
        for k in axes(A, 2)
            ok &= isfinite(A[i, k])
        end
        if ok
            push!(idx, i)
        end
    end
    if !isempty(idx)
        P[1, idx] = StatsAPI.predict(model, A[idx, :])
    end
    return P
end
"""
    target_forecast_latest_variances(vs::Nothing) -> Nothing
    target_forecast_latest_variances(vs::MatNum) -> MatNum

Return the idiosyncratic variances of the latest observation, as a one-row matrix.

[`forecast_return_units`](@ref) converts a whole history, and this member converts one row of one, so the row is kept a matrix rather than read as a vector.

# Arguments

  - `vs`: Idiosyncratic variance history, `observations × assets`, or `nothing`.

# Returns

  - `vs::Option{<:MatNum}`: The last row as a `1 × assets` matrix, or `nothing`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`forecast_return_units`](@ref)
"""
function target_forecast_latest_variances(::Nothing)::Nothing
    return nothing
end
function target_forecast_latest_variances(vs::MatNum)::MatNum
    return vs[size(vs, 1):size(vs, 1), :]
end
"""
    target_forecast_alignment(whole_history::Bool, S::Arr3Num, eps::MatNum,
                              vs::Option{<:MatNum}, w::MatNum,
                              groups::Option{<:AbstractMatrix{<:Integer}},
                              rows::AbstractUnitRange) -> NamedTuple

Put the scores of a [`TargetReturnForecast`](@ref) and the histories it fits on one observation axis.

# Algorithm

 1. `whole_history` is set: the scores, the weights and the group labels stay on the carrier's axis, and the two block histories are placed into the block's rows through [`return_forecast_pad`](@ref). A row before the block carries a `NaN` idiosyncratic return, so it states a target only where its forward window reaches into the block.
 2. `whole_history` is not set: the scores, the weights and the group labels are cut to the block's rows through [`return_forecast_cut`](@ref), and the two block histories are already on them.

# Arguments

  - $(arg_dict[:rf_whole_history])
  - `S`: The Descriptor scores, `observations × assets × descriptors`, on the carrier's axis.
  - `eps`: Idiosyncratic returns of the block, `observations × assets`.
  - `vs`: Idiosyncratic variance history of the block, `observations × assets`, or `nothing`.
  - `w`: Cross-sectional weights, `observations × assets`, on the carrier's axis.
  - `groups`: Group label matrix on the carrier's axis, or `nothing`.
  - `rows`: The rows of the carrier the block lives on.

# Returns

  - `(S, eps, vs, w, groups)::NamedTuple`: The same five, on one observation axis.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`return_forecast_pad`](@ref)
  - [`return_forecast_cut`](@ref)
  - [`return_forecast_rows`](@ref)
"""
function target_forecast_alignment(whole_history::Bool, S::Arr3Num, eps::MatNum,
                                   vs::Option{<:MatNum}, w::MatNum,
                                   groups::Option{<:AbstractMatrix{<:Integer}},
                                   rows::AbstractUnitRange)
    if whole_history
        T = size(S, 1)
        return (; S = S, eps = return_forecast_pad(eps, rows, T),
                vs = return_forecast_pad(vs, rows, T), w = w, groups = groups)
    end
    return (; S = return_forecast_cut(S, rows), eps = eps, vs = vs,
            w = return_forecast_cut(w, rows), groups = return_forecast_cut(groups, rows))
end
"""
    return_forecast(rfe::TargetReturnForecast, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> TargetReturnForecastResult

Fit a Return Forecast with a regression target over every observation and asset at once.

# Algorithm

 1. Compute the Descriptor scores over the whole carrier through [`descriptor_scores`](@ref), and read the idiosyncratic returns off the block. The variances are read through [`target_forecast_variances`](@ref), which states when the member needs them.
 2. Put the scores and the two block histories on one observation axis through [`target_forecast_alignment`](@ref), which `whole_history` chooses.
 3. Take the forward mean target through [`forward_mean_returns`](@ref), convert it to the Forecast Unit through [`forecast_unit_target`](@ref), and pass it through the outlier slot and then the scoring slot.
 4. Flatten the observations whose target has matured, which are all but the last `lag + horizon - 1`, into one sample per `(observation, asset)` pair through [`target_forecast_samples`](@ref), and fit the regression target on the valid samples.
 5. When the member calibrates, predict those samples through [`target_forecast_uncalibrated`](@ref) and fit the calibration coefficient through [`target_forecast_calibration`](@ref). It reads the idiosyncratic variance at the signal row, so a row before the block enters the fit and not the calibration.
 6. Predict the latest observation through [`target_forecast_latest`](@ref), multiply by the calibration coefficient and by `scale`, and convert the row to return units.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. It must carry the cross-sectional fit, its histories state the block's rows, and it must carry the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref).

# Validation

  - The rules of [`descriptor_scores`](@ref), of [`forecast_idiosyncratic_returns`](@ref), of [`target_forecast_variances`](@ref) and of [`target_forecast_uncalibrated`](@ref).

# Returns

  - `rf::TargetReturnForecastResult`: The fitted forecast, the model and the calibration coefficient.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`TargetReturnForecastResult`](@ref)
  - [`target_forecast_alignment`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_latest`](@ref)
  - [`forecast_return_units`](@ref)
"""
function return_forecast(rfe::TargetReturnForecast, rd::ReturnsResult,
                         csfm::CrossSectionalFactorModel)::TargetReturnForecastResult
    ds = rfe.scores
    (; S, rows) = descriptor_scores(ds, rd, csfm)
    al = target_forecast_alignment(rfe.whole_history, S,
                                   forecast_idiosyncratic_returns(csfm),
                                   target_forecast_variances(rfe.unit, csfm, rfe.calibrate),
                                   return_forecast_weights(rd),
                                   exposure_group_labels(rd, ds.group), rows)
    Sa = al.S
    vs = al.vs
    emsk = al.w
    groups = al.groups
    fwd = forward_mean_returns(al.eps, rfe.horizon, rfe.lag)
    y = exposure_transform(rfe.target_scoring,
                           exposure_transform(rfe.target_outlier,
                                              forecast_unit_target(rfe.unit, fwd, vs), emsk,
                                              groups), emsk, groups)
    nt = max(size(Sa, 1) - (rfe.lag + rfe.horizon - 1), 0)
    Sf, yf, ok = target_forecast_samples(Sa, y, emsk, nt)
    model = target_forecast_fit(rfe, Sf, yf, ok)
    calib = target_forecast_coefficient(rfe, model, Sf, yf, ok, fwd, vs, emsk, nt)
    P = forecast_return_units(rfe.unit, target_forecast_latest(model, Sa),
                              target_forecast_latest_variances(vs))
    return TargetReturnForecastResult(;
                                      mu = vec(rfe.scale .*
                                               target_forecast_multiplier(rfe.calibrate,
                                                                          calib) .* P),
                                      model = model, calib = calib)
end
"""
    target_forecast_multiplier(calibrate::Bool, calib::Number) -> Number

Return the multiplier a [`TargetReturnForecast`](@ref) applies to its uncalibrated prediction.

A member that does not calibrate publishes the prediction as it stands, so its multiplier is one and the `NaN` its Result carries states that no coefficient was fitted rather than that the forecast is unavailable.

# Arguments

  - `calibrate`: Whether the member calibrates.
  - `calib`: The calibration coefficient.

# Returns

  - `m::Real`: The multiplier.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_multiplier(calibrate::Bool, calib::Number)::Number
    return calibrate ? calib : one(calib)
end
"""
    target_forecast_coefficient(rfe::TargetReturnForecast, model, Sf::MatNum, yf::VecNum,
                                ok::AbstractVector{Bool}, fwd::MatNum,
                                vs::Option{<:MatNum}, w::MatNum, nt::Integer) -> Real

Return the calibration coefficient of a [`TargetReturnForecast`](@ref), or `NaN`.

This is the one place that states when the calibration runs at all: a member that does not calibrate, and a member that fitted no model, each answer `NaN` without predicting anything.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `model`: The fitted model, or `nothing`.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, or `nothing`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `nt`: Number of observations whose target has matured.

# Returns

  - `calib::Real`: The calibration coefficient, or `NaN`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
"""
function target_forecast_coefficient(rfe::TargetReturnForecast, model, Sf::MatNum,
                                     yf::VecNum, ok::AbstractVector{Bool}, fwd::MatNum,
                                     vs::Option{<:MatNum}, w::MatNum, nt::Integer)::Real
    if !rfe.calibrate || isnothing(model) || isnothing(vs)
        return NaN
    end
    p = target_forecast_uncalibrated(rfe.cv, rfe, model, Sf, yf, ok)
    P = forecast_return_units(rfe.unit, target_forecast_scatter(p, nt, size(w, 2)),
                              vs[1:nt, :])
    return target_forecast_calibration(P, fwd, vs, w, rfe.decay, rfe.min_obs)
end

export TargetReturnForecast, TargetReturnForecastResult
