"""
    target_forecast_folds(cv::PrequentialCalibration, tgt::LinearModel{@NamedTuple{}}) -> Bool
    target_forecast_folds(cv, tgt) -> Bool

Answer whether a [`TargetReturnForecast`](@ref) fits as a fold over its observations in order, from a state that it carries.

Under [`PrequentialCalibration`](@ref), observation `t` reads the model fitted on the targets that had matured at it, and the calibration reads each prediction once, at the maturity of its observation. A [`LinearModel`](@ref) with no keyword argument adds each matured sample to its normal equations, so its fit is a function of a carried state and the new observation. So [`return_forecast_step`](@ref) runs the fit, and the batch fit and the carry fold of a [`CrossSectionalFactorPrior`](@ref) run the same code. Any other pair answers `false`: a cross-validation estimator splits every sample again, and another regression target fits every matured sample again.

# Arguments

  - `cv`: The rule of the calibration, `rfe.cv`.
  - `tgt`: The regression target, `rfe.tgt`.

# Returns

  - `flag::Bool`: Whether the member fits as a fold.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`return_forecast_step`](@ref)
  - [`folds_forecast_rows`](@ref)
"""
function target_forecast_folds(::PrequentialCalibration, ::LinearModel{@NamedTuple{}})::Bool
    return true
end
function target_forecast_folds(::Any, ::Any)::Bool
    return false
end
function folds_forecast_rows(rfe::TargetReturnForecast)::Bool
    return target_forecast_folds(rfe.cv, rfe.tgt)
end
function forecast_target_gap(rfe::TargetReturnForecast)::Integer
    return rfe.lag + rfe.horizon - 1
end
function lookback(rfe::TargetReturnForecast)::Option{<:Integer}
    return lookback(rfe.scores.descriptors)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the scores and the target that a [`TargetReturnForecast`](@ref) fits, on one observation axis.

This type is an implementation detail and is not intended for direct use. [`target_forecast_design`](@ref) makes it, and [`return_forecast_step`](@ref) and [`return_forecast`](@ref) read it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`target_forecast_design`](@ref)
  - [`return_forecast_step`](@ref)
  - [`AlignedScores`](@ref)
"""
@concrete struct TargetDesign
    """
    `Sa`: The scores, with a slice of ones under `intercept`, `assets × descriptors × observations`.
    """
    Sa
    """
    `fwd`: The forward mean idiosyncratic return, `observations × assets`.
    """
    fwd
    """
    `y`: The target of the fit, `observations × assets`.
    """
    y
    """
    `vs`: The idiosyncratic variances, `observations × assets`, or `nothing`.
    """
    vs
    """
    `w`: The cross-sectional weights, `observations × assets`.
    """
    w
    """
    `off`: The number of rows of the observation axis before the first row of the block.
    """
    off
end
"""
    target_forecast_design(rfe::TargetReturnForecast, P::PanelScores,
                           csfm::CrossSectionalFactorModel,
                           rows::AbstractUnitRange) -> TargetDesign

Build the scores and the target that a [`TargetReturnForecast`](@ref) fits, on one observation axis.

# Algorithm

 1. Neutralise the scores of `P` against the block with [`descriptor_neutralised_scores`](@ref). Read the idiosyncratic returns off the block, and the variances through [`target_forecast_variances`](@ref), which states when the member needs them.
 2. Put the scores and the two block histories on one observation axis through [`target_forecast_alignment`](@ref), which reads `whole_history`. Under `intercept`, append a slice of ones to the scores, so that the fit and the latest prediction read the constant.
 3. Take the forward mean target `fwd` through [`forward_mean_returns`](@ref). Convert it to the Forecast Unit through [`forecast_unit_target`](@ref), and pass it through `target_outlier` and then `target_scoring`, giving `y`.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `P`: The scores, the weights and the group labels, as [`descriptor_panel_scores`](@ref) makes the [`PanelScores`](@ref). The function can change `P.S` in place.
  - `csfm`: The factor-model block. It must carry the cross-sectional fit, and the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref).
  - `rows`: The rows of `P` that the block covers.

# Validation

  - The rules of [`descriptor_neutralised_scores`](@ref), of [`forecast_idiosyncratic_returns`](@ref) and of [`target_forecast_variances`](@ref).

# Returns

  - `d::TargetDesign`: The scores `Sa`, the forward mean target `fwd`, the target of the fit `y`, the idiosyncratic variances `vs` or `nothing`, and the weights `w`, on one observation axis. `off` is the number of rows of that axis before the first row of the block.

# Related

  - [`TargetDesign`](@ref)
  - [`PanelScores`](@ref)
  - [`return_forecast`](@ref)
  - [`return_forecast_step`](@ref)
  - [`target_forecast_alignment`](@ref)
"""
function target_forecast_design(rfe::TargetReturnForecast, P::PanelScores,
                                csfm::CrossSectionalFactorModel,
                                rows::AbstractUnitRange)::TargetDesign
    S = descriptor_neutralised_scores(rfe.scores, P, csfm, rows)
    al = target_forecast_alignment(rfe.whole_history, S,
                                   forecast_idiosyncratic_returns(csfm),
                                   target_forecast_variances(rfe.unit, csfm, rfe.calibrate),
                                   P.w, P.g, rows)
    Sa = if rfe.intercept
        cat(al.S, fill(one(eltype(al.S)), size(al.S, 1), 1, size(al.S, 3)); dims = 2)
    else
        al.S
    end
    fwd = forward_mean_returns(al.eps, rfe.horizon, rfe.lag)
    yo = cross_sectional_transform(rfe.target_outlier,
                                   forecast_unit_target(rfe.unit, fwd, al.vs); w = al.w,
                                   groups = al.groups)
    y = cross_sectional_transform(rfe.target_scoring, yo; w = al.w, groups = al.groups)
    return TargetDesign(Sa, fwd, y, al.vs, al.w, rfe.whole_history ? first(rows) - 1 : 0)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the running state of the calibration regression of a [`TargetReturnForecast`](@ref).

This type is an implementation detail and is not intended for direct use. [`target_forecast_calibration_state`](@ref) makes the empty record, and [`target_forecast_calibration_step`](@ref) advances it. [`target_forecast_calibrated`](@ref) reads it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`target_forecast_calibration_state`](@ref)
  - [`target_forecast_calibration_step`](@ref)
  - [`target_forecast_calibrated`](@ref)
  - [`TargetRecursion`](@ref)
"""
@concrete struct CalibrationSums
    """
    `an`: The decayed sum of the squared predictions.
    """
    an
    """
    `ac`: The decayed sum of the products of prediction and target.
    """
    ac
    """
    `n`: The number of observations that advanced the regression.
    """
    n
    """
    `calib`: The calibration coefficient, or `NaN` before the first observation.
    """
    calib
end
"""
    target_forecast_calibration_state(Tf::Type) -> CalibrationSums

Return the empty state of the calibration regression of a [`TargetReturnForecast`](@ref).

# Arguments

  - `Tf`: The number type of the predictions and of the forward returns.

# Returns

  - `cs::CalibrationSums`: The two accumulators at zero, no observation, and a `NaN` coefficient.

# Related

  - [`CalibrationSums`](@ref)
  - [`target_forecast_calibration_step`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_calibration_state(Tf::Type)::CalibrationSums
    return CalibrationSums(zero(Tf), zero(Tf), 0, Tf(NaN))
end
"""
    target_forecast_calibration_step(cs::CalibrationSums, P::MatNum, fwd::MatNum,
                                     vs::MatNum, w::MatNum, decay::Real,
                                     t::Integer) -> CalibrationSums

Advance the calibration regression of a [`TargetReturnForecast`](@ref) by one observation. [`target_forecast_calibration`](@ref) states the steps.

# Arguments

  - `cs`: The [`CalibrationSums`](@ref) before the observation.
  - `P`: The uncalibrated prediction, `observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `decay`: The decay factor.
  - `t`: Index of the observation.

# Returns

  - `cs::CalibrationSums`: The state after the observation. An observation that states no slope returns `cs` as it is.

# Related

  - [`CalibrationSums`](@ref)
  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_calibration_design`](@ref)
"""
function target_forecast_calibration_step(cs::CalibrationSums, P::MatNum, fwd::MatNum,
                                          vs::MatNum, w::MatNum, decay::Real,
                                          t::Integer)::CalibrationSums
    Tf = promote_type(real(eltype(P)), real(eltype(fwd)))
    a, y, wv = target_forecast_calibration_design(P, fwd, vs, w, t)
    s = length(wv) < 2 ? zero(Tf) : sum(wv) / length(wv)
    if !(isfinite(s) && s > zero(s))
        return cs
    end
    u = wv ./ s
    on = LinearAlgebra.dot(u, a .* a)
    oc = LinearAlgebra.dot(u, a .* y)
    if !(isfinite(on) && isfinite(oc) && on > eps(Tf))
        return cs
    end
    an = decay * cs.an + (one(Tf) - decay) * on
    ac = decay * cs.ac + (one(Tf) - decay) * oc
    return CalibrationSums(an, ac, cs.n + 1, ac / (an + Tf(1e-6) * max(abs(an), eps(Tf))))
end
"""
    target_forecast_calibrated(calibrate::Bool, model, cs::CalibrationSums,
                               min_obs::Integer) -> Real

Return a calibration coefficient of a [`TargetReturnForecast`](@ref) from the state of its regression, or `NaN`.

A member that does not calibrate, and a member that fits no model, have no coefficient. The regression has none either while fewer than `min_obs` observations have advanced it.

# Arguments

  - `calibrate`: Whether the member calibrates.
  - `model`: The fitted model, or `nothing`.
  - `cs`: The [`CalibrationSums`](@ref) of the regression.
  - `min_obs`: The warm-up, in observations that advanced the regression.

# Returns

  - `calib::Real`: The coefficient, or `NaN`.

# Related

  - [`target_forecast_calibration_step`](@ref)
  - [`return_forecast_result`](@ref)
"""
function target_forecast_calibrated(calibrate::Bool, model, cs::CalibrationSums,
                                    min_obs::Integer)::Real
    return if calibrate && !isnothing(model) && cs.n >= min_obs
        cs.calib
    else
        oftype(cs.calib, NaN)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the fold state of a [`TargetReturnForecast`](@ref): the normal equations of the matured samples, the model they give, and the states of the calibration regressions.

This type is an implementation detail and is not intended for direct use. [`target_forecast_state`](@ref) makes the empty record and a copy of a carried record, and [`return_forecast_step`](@ref) advances it and returns it as the fold state. [`return_forecast_result`](@ref) reads it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`target_forecast_state`](@ref)
  - [`return_forecast_step`](@ref)
  - [`CalibrationSums`](@ref)
  - [`ForecastStepHistory`](@ref)
"""
@concrete struct TargetRecursion
    """
    `XtX`: The normal matrix of the matured samples, `columns × columns`.
    """
    XtX
    """
    `Xty`: The cross product of the matured samples, of length `columns`.
    """
    Xty
    """
    `n`: The number of matured samples.
    """
    n
    """
    `model`: The model that the normal equations give, or `nothing`.
    """
    model
    """
    `B`: The coefficients that predicted the observations whose target has not matured, one row each.
    """
    B
    """
    `cs`: The [`CalibrationSums`](@ref) of the calibration regression.
    """
    cs
    """
    `ocs`: The [`CalibrationSums`](@ref) of the regression of the orthogonal part, or `nothing` when the caller asks for no `κ⊥`.
    """
    ocs
end
"""
    target_forecast_state(fs::Nothing, Tf::Type, K::Integer, ortho::Bool) -> TargetRecursion
    target_forecast_state(fs::TargetRecursion, Tf::Type, K::Integer, ortho::Bool) -> TargetRecursion

Return the fold state that a step of a [`TargetReturnForecast`](@ref) starts from.

The state holds the normal equations `XtX` and `Xty` of the matured samples, their count `n`, the model `model` they give or `nothing`, and the coefficients `B` that predicted the observations whose target has not matured, one row each. It also holds the state `cs` of the calibration regression, and the state `ocs` of the regression of the orthogonal part, or `nothing` when the caller asks for no `κ⊥`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Nothing`: the empty state, in the number type `Tf`.
 2. `TargetRecursion`: a copy of the arrays of the carried state, so the state of an earlier step stays as it is. The number type is `Tf` promoted with the number type of the carried state.

# Arguments

  - `fs`: The carried fold state, a [`TargetRecursion`](@ref), or `nothing`.
  - `Tf`: The number type of the design of the step.
  - `K`: Number of columns of the design.
  - `ortho`: Whether the caller asks for `κ⊥`. The empty state reads it.

# Returns

  - `fs::TargetRecursion`: The state that the step advances.

# Related

  - [`TargetRecursion`](@ref)
  - [`return_forecast_step`](@ref)
  - [`NormalEquationsFit`](@ref)
"""
function target_forecast_state(::Nothing, Tf::Type, K::Integer,
                               ortho::Bool)::TargetRecursion
    return TargetRecursion(zeros(Tf, K, K), zeros(Tf, K), 0, nothing,
                           Matrix{Tf}(undef, 0, K), target_forecast_calibration_state(Tf),
                           ortho ? target_forecast_calibration_state(Tf) : nothing)
end
function target_forecast_state(fs::TargetRecursion, Tf::Type, ::Integer,
                               ::Bool)::TargetRecursion
    Tp = promote_type(Tf, eltype(fs.XtX))
    return TargetRecursion(Matrix{Tp}(fs.XtX), Vector{Tp}(fs.Xty), fs.n, fs.model,
                           Matrix{Tp}(fs.B), fs.cs, fs.ocs)
end
"""
    target_forecast_latest_row(vs::Nothing, t::Integer, N::Integer) -> Nothing
    target_forecast_latest_row(vs::MatNum, t::Integer, N::Integer) -> MatNum

Return the idiosyncratic variances that a [`TargetReturnForecast`](@ref) converts its forecast with at block row `t`, as a one-row matrix.

The forecast of a block reads the last row of the variances of that block, as [`target_forecast_latest_variances`](@ref) states. A row before the block has no variance, so it reads `NaN`.

# Arguments

  - `vs`: Idiosyncratic variance history of the block, `observations × assets`, or `nothing`.
  - `t`: Index of the row of the block. It is below one for a row before the block.
  - `N`: The number of assets.

# Returns

  - `vs::Option{<:MatNum}`: Row `t` as a `1 × assets` matrix, or `nothing`.

# Related

  - [`target_forecast_latest_variances`](@ref)
  - [`return_forecast_step`](@ref)
"""
function target_forecast_latest_row(::Nothing, ::Integer, ::Integer)::Nothing
    return nothing
end
function target_forecast_latest_row(vs::MatNum, t::Integer, N::Integer)::MatNum
    return if t >= 1
        vs[t:t, :]
    else
        fill(real(eltype(vs))(NaN), 1, N)
    end
end
"""
    target_forecast_matured!(Pm::MatNum, p::VecNum, cs::CalibrationSums, Sf::MatNum,
                             ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum, w::MatNum,
                             vs::Nothing, rfe::TargetReturnForecast, s::Integer) -> CalibrationSums
    target_forecast_matured!(Pm::MatNum, p::VecNum, cs::CalibrationSums, Sf::MatNum,
                             ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum, w::MatNum,
                             vs::MatNum, rfe::TargetReturnForecast, s::Integer) -> CalibrationSums

Advance the calibration regression of a [`TargetReturnForecast`](@ref) by the observation `s`, whose target matures at the step.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Nothing`: the member reads no variance, so it does not calibrate. Return `cs` as it is.
 2. A variance history, and the member calibrates: predict each valid sample of observation `s` with the coefficients that predicted it, row `s` of `B`, into `p`. A sample that is not valid reads `NaN`. Convert `p` to return units through [`forecast_return_units`](@ref) into row `s` of `Pm`, and advance `cs` through [`target_forecast_calibration_step`](@ref). A member that does not calibrate returns `cs` as it is.

# Arguments

  - `Pm`: The uncalibrated prediction of the observations that mature, in return units. The function changes row `s`.
  - `p`: A buffer of one prediction per asset. The function changes it.
  - `cs`: The state of the calibration regression.
  - `Sf`: The flattened design.
  - `ok`: The mask of the valid samples.
  - `B`: The coefficients that predicted each observation of the step, one row each.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `vs`: Idiosyncratic variance history, or `nothing`.
  - `rfe`: Target Return Forecast Estimator.
  - `s`: Index of the observation that matures.

# Returns

  - `cs::CalibrationSums`: The state of the calibration regression after the observation.

# Related

  - [`CalibrationSums`](@ref)
  - [`return_forecast_step`](@ref)
  - [`target_forecast_prequential`](@ref)
"""
function target_forecast_matured!(::MatNum, ::VecNum, cs::CalibrationSums, ::MatNum,
                                  ::AbstractVector{Bool}, ::MatNum, ::MatNum, ::MatNum,
                                  ::Nothing, ::TargetReturnForecast,
                                  ::Integer)::CalibrationSums
    return cs
end
function target_forecast_matured!(Pm::MatNum, p::VecNum, cs::CalibrationSums, Sf::MatNum,
                                  ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum,
                                  w::MatNum, vs::MatNum, rfe::TargetReturnForecast,
                                  s::Integer)::CalibrationSums
    if !rfe.calibrate
        return cs
    end
    # A contiguous copy of the coefficients, so the inner product runs as in the batch rule.
    b = B[s, :]
    js = target_forecast_row(s, length(p))
    for (i, j) in enumerate(js)
        p[i] = ok[j] ? LinearAlgebra.dot(view(Sf, j, :), b) : eltype(p)(NaN)
    end
    Pm[s, :] = forecast_return_units(rfe.unit, permutedims(p), view(vs, s:s, :))
    return target_forecast_calibration_step(cs, Pm, fwd, vs, w, rfe.decay, s)
end
"""
    return_forecast_step(rfe::TargetReturnForecast, P::PanelScores,
                         csfm::CrossSectionalFactorModel, fs::Option{<:TargetRecursion},
                         cre::Option{<:AbstractCrossSectionalRegressionEstimator} = nothing) -> ForecastStepHistory

Fold the observations of `P` into the fit of a [`TargetReturnForecast`](@ref) under [`PrequentialCalibration`](@ref) and a [`LinearModel`](@ref) with no keyword argument, as [`target_forecast_folds`](@ref) answers.

The rule of [`PrequentialCalibration`](@ref) is a fold over the observations in order. The target of observation `t` matures at observation `t + g`, with `g = lag + horizon - 1`. At that observation its valid samples enter the normal equations through [`normal_equations_add!`](@ref), one sample at a time. Observation `t` reads the model of the samples that had matured at it, so its prediction is the forecast that the member publishes at it. The calibration regression reads that prediction once, at the maturity of observation `t`. So the state `fs` and the new observations give every value of the fit. The batch fit runs every observation from an empty state. A step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) runs the observations whose target has not matured, which the rows of `B` count, and the new observations after them, from the carried state. The two do the same arithmetic in the same order.

# Algorithm

 1. Build the scores, the target and the weights on one observation axis through [`target_forecast_design`](@ref). The block covers the last rows of `P`, as [`return_forecast_rows`](@ref) states. Flatten the observations whose target is known, all but the last `g`, through [`target_forecast_samples`](@ref).
 2. Start from a copy of `fs`, or from the empty state, through [`target_forecast_state`](@ref). The first rows of `P` are the observations that `fs` predicted, one for each row of its coefficients `B`.
 3. For each new observation `t`, in order:
     1. The target of observation `t - g` matures. Add its valid samples to the normal equations. When they add a sample, solve them again as a [`NormalEquationsFit`](@ref).
     2. Advance the calibration regression by observation `t - g` through [`target_forecast_matured!`](@ref), with the coefficients that predicted it.
     3. Record the coefficients of the model, or `NaN` before the first fit, as the coefficients that predict observation `t`.
     4. Predict observation `t` through [`target_forecast_latest`](@ref), and convert it to return units with the variances of its row through [`target_forecast_latest_row`](@ref). Multiply by `scale` and by the multiplier of [`target_forecast_multiplier`](@ref) under the coefficient of [`target_forecast_calibrated`](@ref), giving row `t` of the history.
 4. When `cre` is given and the member calibrates, split the prediction of each observation that matured through [`target_forecast_orthogonal_rows`](@ref), and advance the regression of `κ⊥` on the orthogonal parts.
 5. Keep the coefficients of the last `g` observations, whose target has not matured.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `P`: The scores, the weights and the group labels, as [`descriptor_panel_scores`](@ref) makes the [`PanelScores`](@ref). Its first rows are the observations that `fs` predicted and whose target has not matured, and the rest are new. The function can change `P.S` in place.
  - `csfm`: The factor-model block of the last rows of `P`. It must carry the cross-sectional fit, and the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref). Under `cre` it must carry the exposure history and the regression weight history.
  - `fs`: The fold state after the observations before the new ones, as this function returns it in a [`TargetRecursion`](@ref), or `nothing` for an empty state.
  - `cre`: Cross-Sectional Regression Estimator that splits each matured prediction for `κ⊥`, or `nothing` to fit no `κ⊥`.

# Validation

  - The rules of [`target_forecast_design`](@ref) and of [`return_forecast_rows`](@ref).
  - When `cre` is given, the rules of [`target_forecast_orthogonal_rows`](@ref).

# Returns

  - `r::ForecastStepHistory`: The history and the fold state.

      + `hist::MatNum`: The forecast that the member publishes at each new observation, `observations × assets`, on the rows of `P`. A row that `fs` predicted reads `NaN`.
      + `fs::TargetRecursion`: The fold state after the new observations.

# Related

  - [`ForecastStepHistory`](@ref)
  - [`TargetRecursion`](@ref)
  - [`PanelScores`](@ref)
  - [`return_forecast`](@ref)
  - [`return_forecast_result`](@ref)
  - [`folds_forecast_rows`](@ref)
  - [`target_forecast_prequential`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function return_forecast_step(rfe::TargetReturnForecast, P::PanelScores,
                              csfm::CrossSectionalFactorModel,
                              fs::Option{<:TargetRecursion},
                              cre::Option{<:AbstractCrossSectionalRegressionEstimator} = nothing)::ForecastStepHistory
    d = target_forecast_design(rfe, P, csfm, return_forecast_rows(size(P.S, 3), csfm))
    Sa = d.Sa
    N, K, T = size(Sa)
    g = forecast_target_gap(rfe)
    Sf, yf, ok = target_forecast_samples(Sa, d.y, d.w, max(T - g, 0))
    Tf = promote_type(real(eltype(Sf)), real(eltype(yf)))
    st = target_forecast_state(fs, Tf, K, !isnothing(cre))
    (; XtX, Xty, n, model, cs) = st
    p0 = size(st.B, 1)
    B = vcat(st.B, fill(eltype(st.B)(NaN), T - p0, K))
    # The uncalibrated prediction of each observation that matures, in return units.
    Pm = fill(Tf(NaN), max(T - g, 0), N)
    p = fill(Tf(NaN), N)
    # Each step of the loop carries the state of the regression into the next, so the rows
    # are not a comprehension. A row is a vector of numbers (#1979).
    rows = Vector{VecNum}(undef, T - p0)
    for t in (p0 + 1):T
        if t > g
            m = normal_equations_add!(XtX, Xty, Sf, yf, ok, target_forecast_row(t - g, N))
            n += m
            # The model solves the sums when it is built, and the step keeps the last model.
            model = m > 0 ? NormalEquationsFit(XtX, Xty, n) : model
            cs = target_forecast_matured!(Pm, p, cs, Sf, ok, B, d.fwd, d.w, d.vs, rfe,
                                          t - g)
        end
        if !isnothing(model)
            B[t, :] = StatsAPI.coef(model)
        end
        Pl = forecast_return_units(rfe.unit,
                                   target_forecast_latest(model, view(Sa, :, :, t:t)),
                                   target_forecast_latest_row(csfm.vs, t - d.off, N))
        calib = target_forecast_calibrated(rfe.calibrate, model, cs, rfe.min_obs)
        rows[t - p0] = vec(rfe.scale .* target_forecast_multiplier(rfe.calibrate, calib) .*
                           Pl)
    end
    ocs = st.ocs
    if !isnothing(cre) && !isnothing(ocs) && rfe.calibrate
        Q = target_forecast_orthogonal_rows(cre, Pm, csfm, d.off)
        for s in axes(Q, 1)
            ocs = target_forecast_calibration_step(ocs, Q, d.fwd, d.vs, d.w, rfe.decay, s)
        end
    end
    H = reduce(vcat, permutedims.(rows))
    hist = vcat(fill(eltype(H)(NaN), p0, N), H)
    return ForecastStepHistory(hist,
                               TargetRecursion(XtX, Xty, n, model,
                                               B[max(T - g + 1, 1):T, :], cs, ocs))
end
"""
    return_forecast_result(rfe::TargetReturnForecast, hist::MatNum,
                           fs::TargetRecursion) -> TargetReturnForecastResult

Build the Result of a [`TargetReturnForecast`](@ref) from the history and the fold state of [`return_forecast_step`](@ref).

The forecast is the last row of the history. The model is the model of the state. The calibration coefficient and `κ⊥` come from the states of their regressions through [`target_forecast_calibrated`](@ref). `κ⊥` is `nothing` when the state carries no regression of it.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `hist`: The history of the forecast, `observations × assets`.
  - `fs`: The fold state.

# Returns

  - `rf::TargetReturnForecastResult`: The fitted forecast, the model, the calibration coefficient and `κ⊥`.

# Related

  - [`return_forecast_step`](@ref)
  - [`TargetReturnForecastResult`](@ref)
"""
function return_forecast_result(rfe::TargetReturnForecast, hist::MatNum,
                                fs::TargetRecursion)::TargetReturnForecastResult
    model = fs.model
    ocalib = if isnothing(fs.ocs)
        nothing
    else
        target_forecast_calibrated(rfe.calibrate, model, fs.ocs, rfe.min_obs)
    end
    return TargetReturnForecastResult(; mu = hist[end, :], model = model,
                                      calib = target_forecast_calibrated(rfe.calibrate,
                                                                         model, fs.cs,
                                                                         rfe.min_obs),
                                      ocalib = ocalib)
end
