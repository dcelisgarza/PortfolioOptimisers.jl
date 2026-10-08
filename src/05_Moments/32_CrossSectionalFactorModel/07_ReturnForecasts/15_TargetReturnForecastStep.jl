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
    target_forecast_design(rfe::TargetReturnForecast, P::NamedTuple,
                           csfm::CrossSectionalFactorModel,
                           rows::AbstractUnitRange) -> NamedTuple

Build the scores and the target that a [`TargetReturnForecast`](@ref) fits, on one observation axis.

# Algorithm

 1. Neutralise the scores of `P` against the block with [`descriptor_neutralised_scores`](@ref). Read the idiosyncratic returns off the block, and the variances through [`target_forecast_variances`](@ref), which states when the member needs them.
 2. Put the scores and the two block histories on one observation axis through [`target_forecast_alignment`](@ref), which reads `whole_history`. Under `intercept`, append a slice of ones to the scores, so that the fit and the latest prediction read the constant.
 3. Take the forward mean target `fwd` through [`forward_mean_returns`](@ref). Convert it to the Forecast Unit through [`forecast_unit_target`](@ref), and pass it through `target_outlier` and then `target_scoring`, giving `y`.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `P`: The scores, the weights and the group labels, `(; S, w, g)`, as [`descriptor_panel_scores`](@ref) states them. The function can change `P.S` in place.
  - `csfm`: The factor-model block. It must carry the cross-sectional fit, and the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref).
  - `rows`: The rows of `P` that the block covers.

# Validation

  - The rules of [`descriptor_neutralised_scores`](@ref), of [`forecast_idiosyncratic_returns`](@ref) and of [`target_forecast_variances`](@ref).

# Returns

  - `(Sa, fwd, y, vs, w, off)::NamedTuple`: The scores, the forward mean target, the target of the fit, the idiosyncratic variances or `nothing`, and the weights, on one observation axis. `off` is the number of rows of that axis before the first row of the block.

# Related

  - [`return_forecast`](@ref)
  - [`return_forecast_step`](@ref)
  - [`target_forecast_alignment`](@ref)
"""
function target_forecast_design(rfe::TargetReturnForecast, P::NamedTuple,
                                csfm::CrossSectionalFactorModel, rows::AbstractUnitRange)
    S = descriptor_neutralised_scores(rfe.scores, P, csfm, rows)
    al = target_forecast_alignment(rfe.whole_history, S,
                                   forecast_idiosyncratic_returns(csfm),
                                   target_forecast_variances(rfe.unit, csfm, rfe.calibrate),
                                   P.w, P.g, rows)
    Sa = if rfe.intercept
        cat(al.S, fill(one(eltype(al.S)), size(al.S, 1), size(al.S, 2), 1); dims = 3)
    else
        al.S
    end
    fwd = forward_mean_returns(al.eps, rfe.horizon, rfe.lag)
    y = exposure_transform(rfe.target_scoring,
                           exposure_transform(rfe.target_outlier,
                                              forecast_unit_target(rfe.unit, fwd, al.vs),
                                              al.w, al.groups), al.w, al.groups)
    return (; Sa = Sa, fwd = fwd, y = y, vs = al.vs, w = al.w,
            off = rfe.whole_history ? first(rows) - 1 : 0)
end
"""
    target_forecast_calibration_state(Tf::Type) -> NamedTuple

Return the empty state of the calibration regression of a [`TargetReturnForecast`](@ref).

# Arguments

  - `Tf`: The number type of the predictions and of the forward returns.

# Returns

  - `(an, ac, n, calib)::NamedTuple`: The two accumulators at zero, no observation, and a `NaN` coefficient.

# Related

  - [`target_forecast_calibration_step`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_calibration_state(Tf::Type)::NamedTuple
    return (; an = zero(Tf), ac = zero(Tf), n = 0, calib = Tf(NaN))
end
"""
    target_forecast_calibration_step(cs::NamedTuple, P::MatNum, fwd::MatNum, vs::MatNum,
                                     w::MatNum, decay::Real, t::Integer) -> NamedTuple

Advance the calibration regression of a [`TargetReturnForecast`](@ref) by one observation. [`target_forecast_calibration`](@ref) states the steps.

# Arguments

  - `cs`: The state `(; an, ac, n, calib)` before the observation.
  - `P`: The uncalibrated prediction, `observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `decay`: The decay factor.
  - `t`: Index of the observation.

# Returns

  - `cs::NamedTuple`: The state after the observation. An observation that states no slope returns `cs` as it is.

# Related

  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_calibration_design`](@ref)
"""
function target_forecast_calibration_step(cs::NamedTuple, P::MatNum, fwd::MatNum,
                                          vs::MatNum, w::MatNum, decay::Real,
                                          t::Integer)::NamedTuple
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
    return (; an = an, ac = ac, n = cs.n + 1,
            calib = ac / (an + Tf(1e-6) * max(abs(an), eps(Tf))))
end
"""
    target_forecast_calibrated(calibrate::Bool, model, cs::NamedTuple,
                               min_obs::Integer) -> Real

Return a calibration coefficient of a [`TargetReturnForecast`](@ref) from the state of its regression, or `NaN`.

A member that does not calibrate, and a member that fits no model, have no coefficient. The regression has none either while fewer than `min_obs` observations have advanced it.

# Arguments

  - `calibrate`: Whether the member calibrates.
  - `model`: The fitted model, or `nothing`.
  - `cs`: The state `(; an, ac, n, calib)` of the regression.
  - `min_obs`: The warm-up, in observations that advanced the regression.

# Returns

  - `calib::Real`: The coefficient, or `NaN`.

# Related

  - [`target_forecast_calibration_step`](@ref)
  - [`return_forecast_result`](@ref)
"""
function target_forecast_calibrated(calibrate::Bool, model, cs::NamedTuple,
                                    min_obs::Integer)::Real
    return if calibrate && !isnothing(model) && cs.n >= min_obs
        cs.calib
    else
        oftype(cs.calib, NaN)
    end
end
"""
    target_forecast_state(fs::Nothing, Tf::Type, K::Integer, ortho::Bool) -> NamedTuple
    target_forecast_state(fs::NamedTuple, Tf::Type, K::Integer, ortho::Bool) -> NamedTuple

Return the fold state that a step of a [`TargetReturnForecast`](@ref) starts from.

The state holds the normal equations `XtX` and `Xty` of the matured samples, their count `n`, the model `model` they give or `nothing`, and the coefficients `B` that predicted the observations whose target has not matured, one row each. It also holds the state `cs` of the calibration regression, and the state `ocs` of the regression of the orthogonal part, or `nothing` when the caller asks for no `κ⊥`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Nothing`: the empty state, in the number type `Tf`.
 2. `NamedTuple`: a copy of the arrays of the carried state, so the state of an earlier step stays as it is. The number type is `Tf` promoted with the number type of the carried state.

# Arguments

  - `fs`: The carried fold state, or `nothing`.
  - `Tf`: The number type of the design of the step.
  - `K`: Number of columns of the design.
  - `ortho`: Whether the caller asks for `κ⊥`. The empty state reads it.

# Returns

  - `fs::NamedTuple`: The state `(; XtX, Xty, n, model, B, cs, ocs)` that the step advances.

# Related

  - [`return_forecast_step`](@ref)
  - [`NormalEquationsFit`](@ref)
"""
function target_forecast_state(::Nothing, Tf::Type, K::Integer, ortho::Bool)::NamedTuple
    return (; XtX = zeros(Tf, K, K), Xty = zeros(Tf, K), n = 0, model = nothing,
            B = Matrix{Tf}(undef, 0, K), cs = target_forecast_calibration_state(Tf),
            ocs = ortho ? target_forecast_calibration_state(Tf) : nothing)
end
function target_forecast_state(fs::NamedTuple, Tf::Type, ::Integer, ::Bool)::NamedTuple
    Tp = promote_type(Tf, eltype(fs.XtX))
    return (; XtX = Matrix{Tp}(fs.XtX), Xty = Vector{Tp}(fs.Xty), n = fs.n,
            model = fs.model, B = Matrix{Tp}(fs.B), cs = fs.cs, ocs = fs.ocs)
end
"""
    target_forecast_latest_row(vs::Nothing, t::Integer, N::Integer) -> Nothing
    target_forecast_latest_row(vs::MatNum, t::Integer, N::Integer) -> MatNum

Return the idiosyncratic variances that the read-out of a [`TargetReturnForecast`](@ref) at block row `t` converts with, as a one-row matrix.

The read-out of a block reads the last row of the variances of that block, as [`target_forecast_latest_variances`](@ref) states. A row before the block has no variance, so it reads `NaN`.

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
    target_forecast_matured!(Pm::MatNum, p::VecNum, cs::NamedTuple, Sf::MatNum,
                             ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum, w::MatNum,
                             vs::Nothing, rfe::TargetReturnForecast, s::Integer) -> NamedTuple
    target_forecast_matured!(Pm::MatNum, p::VecNum, cs::NamedTuple, Sf::MatNum,
                             ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum, w::MatNum,
                             vs::MatNum, rfe::TargetReturnForecast, s::Integer) -> NamedTuple

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

  - `cs::NamedTuple`: The state of the calibration regression after the observation.

# Related

  - [`return_forecast_step`](@ref)
  - [`target_forecast_prequential`](@ref)
"""
function target_forecast_matured!(::MatNum, ::VecNum, cs::NamedTuple, ::MatNum,
                                  ::AbstractVector{Bool}, ::MatNum, ::MatNum, ::MatNum,
                                  ::Nothing, ::TargetReturnForecast, ::Integer)::NamedTuple
    return cs
end
function target_forecast_matured!(Pm::MatNum, p::VecNum, cs::NamedTuple, Sf::MatNum,
                                  ok::AbstractVector{Bool}, B::MatNum, fwd::MatNum,
                                  w::MatNum, vs::MatNum, rfe::TargetReturnForecast,
                                  s::Integer)::NamedTuple
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
    return_forecast_step(rfe::TargetReturnForecast, P::NamedTuple,
                         csfm::CrossSectionalFactorModel, fs::Option{<:NamedTuple},
                         cre::Option{<:AbstractCrossSectionalRegressionEstimator}) -> NamedTuple

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
  - `P`: The scores, the weights and the group labels, `(; S, w, g)`, as [`descriptor_panel_scores`](@ref) states them. Its first rows are the observations that `fs` predicted and whose target has not matured, and the rest are new. The function can change `P.S` in place.
  - `csfm`: The factor-model block of the last rows of `P`. It must carry the cross-sectional fit, and the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref). Under `cre` it must carry the exposure history and the regression weight history.
  - `fs`: The fold state after the observations before the new ones, as this function returns it, or `nothing` for an empty state.
  - `cre`: Cross-Sectional Regression Estimator that splits each matured prediction for `κ⊥`, or `nothing` to fit no `κ⊥`.

# Validation

  - The rules of [`target_forecast_design`](@ref) and of [`return_forecast_rows`](@ref).
  - When `cre` is given, the rules of [`target_forecast_orthogonal_rows`](@ref).

# Returns

  - `hist::MatNum`: The forecast that the member publishes at each new observation, `observations × assets`, on the rows of `P`. A row that `fs` predicted reads `NaN`.
  - `fs::NamedTuple`: The fold state after the new observations.

# Related

  - [`return_forecast`](@ref)
  - [`return_forecast_result`](@ref)
  - [`folds_forecast_rows`](@ref)
  - [`target_forecast_prequential`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function return_forecast_step(rfe::TargetReturnForecast, P::NamedTuple,
                              csfm::CrossSectionalFactorModel, fs::Option{<:NamedTuple},
                              cre::Option{<:AbstractCrossSectionalRegressionEstimator})::NamedTuple
    d = target_forecast_design(rfe, P, csfm, return_forecast_rows(size(P.S, 1), csfm))
    Sa = d.Sa
    T, N, K = size(Sa)
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
    rows = Vector{Any}(undef, T - p0)
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
                                   target_forecast_latest(model, view(Sa, t:t, :, :)),
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
    return (; hist = hist,
            fs = (; XtX = XtX, Xty = Xty, n = n, model = model,
                  B = B[max(T - g + 1, 1):T, :], cs = cs, ocs = ocs))
end
function return_forecast_step(rfe::TargetReturnForecast, P::NamedTuple,
                              csfm::CrossSectionalFactorModel,
                              fs::Option{<:NamedTuple})::NamedTuple
    return return_forecast_step(rfe, P, csfm, fs, nothing)
end
"""
    return_forecast_result(rfe::TargetReturnForecast, hist::MatNum,
                           fs::NamedTuple) -> TargetReturnForecastResult

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
                                fs::NamedTuple)::TargetReturnForecastResult
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
