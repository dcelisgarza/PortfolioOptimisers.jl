"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Entropic Value-at-Risk, EVaR range, or Entropic Drawdown-at-Risk constraints to `model`.

Each overload uses exponential cone constraints (`ExponentialCone`) to encode the cumulant
generating function bound. Scalar variables `t`, `z`, and per-observation variables `u` are
introduced. `EVaR` and `EDaR` encode the single-tail bound; the range variant encodes both a
lower and upper exponential cone.

# Mathematical definition

Entropic Value-at-Risk via exponential cone (Ahmadi-Javid 2012):

```math
\\begin{align}
\\mathrm{EVaR}_\\alpha(\\boldsymbol{w}) &= t - z \\ln(\\alpha T)\\,, \\\\
\\sum_{t&=1}^T u_t \\leq z\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EVaR}_\\alpha(\\boldsymbol{w})``: Entropic Value-at-Risk.
  - ``t``, ``z``: Dual variables for the exponential cone programme.
  - ``u_t``: Auxiliary exponential cone variables.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

```math
\\begin{align}
(-\\hat{r}_t - t,\\; z,\\; u_t) &\\in \\mathcal{K}_{\\exp} \\quad \\forall\\, t\\,.
\\end{align}
```

Where:

  - ``\\hat{r}_t``: Portfolio return at time ``t``.
  - ``\\mathcal{K}_{\\exp} = \\{(a,b,c) : b e^{a/b} \\leq c,\\, b > 0\\}``: Exponential cone.
  - ``u_t``: Auxiliary exponential cone variables.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - $(arg_dict[:r_risk])
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    key = Symbol(:evar_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    T = length(net_X)
    t_evar, z_evar, u_evar = model[Symbol(:t_evar_, i)], model[Symbol(:z_evar_, i)], model[Symbol(:u_evar_, i)] = JuMP.@variables(model,
                                                                                                                                  begin
                                                                                                                                      ()
                                                                                                                                      (),
                                                                                                                                      (lower_bound = 0)
                                                                                                                                      [1:T]
                                                                                                                                  end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    at = if isnothing(wi)
        model[Symbol(:cevar_, i)] = JuMP.@constraint(model,
                                                     sc * (sum(u_evar) - z_evar) <= 0)
        r.alpha * T
    else
        model[Symbol(:cevar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (LinearAlgebra.dot(wi, u_evar) -
                                                      z_evar) <= 0)
        r.alpha * sum(wi)
    end
    model[Symbol(:cevar_exp_cone_, i)] = JuMP.@constraint(model, [i = 1:T],
                                                          [sc * (-net_X[i] - t_evar),
                                                           sc * z_evar, sc * u_evar[i]] in
                                                          JuMP.MOI.ExponentialCone())
    evar_risk = model[Symbol(:evar_risk_, i)] = JuMP.@expression(model,
                                                                 t_evar - z_evar * log(at))
    set_risk_bounds_and_expression!(model, opt, evar_risk, r.settings, key)
    return evar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `EntropicValueatRiskRange` (EVaR range) to `model`.

Introduces two sets of exponential cone variables for the lower-tail and upper-tail EVaR
expressions and computes their difference as the range risk.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::EntropicValueatRiskRange`: The EVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`EntropicValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:evar_risk_range_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    t_evar_l, z_evar_l, u_evar_l, t_evar_h, z_evar_h, u_evar_h = model[Symbol(:t_evar_l_, i)], model[Symbol(:z_evar_l_, i)], model[Symbol(:u_evar_l_, i)], model[Symbol(:t_evar_h_, i)], model[Symbol(:z_evar_h_, i)], model[Symbol(:u_evar_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                      begin
                                                                                                                                                                                                                                                                          ()
                                                                                                                                                                                                                                                                          (),
                                                                                                                                                                                                                                                                          (lower_bound = 0)
                                                                                                                                                                                                                                                                          [1:T]
                                                                                                                                                                                                                                                                          ()
                                                                                                                                                                                                                                                                          (),
                                                                                                                                                                                                                                                                          (lower_bound = 0)
                                                                                                                                                                                                                                                                          [1:T]
                                                                                                                                                                                                                                                                      end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    at, bt = if isnothing(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (sum(u_evar_l) -
                                                                                          z_evar_l) <=
                                                                                         0
                                                                                         sc *
                                                                                         (sum(u_evar_h) -
                                                                                          z_evar_h) <=
                                                                                         0
                                                                                     end)
        r.alpha * T, r.beta * T
    else
        sw = sum(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            u_evar_l) -
                                                                                          z_evar_l) <=
                                                                                         0
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            u_evar_h) -
                                                                                          z_evar_h) <=
                                                                                         0
                                                                                     end)
        r.alpha * sw, r.beta * sw
    end
    model[Symbol(:cevar_exp_cone_l_, i)], model[Symbol(:cevar_exp_cone_h_, i)] = JuMP.@constraints(model,
                                                                                                   begin
                                                                                                       [i = 1:T],
                                                                                                       [sc *
                                                                                                        (-net_X[i] -
                                                                                                         t_evar_l),
                                                                                                        sc *
                                                                                                        z_evar_l,
                                                                                                        sc *
                                                                                                        u_evar_l[i]] in
                                                                                                       JuMP.MOI.ExponentialCone()
                                                                                                       [i = 1:T],
                                                                                                       [sc *
                                                                                                        (net_X[i] -
                                                                                                         t_evar_h),
                                                                                                        sc *
                                                                                                        z_evar_h,
                                                                                                        sc *
                                                                                                        u_evar_h[i]] in
                                                                                                       JuMP.MOI.ExponentialCone()
                                                                                                   end)
    evar_risk_l, evar_risk_h = model[Symbol(:evar_risk_l_, i)], model[Symbol(:evar_risk_h_, i)] = JuMP.@expressions(model,
                                                                                                                    begin
                                                                                                                        t_evar_l -
                                                                                                                        z_evar_l *
                                                                                                                        log(at)
                                                                                                                        t_evar_h -
                                                                                                                        z_evar_h *
                                                                                                                        log(bt)
                                                                                                                    end)
    evar_risk_range = model[key] = JuMP.@expression(model, evar_risk_l + evar_risk_h)
    set_risk_bounds_and_expression!(model, opt, evar_risk_range, r.settings, key)
    return evar_risk_range
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `EntropicDrawdownatRisk` (EDaR) to `model`.

Uses exponential cone constraints applied to the drawdown series to encode the entropic
drawdown-at-risk at confidence level `r.alpha`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::EntropicDrawdownatRisk`: The EDaR risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`EntropicDrawdownatRisk`](@ref)
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:edar_risk_, i)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    at = r.alpha * T
    t_edar, z_edar, u_edar = model[Symbol(:t_edar_, i)], model[Symbol(:z_edar_, i)], model[Symbol(:u_edar_, i)] = JuMP.@variables(model,
                                                                                                                                  begin
                                                                                                                                      ()
                                                                                                                                      (),
                                                                                                                                      (lower_bound = 0)
                                                                                                                                      [1:T]
                                                                                                                                  end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    at = if isnothing(wi)
        model[Symbol(:cedar_, i)] = JuMP.@constraint(model,
                                                     sc * (sum(u_edar) - z_edar) <= 0)
        r.alpha * T
    else
        model[Symbol(:cedar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (LinearAlgebra.dot(wi, u_edar) -
                                                      z_edar) <= 0)
        r.alpha * sum(wi)
    end
    model[Symbol(:cedar_exp_cone_, i)] = JuMP.@constraint(model, [i = 1:T],
                                                          [sc * (dd[i + 1] - t_edar),
                                                           sc * z_edar, sc * u_edar[i]] in
                                                          JuMP.MOI.ExponentialCone())
    edar_risk = model[key] = JuMP.@expression(model, t_edar - z_edar * log(at))
    set_risk_bounds_and_expression!(model, opt, edar_risk, r.settings, key)
    return edar_risk
end
