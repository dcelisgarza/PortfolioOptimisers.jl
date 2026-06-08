"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Power-Norm Value-at-Risk, PNVaR range, or Power-Norm Drawdown-at-Risk constraints to
`model`.

Each overload uses power cone constraints (`PowerCone`) parameterised by `r.p` (or `r.pa`,
`r.pb` for the range variant) to encode the power-norm VaR. Auxiliary non-negative variables
`pvar_w` and `pvar_v` encode per-observation exceedances, and a scalar `pvar_t` aggregates
the total. The range variant introduces separate lower and upper tail variables. The drawdown
variant operates on the drawdown path.

# Mathematical definition

Power-Norm Value-at-Risk:

```math
\\begin{align}
\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{w}) &= \\eta + \\frac{1}{\\alpha T^{1/p}} t\\,, \\\\
\\sum_{t&=1}^T v_t \\leq t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{w})``: Power Norm Value-at-Risk.
  - ``\\eta``, ``t``, ``v_t``: Conic optimisation variables.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``p \\geq 1``: Power parameter.

```math
\\begin{align}
(w_t,\\, \\eta + \\hat{r}_t,\\, v_t) &\\in \\mathcal{K}_{1/p} \\quad \\forall\\, t,\\quad w_t \\geq 0\\,.
\\end{align}
```

Where:

  - ``w_t``: Auxiliary variable for the power cone constraint.
  - ``\\hat{r}_t``: Portfolio return at time ``t``.
  - ``\\mathcal{K}_{1/p}``: Power cone with exponent ``1/p``.

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
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pvar_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    ip = inv(r.p)
    pvar_eta, pvar_t, pvar_w, pvar_v = model[Symbol(:pvar_eta_, i)], model[Symbol(:pvar_t_, i)], model[Symbol(:pvar_w_, i)], model[Symbol(:pvar_v_, i)] = JuMP.@variables(model,
                                                                                                                                                                          begin
                                                                                                                                                                              ()
                                                                                                                                                                              ()
                                                                                                                                                                              [1:T],
                                                                                                                                                                              (lower_bound = 0)
                                                                                                                                                                              [1:T]
                                                                                                                                                                          end)

    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    iaT = if isnothing(wi)
        model[Symbol(:cpvar_eq_, i)] = JuMP.@constraint(model,
                                                        sc * (sum(pvar_v) - pvar_t) <= 0)
        inv(r.alpha * T^ip)
    else
        model[Symbol(:cpvar_eq_, i)] = JuMP.@constraint(model,
                                                        sc *
                                                        (LinearAlgebra.dot(wi, pvar_v) -
                                                         pvar_t) <= 0)
        inv(r.alpha * sum(wi)^ip)
    end
    model[Symbol(:cpvar_, i)], model[Symbol(:cpvar_pcone_, i)] = JuMP.@constraints(model,
                                                                                   begin
                                                                                       sc *
                                                                                       ((net_X +
                                                                                         pvar_w) .+
                                                                                        pvar_eta) >=
                                                                                       0
                                                                                       [i = 1:T],
                                                                                       [sc *
                                                                                        pvar_v[i],
                                                                                        sc *
                                                                                        pvar_t,
                                                                                        sc *
                                                                                        pvar_w[i]] in
                                                                                       JuMP.MOI.PowerCone(ip)
                                                                                   end)
    pvar_risk = model[key] = JuMP.@expression(model, pvar_eta + iaT * pvar_t)
    set_risk_bounds_and_expression!(model, opt, pvar_risk, r.settings, key)
    return pvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `PowerNormValueatRiskRange` to `model`.

Introduces variables and power-cone constraints to encode the range between a lower and
upper power-norm value-at-risk, parameterised by `r.pa` and `r.pb`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::PowerNormValueatRiskRange`: The power-norm VaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`PowerNormValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pvar_range_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    ipa = inv(r.pa)
    ipb = inv(r.pb)
    pvar_eta_l, pvar_t_l, pvar_w_l, pvar_v_l, pvar_eta_h, pvar_t_h, pvar_w_h, pvar_v_h = model[Symbol(:pvar_eta_l_, i)], model[Symbol(:pvar_t_l_, i)], model[Symbol(:pvar_w_l_, i)], model[Symbol(:pvar_v_l_, i)], model[Symbol(:pvar_eta_h_, i)], model[Symbol(:pvar_t_h_, i)], model[Symbol(:pvar_w_h_, i)], model[Symbol(:pvar_v_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                                                                                              begin
                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                  [1:T],
                                                                                                                                                                                                                                                                                                                                                                  (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                  [1:T],
                                                                                                                                                                                                                                                                                                                                                                  (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                              end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    iaT, ibT = if isnothing(wi)
        model[Symbol(:cpvar_eq_l_, i)], model[Symbol(:cpvar_eq_h_, i)] = JuMP.@constraints(model,
                                                                                           begin
                                                                                               sc *
                                                                                               (sum(pvar_v_l) -
                                                                                                pvar_t_l) <=
                                                                                               0
                                                                                               sc *
                                                                                               (sum(pvar_v_h) -
                                                                                                pvar_t_h) >=
                                                                                               0
                                                                                           end)
        inv(r.alpha * T^ipa), inv(r.beta * T^ipb)
    else
        sw = sum(wi)
        model[Symbol(:cpvar_eq_l_, i)], model[Symbol(:cpvar_eq_h_, i)] = JuMP.@constraints(model,
                                                                                           begin
                                                                                               sc *
                                                                                               (LinearAlgebra.dot(wi,
                                                                                                                  pvar_v_l) -
                                                                                                pvar_t_l) <=
                                                                                               0
                                                                                               sc *
                                                                                               (LinearAlgebra.dot(wi,
                                                                                                                  pvar_v_h) -
                                                                                                pvar_t_h) >=
                                                                                               0
                                                                                           end)
        inv(r.alpha * sw^ipa), inv(r.beta * sw^ipb)
    end
    model[Symbol(:cpvar_l_, i)], model[Symbol(:cpvar_pcone_l_, i)], model[Symbol(:cpvar_h_, i)] = JuMP.@constraints(model,
                                                                                                                    begin
                                                                                                                        sc *
                                                                                                                        ((net_X +
                                                                                                                          pvar_w_l) .+
                                                                                                                         pvar_eta_l) >=
                                                                                                                        0
                                                                                                                        [i = 1:T],
                                                                                                                        [sc *
                                                                                                                         pvar_v_l[i],
                                                                                                                         sc *
                                                                                                                         pvar_t_l,
                                                                                                                         sc *
                                                                                                                         pvar_w_l[i]] in
                                                                                                                        JuMP.MOI.PowerCone(ipa)
                                                                                                                        sc *
                                                                                                                        ((net_X +
                                                                                                                          pvar_w_h) .+
                                                                                                                         pvar_eta_h) <=
                                                                                                                        0
                                                                                                                        [i = 1:T],
                                                                                                                        [sc *
                                                                                                                         -pvar_v_h[i],
                                                                                                                         sc *
                                                                                                                         -pvar_t_h,
                                                                                                                         sc *
                                                                                                                         -pvar_w_h[i]] in
                                                                                                                        JuMP.MOI.PowerCone(ipb)
                                                                                                                    end)
    pvar_risk_l, pvar_risk_h = model[Symbol(:pvar_risk_l_, i)], model[Symbol(:pvar_risk_h_, i)] = JuMP.@expressions(model,
                                                                                                                    begin
                                                                                                                        pvar_eta_l +
                                                                                                                        iaT *
                                                                                                                        pvar_t_l
                                                                                                                        pvar_eta_h +
                                                                                                                        ibT *
                                                                                                                        pvar_t_h
                                                                                                                    end)
    pvar_range_risk = model[key] = JuMP.@expression(model, pvar_risk_l - pvar_risk_h)
    set_risk_bounds_and_expression!(model, opt, pvar_range_risk, r.settings, key)
    return pvar_range_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `PowerNormDrawdownatRisk` to `model`.

Introduces variables and power-cone constraints to encode the power-norm drawdown-at-risk,
computed over the drawdown path of portfolio returns.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::PowerNormDrawdownatRisk`: The power-norm drawdown-at-risk risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`PowerNormDrawdownatRisk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pdar_risk_, i)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    ip = inv(r.p)
    pdar_eta, pdar_t, pdar_w, pdar_v = model[Symbol(:pdar_eta_, i)], model[Symbol(:pdar_t_, i)], model[Symbol(:pdar_w_, i)], model[Symbol(:pdar_v_, i)] = JuMP.@variables(model,
                                                                                                                                                                          begin
                                                                                                                                                                              ()
                                                                                                                                                                              ()
                                                                                                                                                                              [1:T],
                                                                                                                                                                              (lower_bound = 0)
                                                                                                                                                                              [1:T]
                                                                                                                                                                          end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    iaT = if isnothing(wi)
        model[Symbol(:cpdar_eq_, i)] = JuMP.@constraint(model,
                                                        sc * (sum(pdar_v) - pdar_t) <= 0)
        inv(r.alpha * T^ip)
    else
        model[Symbol(:cpdar_eq_, i)] = JuMP.@constraint(model,
                                                        sc *
                                                        (LinearAlgebra.dot(wi, pdar_v) -
                                                         pdar_t) <= 0)
        inv(r.alpha * sum(wi)^ip)
    end
    model[Symbol(:cpdar_, i)], model[Symbol(:cpdar_pcone_, i)] = JuMP.@constraints(model,
                                                                                   begin
                                                                                       sc *
                                                                                       ((pdar_w -
                                                                                         view(dd,
                                                                                              2:(T + 1))) .+
                                                                                        pdar_eta) >=
                                                                                       0
                                                                                       [i = 1:T],
                                                                                       [sc *
                                                                                        pdar_v[i],
                                                                                        sc *
                                                                                        pdar_t,
                                                                                        sc *
                                                                                        pdar_w[i]] in
                                                                                       JuMP.MOI.PowerCone(ip)
                                                                                   end)
    pdar_risk = model[key] = JuMP.@expression(model, pdar_eta + iaT * pdar_t)
    set_risk_bounds_and_expression!(model, opt, pdar_risk, r.settings, key)
    return pdar_risk
end
