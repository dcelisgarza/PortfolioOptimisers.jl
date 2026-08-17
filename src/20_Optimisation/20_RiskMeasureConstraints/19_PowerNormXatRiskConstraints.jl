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
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    T = length(net_X)
    ip = inv(r.p)
    pvar_eta, pvar_t, pvar_w, pvar_v = JuMP.@variables(model, begin
                                                           ()
                                                           ()
                                                           [1:T], (lower_bound = 0)
                                                           [1:T]
                                                       end)
    state_set!(model, prefix, :pvar_eta_, i, pvar_eta)
    state_set!(model, prefix, :pvar_t_, i, pvar_t)
    state_set!(model, prefix, :pvar_w_, i, pvar_w)
    state_set!(model, prefix, :pvar_v_, i, pvar_v)

    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    iaT = if isnothing(wi)
        state_set!(model, prefix, :cpvar_eq_, i,
                   JuMP.@constraint(model, sc * (sum(pvar_v) - pvar_t) <= 0))
        inv(r.alpha * T^ip)
    else
        state_set!(model, prefix, :cpvar_eq_, i,
                   JuMP.@constraint(model,
                                    sc * (LinearAlgebra.dot(wi, pvar_v) - pvar_t) <= 0))
        inv(r.alpha * sum(wi)^ip)
    end
    cpvar, cpvar_pcone = JuMP.@constraints(model,
                                           begin
                                               sc * ((net_X + pvar_w) .+ pvar_eta) >= 0
                                               [i = 1:T],
                                               [sc * pvar_v[i], sc * pvar_t,
                                                sc * pvar_w[i]] in JuMP.MOI.PowerCone(ip)
                                           end)
    state_set!(model, prefix, :cpvar_, i, cpvar)
    state_set!(model, prefix, :cpvar_pcone_, i, cpvar_pcone)
    pvar_risk = state_set!(model, prefix, :pvar_risk_, i,
                           JuMP.@expression(model, pvar_eta + iaT * pvar_t))
    set_risk_bounds_and_expression!(model, opt, pvar_risk, r.settings, :pvar_risk_, i;
                                    prefix = prefix)
    return pvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `PowerNormValueatRiskRange` (PNVaR range) to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail from `alpha`
and `pa` on the net portfolio returns, and the gain tail from `beta` and `pb` on their
negation, then sums the two PNVaR expressions. Each tail brings its own power cone, shaped
by *its own* norm order.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::PowerNormValueatRiskRange`: The power-norm VaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `pvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`PowerNormValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :pvar_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
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
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    ip = inv(r.p)
    pdar_eta, pdar_t, pdar_w, pdar_v = JuMP.@variables(model, begin
                                                           ()
                                                           ()
                                                           [1:T], (lower_bound = 0)
                                                           [1:T]
                                                       end)
    state_set!(model, prefix, :pdar_eta_, i, pdar_eta)
    state_set!(model, prefix, :pdar_t_, i, pdar_t)
    state_set!(model, prefix, :pdar_w_, i, pdar_w)
    state_set!(model, prefix, :pdar_v_, i, pdar_v)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    iaT = if isnothing(wi)
        state_set!(model, prefix, :cpdar_eq_, i,
                   JuMP.@constraint(model, sc * (sum(pdar_v) - pdar_t) <= 0))
        inv(r.alpha * T^ip)
    else
        state_set!(model, prefix, :cpdar_eq_, i,
                   JuMP.@constraint(model,
                                    sc * (LinearAlgebra.dot(wi, pdar_v) - pdar_t) <= 0))
        inv(r.alpha * sum(wi)^ip)
    end
    cpdar, cpdar_pcone = JuMP.@constraints(model,
                                           begin
                                               sc *
                                               ((pdar_w - view(dd, 2:(T + 1))) .+ pdar_eta) >=
                                               0
                                               [i = 1:T],
                                               [sc * pdar_v[i], sc * pdar_t,
                                                sc * pdar_w[i]] in JuMP.MOI.PowerCone(ip)
                                           end)
    state_set!(model, prefix, :cpdar_, i, cpdar)
    state_set!(model, prefix, :cpdar_pcone_, i, cpdar_pcone)
    pdar_risk = state_set!(model, prefix, :pdar_risk_, i,
                           JuMP.@expression(model, pdar_eta + iaT * pdar_t))
    set_risk_bounds_and_expression!(model, opt, pdar_risk, r.settings, :pdar_risk_, i;
                                    prefix = prefix)
    return pdar_risk
end
