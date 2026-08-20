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

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    return set_power_norm_risk_constraints!(model, i, r, opt, pr, series, T,
                                            (; eta = :pvar_eta_, t = :pvar_t_,
                                             slack = :pvar_w_, v = :pvar_v_,
                                             budget = :cpvar_eq_, exceedance = :cpvar_,
                                             pcone = :cpvar_pcone_, risk = :pvar_risk_);
                                            prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the power-norm tail programme of `series` and register it under the names in `keys`.

This is the shared body of `PowerNormValueatRisk` and `PowerNormDrawdownatRisk`. The two are
one power cone programme over different series, so [`risk_series`](@ref) chooses the series
and this function writes the cone once.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The power-norm risk measure, read for `alpha`, `p`, `w` and
    `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `T::Int`: The number of observations.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The power-norm risk expression added to the model.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_power_norm_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                          opt::RiskJuMPOptimisationEstimator,
                                          pr::AbstractPriorResult, series, T::Int,
                                          keys::NamedTuple; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    ip = inv(r.p)
    eta, t, slack, v = JuMP.@variables(model, begin
                                           ()
                                           ()
                                           [1:T], (lower_bound = 0)
                                           [1:T]
                                       end)
    state_set!(model, prefix, keys.eta, i, eta)
    state_set!(model, prefix, keys.t, i, t)
    state_set!(model, prefix, keys.slack, i, slack)
    state_set!(model, prefix, keys.v, i, v)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    iaT = if isnothing(wi)
        state_set!(model, prefix, keys.budget, i,
                   JuMP.@constraint(model, sc * (sum(v) - t) <= 0))
        inv(r.alpha * T^ip)
    else
        state_set!(model, prefix, keys.budget, i,
                   JuMP.@constraint(model, sc * (LinearAlgebra.dot(wi, v) - t) <= 0))
        inv(r.alpha * sum(wi)^ip)
    end
    exceedance, pcone = JuMP.@constraints(model,
                                          begin
                                              sc * ((series + slack) .+ eta) >= 0
                                              [i = 1:T],
                                              [sc * v[i], sc * t, sc * slack[i]] in
                                              JuMP.MOI.PowerCone(ip)
                                          end)
    state_set!(model, prefix, keys.exceedance, i, exceedance)
    state_set!(model, prefix, keys.pcone, i, pcone)
    risk = state_set!(model, prefix, keys.risk, i, JuMP.@expression(model, eta + iaT * t))
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
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
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_power_norm_risk_constraints!(model, i, r, opt, pr, series, T,
                                            (; eta = :pdar_eta_, t = :pdar_t_,
                                             slack = :pdar_w_, v = :pdar_v_,
                                             budget = :cpdar_eq_, exceedance = :cpdar_,
                                             pcone = :cpdar_pcone_, risk = :pdar_risk_);
                                            prefix = prefix)
end
