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

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    return set_entropic_risk_constraints!(model, i, r, opt, pr, series, T,
                                          (; t = :t_evar_, z = :z_evar_, u = :u_evar_,
                                           budget = :cevar_, cone = :cevar_exp_cone_,
                                           risk = :evar_risk_); prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the entropic tail programme of `series` and register it under the names in `keys`.

This is the shared body of `EntropicValueatRisk` and `EntropicDrawdownatRisk`. The two are
one exponential cone programme over different series, so [`risk_series`](@ref) chooses the
series and this function writes the cone once.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The entropic risk measure, read for `alpha`, `w` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `T::Int`: The number of observations.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The entropic risk expression added to the model.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_entropic_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                        opt::RiskJuMPOptimisationEstimator,
                                        pr::AbstractPriorResult, series, T::Int,
                                        keys::NamedTuple; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    t, z, u = JuMP.@variables(model, begin
                                  ()
                                  (), (lower_bound = 0)
                                  [1:T]
                              end)
    state_set!(model, prefix, keys.t, i, t)
    state_set!(model, prefix, keys.z, i, z)
    state_set!(model, prefix, keys.u, i, u)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    at = if isnothing(wi)
        state_set!(model, prefix, keys.budget, i,
                   JuMP.@constraint(model, sc * (sum(u) - z) <= 0))
        r.alpha * T
    else
        state_set!(model, prefix, keys.budget, i,
                   JuMP.@constraint(model, sc * (LinearAlgebra.dot(wi, u) - z) <= 0))
        r.alpha * sum(wi)
    end
    state_set!(model, prefix, keys.cone, i,
               JuMP.@constraint(model, [i = 1:T],
                                [sc * (-series[i] - t), sc * z, sc * u[i]] in
                                JuMP.MOI.ExponentialCone()))
    risk = state_set!(model, prefix, keys.risk, i, JuMP.@expression(model, t - z * log(at)))
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `EntropicValueatRiskRange` (EVaR range) to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail at `alpha` on
the net portfolio returns and the gain tail at `beta` on their negation, then sums the two
EVaR expressions. Each tail brings its own exponential cone block.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::EntropicValueatRiskRange`: The EVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `evar_risk_range`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`EntropicValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :evar_risk_range_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
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
  - [`risk_series`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_entropic_risk_constraints!(model, i, r, opt, pr, series, T,
                                          (; t = :t_edar_, z = :z_edar_, u = :u_edar_,
                                           budget = :cedar_, cone = :cedar_exp_cone_,
                                           risk = :edar_risk_); prefix = prefix)
end
