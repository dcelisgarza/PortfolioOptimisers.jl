"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Ulcer Index risk constraints to `model`.

Introduces a scalar variable `uci` and one second-order cone over the drawdowns, then defines the risk expression `uci_risk`. Without observation weights, the cone is `[sc * uci; sc * dd[2:T+1]] in SecondOrderCone()` and `uci_risk = uci / sqrt(T)`. With observation weights `w`, each drawdown is scaled by `sqrt(w_t)` inside the cone and `uci_risk = uci / sqrt(sum(w))`. The keys are indexed by `i`, because two Ulcer Index measures with different weights are two expressions.

# Mathematical definition

```math
\\begin{align}
\\mathrm{UCI}(\\boldsymbol{w}) &= \\frac{\\lVert \\boldsymbol{dd} \\rVert_2}{\\sqrt{T}} = \\sqrt{\\frac{1}{T}\\sum_{t=1}^T dd_t^2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{UCI}(\\boldsymbol{w})``: Ulcer index.
  - ``\\boldsymbol{dd}``: Drawdown vector.
  - $(math_dict[:T])
  - ``dd_t``: Portfolio drawdown at time ``t``.

With observation weights, the weighted mean of the squared drawdowns is used instead:

```math
\\begin{align}
\\mathrm{UCI}(\\boldsymbol{w}) &= \\sqrt{\\frac{1}{W_{T}}\\sum_{t=1}^T w_{t} dd_t^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::UlcerIndex`: Ulcer index risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - The observation weights pass [`checked_observation_weights`](@ref).

# Returns

  - `uci_risk`: The Ulcer Index risk expression.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`checked_observation_weights`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::UlcerIndex,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = checked_observation_weights(wi, pr.X)
    uci = state_set!(model, prefix, :uci_, i, JuMP.@variable(model))
    ddt = view(dd, 2:(T + 1))
    uci_risk, cone = if isnothing(wi)
        JuMP.@expression(model, uci / sqrt(T)), [sc * uci; sc * ddt]
    else
        JuMP.@expression(model, uci / sqrt(sum(wi))), [sc * uci; sc * (sqrt.(wi) .* ddt)]
    end
    state_set!(model, prefix, :cuci_soc_, i,
               JuMP.@constraint(model, cone in JuMP.SecondOrderCone()))
    state_set!(model, prefix, :uci_risk_, i, uci_risk)
    set_risk_bounds_and_expression!(model, opt, uci_risk, r.settings, :uci_risk_, i;
                                    prefix = prefix)
    return uci_risk
end
