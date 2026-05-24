"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Ulcer Index risk constraints to `model`.

Introduces a scalar variable `uci` and the SOC constraint
`[sc * uci; sc * dd[2:T+1]] in SecondOrderCone()`, then defines
`uci_risk = uci / sqrt(T)`. Returns the existing expression if already present.

# Mathematical definition

```math
\\mathrm{UCI}(\\boldsymbol{w}) = \\frac{\\|\\boldsymbol{dd}\\|_2}{\\sqrt{T}} = \\sqrt{\\frac{1}{T}\\sum_{t=1}^T dd_t^2}
```

where ``dd_t`` is the portfolio drawdown at time ``t``.

# Arguments

  - $(arg_dict[:model])
  - `r::UlcerIndex`: Ulcer index risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::UlcerIndex,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :uci)
        return model[:uci_risk]
    end
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    JuMP.@variable(model, uci)
    JuMP.@expression(model, uci_risk, uci / sqrt(T))
    JuMP.@constraint(model, cuci_soc,
                     [sc * uci; sc * view(dd, 2:(T + 1))] in JuMP.SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, uci_risk, r.settings, :uci_risk)
    return uci_risk
end
