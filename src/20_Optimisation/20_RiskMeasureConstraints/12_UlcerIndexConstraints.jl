"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Ulcer Index risk constraints to `model`.

Introduces a scalar variable `uci` and the SOC constraint
`[sc * uci; sc * dd[2:T+1]] in SecondOrderCone()`, then defines
`uci_risk = uci / sqrt(T)`. Returns the existing expression if already present.

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
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    if haskey(model, Symbol(prefix, :uci))
        return model[Symbol(prefix, :uci_risk)]
    end
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    uci = preg!(model, prefix, :uci, JuMP.@variable(model))
    uci_risk = preg!(model, prefix, :uci_risk, JuMP.@expression(model, uci / sqrt(T)))
    preg!(model, prefix, :cuci_soc,
          JuMP.@constraint(model,
                           [sc * uci; sc * view(dd, 2:(T + 1))] in JuMP.SecondOrderCone()))
    set_risk_bounds_and_expression!(model, opt, uci_risk, r.settings,
                                    Symbol(prefix, :uci_risk))
    return uci_risk
end
