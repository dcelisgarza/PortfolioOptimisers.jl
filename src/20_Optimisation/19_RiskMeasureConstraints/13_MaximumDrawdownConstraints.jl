"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add maximum drawdown risk constraints to `model`.

Introduces a scalar variable `mdd_risk` and the constraint
`sc * (mdd_risk .- dd[2:T+1]) >= 0` so that `mdd_risk` upper-bounds every drawdown
observation. Returns the existing expression if already present.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MDD}(\\boldsymbol{w}) &= \\max_{t=1,\\ldots,T} dd_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MDD}(\\boldsymbol{w})``: Maximum drawdown.
  - $(math_dict[:T])
  - ``dd_t``: Portfolio drawdown at time ``t``.

where ``dd_t`` is the portfolio drawdown at time ``t``.

# Arguments

  - $(arg_dict[:model])
  - `r::MaximumDrawdown`: Maximum drawdown risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::MaximumDrawdown,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :mdd_risk)
        return model[:mdd_risk]
    end
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    JuMP.@variable(model, mdd_risk)
    JuMP.@constraint(model, cmdd_risk, sc * (mdd_risk .- view(dd, 2:(T + 1))) >= 0)
    set_risk_bounds_and_expression!(model, opt, mdd_risk, r.settings, :mdd_risk)
    return mdd_risk
end
