"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add maximum drawdown risk constraints to `model`.

Introduces a scalar variable `mdd_risk` and the constraint
`sc * (mdd_risk .- dd[2:T+1]) >= 0` so that `mdd_risk` upper-bounds every drawdown
observation. A second call with the same `prefix` uses the registered variable and builds no
rows. The settings of the first [`MaximumDrawdown`](@ref) take effect, and a second measure
logs a warning, because its bound and its scale have no effect.

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

  - `mdd_risk`: The risk variable.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_unindexed_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::MaximumDrawdown,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    mdd_risk = state_build!(model, prefix, :mdd_risk) do
        sc = get_constraint_scale(model)
        dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
        T = length(dd) - 1
        mdd_risk = JuMP.@variable(model)
        state_set!(model, prefix, :cmdd_risk,
                   JuMP.@constraint(model, sc * (mdd_risk .- view(dd, 2:(T + 1))) >= 0))
        return mdd_risk
    end
    set_unindexed_risk_bounds_and_expression!(model, opt, mdd_risk, r, :mdd_risk;
                                              prefix = prefix)
    return mdd_risk
end
