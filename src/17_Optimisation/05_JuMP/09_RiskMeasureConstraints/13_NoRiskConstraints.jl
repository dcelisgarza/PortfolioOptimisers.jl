"""
    set_risk_constraints!(model::JuMP.Model, ::Any, r::NoRisk,
                          opt::RiskConstraintOwner, pr::AbstractPriorResult,
                          args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                          kwargs...)

Add the [`NoRisk`](@ref) risk expression to the JuMP model.

Registers a zero affine expression, so the risk contributes nothing to the objective and no variables or constraints are created. The model class is left untouched — a linear problem stays a linear program.

The expression is built once for each `prefix`. The settings of the first [`NoRisk`](@ref) take effect, and a second measure logs a warning, because its bound and its scale have no effect.

# Arguments

  - $(arg_dict[:model])
  - `r::NoRisk`: No-risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nr_risk`: The zero risk expression.

# Related

  - [`NoRisk`](@ref)
  - [`set_unindexed_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::NoRisk,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               loss::Bool = true, prefix::Symbol = Symbol(""), kwargs...)
    key = ifelse(loss, :nr_risk, :nr_risk_gain)
    nr_risk = state_build!(model, prefix, key) do
        return JuMP.@expression(model, zero(JuMP.AffExpr))
    end
    set_unindexed_risk_bounds_and_expression!(model, opt, nr_risk, r, key; prefix = prefix)
    return nr_risk
end
