"""
    set_risk_constraints!(model::JuMP.Model, ::Any, r::NoRisk,
                          opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                          args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                          kwargs...)

Add the [`NoRisk`](@ref) risk expression to the JuMP model.

Registers a zero affine expression, so the risk contributes nothing to the objective and no variables or constraints are created. The model class is left untouched — a linear problem stays a linear program.

# Arguments

  - $(arg_dict[:model])
  - `r::NoRisk`: No-risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nr_risk`: The zero risk expression.

# Related

  - [`NoRisk`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::NoRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    key = ifelse(loss, :nr_risk, :nr_risk_gain)
    return state_build!(model, prefix, key) do
        nr_risk = JuMP.@expression(model, zero(JuMP.AffExpr))
        set_risk_bounds_and_expression!(model, opt, nr_risk, r.settings, key;
                                        prefix = prefix)
        return nr_risk
    end
end
