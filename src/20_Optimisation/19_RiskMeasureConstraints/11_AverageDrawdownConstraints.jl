"""
    set_risk_constraints!(model, i, r::AverageDrawdown, opt, pr, args...; kwargs...)

Add average drawdown risk constraints to `model`.

Calls [`set_drawdown_constraints!`](@ref) to ensure drawdown variables exist, then creates
an observation-weighted mean of the drawdown path as the risk expression.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `r::AverageDrawdown`: Average drawdown risk measure instance.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result containing `X`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::AverageDrawdown,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:add_risk_, i)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    add_risk = model[Symbol(key)] = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(view(dd, 2:(T + 1))))
    else
        JuMP.@expression(model, Statistics.mean(view(dd, 2:(T + 1)), wi))
    end
    set_risk_bounds_and_expression!(model, opt, add_risk, r.settings, key)
    return add_risk
end
