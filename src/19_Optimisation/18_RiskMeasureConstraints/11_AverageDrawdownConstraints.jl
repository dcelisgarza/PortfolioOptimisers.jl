function set_risk_constraints!(model::JuMP.Model, i::Any, r::AverageDrawdown,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:add_risk_, i)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    wi = nothing_scalar_array_selector(r.w, pr.w)
    add_risk = model[Symbol(key)] = if isnothing(wi)
        @expression(model, mean(view(dd, 2:(T + 1))))
    else
        @expression(model, mean(view(dd, 2:(T + 1)), wi))
    end
    set_risk_bounds_and_expression!(model, opt, add_risk, r.settings, key)
    return add_risk
end
