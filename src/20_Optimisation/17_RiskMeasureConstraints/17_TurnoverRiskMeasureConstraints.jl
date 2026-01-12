function set_risk_constraints!(model::JuMP.Model, i::Any, r::TurnoverRiskMeasure,
                               opt::RiskJuMPOptimisationEstimator, ::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:turnover_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    N = length(w)
    turnover_risk = model[key] = JuMP.@variable(model)
    benchmark = r.w
    turnover_r = model[Symbol(:turnover_r_, i)] = JuMP.@expression(model, w - benchmark * k)
    model[Symbol(:cturnover_r_noc_, i)] = JuMP.@constraint(model,
                                                           [sc * turnover_risk;
                                                            sc * turnover_r] in
                                                           JuMP.MOI.NormOneCone(1 + N))
    set_risk_bounds_and_expression!(model, opt, turnover_risk, r.settings, key)
    return turnover_risk
end
