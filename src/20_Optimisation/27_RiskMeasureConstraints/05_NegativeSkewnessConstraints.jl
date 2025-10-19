function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:SOCRiskExpr},
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:nskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.V) ? get_chol_or_V_pm(model, pr) : cholesky(r.V).U
    nskew_risk = model[key] = @variable(model)
    model[Symbol(:cnskew_soc_, i)] = @constraint(model,
                                                 [sc * nskew_risk; sc * G * w] in
                                                 SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, nskew_risk, r.settings, key)
    return nskew_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:SquaredSOCRiskExpr},
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:sqsoc_nskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.V) ? get_chol_or_V_pm(model, pr) : cholesky(r.V).U
    t_qnskew_risk = model[Symbol(:t_sqsoc_skew_risk_, i)] = @variable(model)
    model[Symbol(:csqsoc_nskew_soc_, i)] = @constraint(model,
                                                       [sc * t_qnskew_risk; sc * G * w] in
                                                       SecondOrderCone())
    qnskew_risk = model[key] = @expression(model, t_qnskew_risk^2)
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, t_qnskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:QuadRiskExpr},
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:qnskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    V, G = if isnothing(r.V)
        pr.V, get_chol_or_V_pm(model, pr)
    else
        r.V, cholesky(r.V).U
    end
    t_qnskew_risk = model[Symbol(:t_qnskew_risk_, i)] = @variable(model)
    model[Symbol(:cqnskew_soc_, i)] = @constraint(model,
                                                  [sc * t_qnskew_risk; sc * G * w] in
                                                  SecondOrderCone())
    qnskew_risk = model[key] = @expression(model, dot(w, V, w))
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, t_qnskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_risk_constraints!(::JuMP.Model, ::Any, ::NegativeSkewness,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               pr::LowOrderPrior, args...; kwargs...)
    throw(ArgumentError("NegativeSkewness requires a HighOrderPrior, not a $(typeof(pr))."))
end
