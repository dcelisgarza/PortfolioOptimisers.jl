function get_chol_or_V_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !haskey(model, :GV)
        G = cholesky(pr.V).U
        @expression(model, GV, G)
    end
    return model[:GV]
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:SOCRiskExpr},
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::AbstractJuMPScalar, key::Symbol, args...)
    set_risk_bounds_and_expression!(model, opt, nskew_risk, r.settings, key)
    return nskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:SquaredSOCRiskExpr},
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::AbstractJuMPScalar, key::Symbol, args...)
    qnskew_risk = model[Symbol(:sq_, key)] = @expression(model, nskew_risk^2)
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:QuadRiskExpr},
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::AbstractJuMPScalar, key::Symbol,
                                     V::AbstractMatrix)
    w = model[:w]
    qnskew_risk = model[Symbol(:qd_, key)] = @expression(model, dot(w, V, w))
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness,
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:nskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    V, G = isnothing(r.V) ? (pr.V, get_chol_or_V_pm(model, pr)) : (r.V, cholesky(r.V).U)
    nskew_risk = model[key] = @variable(model)
    model[Symbol(:cnskew_soc_, i)] = @constraint(model,
                                                 [sc * nskew_risk; sc * G * w] in
                                                 SecondOrderCone())
    return set_negative_skewness_risk!(model, r, opt, nskew_risk, key, V)
end
function set_risk_constraints!(::JuMP.Model, ::Any, ::NegativeSkewness,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               pr::LowOrderPrior, args...; kwargs...)
    throw(ArgumentError("NegativeSkewness requires a HighOrderPrior, not a $(typeof(pr))."))
end
