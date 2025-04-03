function get_chol_or_sigma_pm(model::JuMP.Model, pm::AbstractPriorModel)
    if !haskey(model, :G)
        G = cholesky(pm.sigma).U
        @expression(model, G, G)
    end
    return model[:G]
end
function get_chol_or_sigma_pm(model::JuMP.Model,
                              pm::Union{<:FactorPriorModel,
                                        <:FactorBlackLittermanPriorModel})
    if !haskey(model, :G)
        G = pm.chol
        @expression(model, G, G)
    end
    return model[:G]
end
function set_risk_upper_bound!(args...)
    return nothing
end
function set_risk_upper_bound!(::MeanRisk, model::JuMP.Model, r_expr, ub::Real, key)
    k = model[:k]
    sc = model[:sc]
    model[Symbol("$(key)_ub")] = @constraint(model, sc * r_expr <= sc * ub * k)
    return nothing
end
function set_risk_expression!(model::JuMP.Model, r_expr, scale::Real, rke::Bool)
    if !rke
        return nothing
    end
    if !haskey(model, :risk_vec)
        @expression(model, risk_vec, Union{AffExpr, QuadExpr}[])
    end
    risk_vec = model[:risk_vec]
    push!(risk_vec, scale * r_expr)
    return nothing
end
function set_risk_bounds_and_expression!(opt::MeanRisk, model::JuMP.Model, r_expr,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(opt, model, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, r::StandardDeviation,
                               pm::AbstractPriorModel, opt::MeanRisk, i::Integer)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pm) : cholesky(r.sigma).L
    key = Symbol("sd_risk_$(i)")
    sd_risk = model[key] = @variable(model)
    model[Symbol("sd_soc_$(i)")] = @constraint(model,
                                               [sc * sd_risk; sc * G * w] ∈
                                               SecondOrderCone())
    set_risk_bounds_and_expression!(opt, model, sd_risk, r.settings, key)
    return nothing
end
