function get_chol_or_sigma_pm(model::JuMP.Model, pm::AbstractPriorResult)
    if !haskey(model, :G)
        G = cholesky(pm.sigma).U
        # G = sqrt(pm.sigma)
        @expression(model, G, G)
    end
    return model[:G]
end
function get_chol_or_sigma_pm(model::JuMP.Model,
                              pm::Union{<:FactorPriorResult,
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
function set_risk_upper_bound!(::MeanRiskEstimator, model::JuMP.Model, r_expr, ub::Real,
                               key)
    k = model[:k]
    sc = model[:sc]
    model[Symbol(ShortString("$(key)_ub"))] = @constraint(model, sc * r_expr <= sc * ub * k)
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
function set_risk_bounds_and_expression!(opt::MeanRiskEstimator, model::JuMP.Model, r_expr,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(opt, model, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk_bounds_and_expression!(opt::MeanRiskEstimator, model::JuMP.Model,
                                         r_expr_ub, ub::Union{Nothing, Real}, key::Symbol,
                                         r_expr, settings::RiskMeasureSettings)
    set_risk_upper_bound!(opt, model, r_expr_ub, ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function _set_risk_constraints!(model::JuMP.Model, r::StandardDeviation,
                                opt::MeanRiskEstimator, pm::AbstractPriorResult, i::Integer,
                                args...)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pm) : cholesky(r.sigma).U
    key = Symbol(ShortString("sd_risk_$(i)"))
    sd_risk = model[key] = @variable(model)
    model[Symbol(ShortString("$(key)_soc"))] = @constraint(model,
                                                           [sc * sd_risk; sc * G * w] ∈
                                                           SecondOrderCone())
    set_risk_bounds_and_expression!(opt, model, sd_risk, r.settings, key)
    return nothing
end
function sdp_rc_variance_flag!(::JuMP.Model, ::MeanRiskEstimator,
                               ::Union{Nothing,
                                       LinearConstraintResult{<:PartialLinearConstraintResult{Nothing,
                                                                                              Nothing},
                                                              <:PartialLinearConstraintResult{Nothing,
                                                                                              Nothing}}})
    return false
end
function sdp_rc_variance_flag!(model::JuMP.Model, ::MeanRiskEstimator,
                               ::LinearConstraintResult)
    set_sdp_constraints!(model)
    return true
end
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool,
                            cplg::Union{Nothing, <:SemiDefinitePhilogenyModel,
                                        <:IntegerPhilogenyModel},
                            nplg::Union{Nothing, <:SemiDefinitePhilogenyModel,
                                        <:IntegerPhilogenyModel})
    return if rc_flag ||
              haskey(model, :rc_variance) ||
              isa(cplg, SemiDefinitePhilogenyModel) ||
              isa(nplg, SemiDefinitePhilogenyModel)
        true
    else
        false
    end
end
function set_variance_risk!(model::JuMP.Model, flag::Bool, pm::AbstractPriorResult,
                            i::Integer, r::Variance, key::Symbol)
    if flag
        set_sdp_variance_risk!(model, pm, i, r, key)
    else
        set_variance_risk!(model, pm, i, r, key)
    end
    return nothing
end
function set_sdp_variance_risk!(model::JuMP.Model, pm::AbstractPriorResult, i::Integer,
                                r::Variance, key::Symbol)
    W = model[:W]
    sigma = isnothing(r.sigma) ? pm.sigma : r.sigma
    sigma_W = model[Symbol(ShortString("sigma_W_$(i)"))] = @expression(model, sigma * W)
    model[key] = @expression(model, tr(sigma_W))
    return nothing
end
function set_variance_risk!(model::JuMP.Model, pm::AbstractPriorResult, i::Integer,
                            r::Variance{<:Any, <:SOC, <:Any, <:Any}, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pm) : cholesky(r.sigma).U
    key_dev = Symbol(ShortString("dev_$(i)"))
    dev = model[key_dev] = @variable(model)
    model[key] = @expression(model, dev^2)
    model[Symbol(ShortString("$(key_dev)_soc"))] = @constraint(model,
                                                               [sc * dev; sc * G * w] ∈
                                                               SecondOrderCone())
    return nothing
end
function set_variance_risk!(model::JuMP.Model, pm::AbstractPriorResult, i::Integer,
                            r::Variance{<:Any, <:Quad, <:Any, <:Any}, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    sigma = isnothing(r.sigma) ? pm.sigma : r.sigma
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pm) : cholesky(r.sigma).U
    model[key] = @expression(model, dot(w, sigma, w))
    key_dev = Symbol(ShortString("dev_$(i)"))
    dev = model[key_dev] = @variable(model)
    model[Symbol(ShortString("$(key_dev)_soc"))] = @constraint(model,
                                                               [sc * dev; sc * G * w] ∈
                                                               SecondOrderCone())
    return nothing
end
function set_variance_risk!(model::JuMP.Model, pm::AbstractPriorResult, i::Integer,
                            r::Variance{<:Any, <:RSOC, <:Any, <:Any}, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    set_net_portfolio_returns!(model, pm.X)
    net_X = model[:net_X]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pm) : cholesky(r.sigma).U
    key_dev = Symbol(ShortString("dev_$(i)"))
    t_variance = model[Symbol(ShortString("t_variance_$(i)"))] = @variable(model)
    dev = model[key_dev] = @variable(model)
    mu = pm.mu
    T = length(net_X)
    variance = model[Symbol(ShortString("variance_$(i)"))] = @expression(model,
                                                                         net_X .-
                                                                         dot(mu, w))
    model[key] = @expression(model, t_variance / (T - one(T)))
    model[Symbol(ShortString("$(key_dev)_soc"))] = @constraint(model,
                                                               [sc * dev; sc * G * w] ∈
                                                               SecondOrderCone())
    model[Symbol(ShortString("$(key)_rsoc"))] = @constraint(model,
                                                            [sc * t_variance; 0.5;
                                                             sc * variance] in
                                                            RotatedSecondOrderCone())
    return nothing
end
function variance_risk_bounds_expr(flag::Bool, model::JuMP.Model, i::Integer)
    return if flag
        key = Symbol(ShortString("variance_risk_$(i)"))
        model[key], key
    else
        key = Symbol(ShortString("dev_$(i)"))
        model[key], key
    end
end
function variance_risk_bounds_val(flag::Bool, ub::Real)
    return flag ? ub : sqrt(ub)
end
function variance_risk_bounds_val(::Any, ::Nothing)
    return nothing
end
function rc_variance_constraints!(args...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Integer, rc::LinearConstraintResult,
                                  variance_risk::AbstractJuMPScalar)
    sigma_W = model[Symbol(ShortString("sigma_W_$(i)"))]
    sc = model[:sc]
    if !haskey(model, :rc_variance)
        @expression(model, rc_variance, true)
    end
    rc_key = Symbol(ShortString("rc_variance_$(i)"))
    vsw = vec(diag(sigma_W))
    if !isnothing(rc.A_ineq)
        model[Symbol(ShortString("$(rc_key)_ineq"))] = @constraint(model,
                                                                   sc * rc.A_ineq * vsw <=
                                                                   sc *
                                                                   rc.B_ineq *
                                                                   variance_risk)
    end
    if !isnothing(rc.A_eq)
        model[Symbol(ShortString("$(rc_key)_eq"))] = @constraint(model,
                                                                 sc * rc.A_eq * vsw ==
                                                                 sc *
                                                                 rc.B_eq *
                                                                 variance_risk)
    end
    return nothing
end
function _set_risk_constraints!(model::JuMP.Model, r::Variance, opt::MeanRiskEstimator,
                                pm::AbstractPriorResult, i::Integer,
                                cplg::Union{Nothing, <:SemiDefinitePhilogenyModel,
                                            <:IntegerPhilogenyModel},
                                nplg::Union{Nothing, <:SemiDefinitePhilogenyModel,
                                            <:IntegerPhilogenyModel})
    if !haskey(model, :variance_flag) && r.settings.rke
        @expression(model, variance_flag, true)
    end
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pm.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, cplg, nplg)
    key = Symbol(ShortString("variance_risk_$(i)"))
    set_variance_risk!(model, sdp_flag, pm, i, r, key)
    variance_risk = model[key]
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(sdp_flag, model, i)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_risk_bounds_and_expression!(opt, model, var_bound_expr, ub, var_bound_key,
                                    variance_risk, r.settings)
    return nothing
end
