function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !haskey(model, :G)
        G = isnothing(pr.chol) ? cholesky(pr.sigma).U : pr.chol
        @expression(model, G, G)
    end
    return model[:G]
end
function set_variance_risk_bounds_and_expression!(model::JuMP.Model,
                                                  opt::RiskJuMPOptimisationEstimator,
                                                  r_expr_ub::AbstractJuMPScalar,
                                                  ub::Option{<:RkRtBounds}, key::Symbol,
                                                  r_expr::AbstractJuMPScalar,
                                                  settings::RiskMeasureSettings)
    set_risk_upper_bound!(model, opt, r_expr_ub, ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation,
                   opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...;
                   kwargs...)
    key = Symbol(:sd_risk_, i)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    sd_risk = model[key] = @variable(model)
    model[Symbol(:csd_risk_soc_, i)] = @constraint(model,
                                                   [sc * sd_risk; sc * G * w] in
                                                   SecondOrderCone())
    return sd_risk, key
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    sd_risk, key = set_risk!(model, i, r, opt, pr, args...; kwargs...)
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, key)
    return sd_risk
end
function sdp_rc_variance_flag!(::JuMP.Model, ::RkJuMPOpt, ::Nothing)
    return false
end
function sdp_rc_variance_flag!(::JuMP.Model, ::RkJuMPOpt, ::LinearConstraint)
    return true
end
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool, plg::Option{<:PhC_VecPhC})
    return if rc_flag ||
              haskey(model, :rc_variance) ||
              isa(plg, SemiDefinitePhylogeny) ||
              isa(plg, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plg)
        true
    else
        false
    end
end
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            flag::Bool, key::Symbol)
    return if flag
        set_sdp_variance_risk!(model, i, r, pr, key)
    else
        set_variance_risk!(model, i, r, pr, key)
    end
end
function set_sdp_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                                pr::AbstractPriorResult, key::Symbol)
    W = set_sdp_constraints!(model)
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model, sigma * W)
    return model[key] = @expression(model, tr(sigma_W))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:SquaredSOCRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key_dev = Symbol(:dev_, i)
    dev = model[key_dev] = @variable(model)
    model[Symbol(key_dev, :_soc)] = @constraint(model,
                                                [sc * dev; sc * G * w] in SecondOrderCone())
    return model[key] = @expression(model, dev^2)
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:QuadRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    dev = model[Symbol(:dev_, i)] = @variable(model)
    model[Symbol(:cdev_soc_, i)] = @constraint(model,
                                               [sc * dev; sc * G * w] in SecondOrderCone())
    return model[key] = @expression(model, dot(w, sigma, w))
end
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, flag::Bool)
    return if flag
        key = Symbol(:variance_risk_, i)
        model[key], key
    else
        key = Symbol(:dev_, i)
        model[key], key
    end
end
"""
"""
function variance_risk_bounds_val(flag::Bool, ub::Frontier)
    return _Frontier(; N = ub.N, factor = 1, flag = flag)
end
function variance_risk_bounds_val(flag::Bool, ub::VecNum)
    return flag ? ub : sqrt.(ub)
end
function variance_risk_bounds_val(flag::Bool, ub::Number)
    return flag ? ub : sqrt(ub)
end
function variance_risk_bounds_val(::Any, ::Nothing)
    return nothing
end
function rc_variance_constraints!(args...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Any, rc::LinearConstraint,
                                  variance_risk::AbstractJuMPScalar)
    sigma_W = model[Symbol(:sigma_W_, i)]
    sc = model[:sc]
    if !haskey(model, :rc_variance)
        @expression(model, rc_variance, true)
    end
    rc_key = Symbol(:rc_variance_, i)
    vsw = vec(diag(sigma_W))
    if !isnothing(rc.A_ineq)
        model[Symbol(rc_key, :_ineq)] = @constraint(model,
                                                    sc * (rc.A_ineq * vsw -
                                                          rc.B_ineq * variance_risk) <= 0)
    end
    if !isnothing(rc.A_eq)
        model[Symbol(rc_key, :_eq)] = @constraint(model,
                                                  sc *
                                                  (rc.A_eq * vsw - rc.B_eq * variance_risk) ==
                                                  0)
    end
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::Variance, opt::RkJuMPOpt,
                   pr::AbstractPriorResult, plg::Option{<:PhC_VecPhC}, args...; kwargs...)
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, plg)
    key = Symbol(:variance_risk_, i)
    variance_risk = set_variance_risk!(model, i, r, pr, sdp_flag, key)
    rc_variance_constraints!(model, i, rc, variance_risk)
    return variance_risk, sdp_flag
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::RkJuMPOpt,
                               pr::AbstractPriorResult, plg::Option{<:PhC_VecPhC}, args...;
                               kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    variance_risk, sdp_flag = set_risk!(model, i, r, opt, pr, plg, args...; kwargs...)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, sdp_flag)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return variance_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance,
                               opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any,
                               ::Any, b1::MatNum, args...; kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    rc = linear_constraints(r.rc, opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    key = Symbol(:variance_risk_, i)
    set_sdp_frc_constraints!(model)
    W = model[:frc_W]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model,
                                                        transpose(b1) * sigma * b1 * W)
    variance_risk = model[key] = @expression(model, tr(sigma_W))
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, true)
    ub = variance_risk_bounds_val(true, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return variance_risk
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::BoxUncertaintySet, args...)
    if !haskey(model, :Au)
        sc = model[:sc]
        W = model[:W]
        N = size(W, 1)
        @variables(model, begin
                       Au[1:N, 1:N] >= 0, Symmetric
                       Al[1:N, 1:N] >= 0, Symmetric
                   end)
        @constraint(model, cbucs_variance, sc * (Au - Al - W) == 0)
    end
    key = Symbol(:bucs_variance_risk_, i)
    Au = model[:Au]
    Al = model[:Al]
    ub = ucs.ub
    lb = ucs.lb
    ucs_variance_risk = model[key] = @expression(model, tr(Au * ub) - tr(Al * lb))
    return ucs_variance_risk, key
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::EllipseUncertaintySet,
                                sigma::MatNum)
    sc = model[:sc]
    if !haskey(model, :E)
        W = model[:W]
        N = size(W, 1)
        @variable(model, E[1:N, 1:N], Symmetric)
        @expression(model, WpE, W + E)
        @constraint(model, ceucs_variance, sc * E in PSDCone())
    end
    key = Symbol(:eucs_variance_risk_, i)
    WpE = model[:WpE]
    k = ucs.k
    G = cholesky(ucs.sigma).U
    t_eucs = model[Symbol(:t_eucs, i)] = @variable(model)
    x_eucs, ucs_variance_risk = model[Symbol(:x_eucs, i)], model[key] = @expressions(model,
                                                                                     begin
                                                                                         G *
                                                                                         vec(WpE)
                                                                                         tr(sigma *
                                                                                            WpE) +
                                                                                         k *
                                                                                         t_eucs
                                                                                     end)
    model[Symbol(:ge_soc, i)] = @constraint(model,
                                            [sc * t_eucs; sc * x_eucs] in SecondOrderCone())
    return ucs_variance_risk, key
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; rd::ReturnsResult = ReturnsResult(), kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    set_sdp_constraints!(model)
    ucs = r.ucs
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    ucs_variance_risk, key = set_ucs_variance_risk!(model, i, sigma_ucs(ucs, rd; kwargs...),
                                                    sigma)
    set_risk_bounds_and_expression!(model, opt, ucs_variance_risk, r.settings, key)
    return ucs_variance_risk
end
