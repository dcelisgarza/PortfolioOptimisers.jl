function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !haskey(model, :G)
        G = cholesky(pr.sigma).U
        @expression(model, G, G)
    end
    return model[:G]
end
function get_chol_or_sigma_pm(model::JuMP.Model, pr::FactorPriorResult)
    if !haskey(model, :G)
        G = pr.chol
        @expression(model, G, G)
    end
    return model[:G]
end
function set_risk_upper_bound!(args...)
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model, ::MeanRiskEstimator, r_expr, ub::Real,
                               key)
    k = model[:k]
    sc = model[:sc]
    model[Symbol(key, :_ub)] = @constraint(model, sc * r_expr <= sc * ub * k)
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
function set_risk_bounds_and_expression!(model::JuMP.Model, opt::MeanRiskEstimator, r_expr,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_variance_risk_bounds_and_expression!(model::JuMP.Model, opt::MeanRiskEstimator,
                                                  r_expr_ub, ub::Union{Nothing, <:Real},
                                                  key::Symbol, r_expr,
                                                  settings::RiskMeasureSettings)
    set_risk_upper_bound!(model, opt, r_expr_ub, ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::StandardDeviation,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key = Symbol(:sd_risk_, i)
    sd_risk = model[key] = @variable(model)
    model[Symbol(key, :_soc)] = @constraint(model,
                                            [sc * sd_risk; sc * G * w] ∈ SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, key)
    return nothing
end
function sdp_rc_variance_flag!(::JuMP.Model, ::MeanRiskEstimator, ::Nothing)
    return false
end
function sdp_rc_variance_flag!(model::JuMP.Model, ::MeanRiskEstimator,
                               ::LinearConstraintResult)
    set_sdp_constraints!(model)
    return true
end
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool,
                            cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                        <:IntegerPhilogenyResult},
                            nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                        <:IntegerPhilogenyResult})
    return if rc_flag ||
              haskey(model, :rc_variance) ||
              isa(cplg, SemiDefinitePhilogenyResult) ||
              isa(nplg, SemiDefinitePhilogenyResult)
        true
    else
        false
    end
end
function set_variance_risk!(model::JuMP.Model, i::Integer, r::Variance,
                            pr::AbstractPriorResult, flag::Bool, key::Symbol)
    if flag
        set_sdp_variance_risk!(model, i, r, pr, key)
    else
        set_variance_risk!(model, i, r, pr, key)
    end
    return nothing
end
function set_sdp_variance_risk!(model::JuMP.Model, i::Integer, r::Variance,
                                pr::AbstractPriorResult, key::Symbol)
    W = model[:W]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model, sigma * W)
    model[key] = @expression(model, tr(sigma_W))
    return nothing
end
function set_variance_risk!(model::JuMP.Model, i::Integer,
                            r::Variance{<:Any, <:SOC, <:Any, <:Any},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key_dev = Symbol(:dev_, i)
    dev = model[key_dev] = @variable(model)
    model[key] = @expression(model, dev^2)
    model[Symbol(key_dev, :_soc)] = @constraint(model,
                                                [sc * dev; sc * G * w] ∈ SecondOrderCone())
    return nothing
end
function set_variance_risk!(model::JuMP.Model, i::Integer,
                            r::Variance{<:Any, <:Quad, <:Any, <:Any},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    model[key] = @expression(model, dot(w, sigma, w))
    key_dev = Symbol(:dev_, i)
    dev = model[key_dev] = @variable(model)
    model[Symbol(key_dev, :_soc)] = @constraint(model,
                                                [sc * dev; sc * G * w] ∈ SecondOrderCone())
    return nothing
end
function set_variance_risk!(model::JuMP.Model, i::Integer,
                            r::Variance{<:Any, <:RSOC, <:Any, <:Any},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    set_net_portfolio_returns!(model, pr.X)
    net_X = model[:net_X]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key_dev = Symbol(:dev_, i)
    t_variance = model[Symbol(:t_variance_, i)] = @variable(model)
    dev = model[key_dev] = @variable(model)
    mu = pr.mu
    T = length(net_X)
    variance = model[Symbol(:variance_, i)] = @expression(model, net_X .- dot(mu, w))
    model[key] = @expression(model, t_variance / (T - one(T)))
    model[Symbol(key_dev, :_soc)] = @constraint(model,
                                                [sc * dev; sc * G * w] ∈ SecondOrderCone())
    model[Symbol(key, :_rsoc)] = @constraint(model,
                                             [sc * t_variance; 0.5;
                                              sc * variance] in RotatedSecondOrderCone())
    return nothing
end
function variance_risk_bounds_expr(model::JuMP.Model, i::Integer, flag::Bool)
    return if flag
        key = Symbol(:variance_risk_, i)
        model[key], key
    else
        key = Symbol(:dev_, i)
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
    sigma_W = model[Symbol(:sigma_W_, i)]
    sc = model[:sc]
    if !haskey(model, :rc_variance)
        @expression(model, rc_variance, true)
    end
    rc_key = Symbol(:rc_variance_, i)
    vsw = vec(diag(sigma_W))
    if !isnothing(rc.A_ineq)
        model[Symbol(rc_key, :_ineq)] = @constraint(model,
                                                    sc * rc.A_ineq * vsw <=
                                                    sc * rc.B_ineq * variance_risk)
    end
    if !isnothing(rc.A_eq)
        model[Symbol(rc_key, :_eq)] = @constraint(model,
                                                  sc * rc.A_eq * vsw ==
                                                  sc * rc.B_eq * variance_risk)
    end
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::Variance,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult},
                               nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult})
    if !haskey(model, :variance_flag) && r.settings.rke
        @expression(model, variance_flag, true)
    end
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, cplg, nplg)
    key = Symbol(:variance_risk_, i)
    set_variance_risk!(model, i, r, pr, sdp_flag, key)
    variance_risk = model[key]
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, sdp_flag)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::Variance,
                               opt::FactorRiskContribution, pr::AbstractPriorResult,
                               M::AbstractMatrix)
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    key = Symbol(:variance_risk_, i)
    set_sdp_frc_constraints!(model)
    W = model[:W]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model, transpose(M) * sigma * M * W)
    variance_risk = model[key] = @expression(model, tr(sigma_W))
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, true)
    ub = variance_risk_bounds_val(true, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return nothing
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Integer, ucs::BoxUncertaintySetResult,
                                args...)
    if !haskey(model, :Au)
        sc = model[:sc]
        W = model[:W]
        N = size(W, 1)
        @variables(model, begin
                       Au[1:N, 1:N] .>= 0, Symmetric
                       Al[1:N, 1:N] .>= 0, Symmetric
                   end)
        @constraint(model, cbucs_variance, sc * (Au - Al) == sc * W)
    end
    key = Symbol(:bucs_variance_risk_, i)
    Au = model[:Au]
    Al = model[:Al]
    ub = ucs.ub
    lb = ucs.lb
    ucs_variance_risk = model[key] = @expression(model, tr(Au * ub) - tr(Al * lb))
    return ucs_variance_risk, key
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Integer,
                                ucs::EllipseUncertaintySetResult,
                                r_sigma::Union{Nothing, <:AbstractMatrix},
                                sigma::AbstractMatrix)
    sc = model[:sc]
    if !haskey(model, :E)
        W = model[:W]
        N = size(W, 1)
        @variable(model, E[1:N, 1:N], Symmetric)
        @expression(model, WpE, W + E)
        @constraint(model, ceucs_variance, sc * E ∈ PSDCone())
    end
    if !isnothing(r_sigma)
        sigma = r_sigma
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
                                            [sc * t_eucs; sc * x_eucs] ∈ SecondOrderCone())
    return ucs_variance_risk, key
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::UncertaintySetVariance,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    set_sdp_constraints!(model)
    ucs = r.ucs
    r_sigma = r.sigma
    X = pr.X
    sigma = pr.sigma
    ucs_variance_risk, key = set_ucs_variance_risk!(model, i, sigma_ucs(ucs, X), r_sigma,
                                                    sigma)
    set_risk_bounds_and_expression!(model, opt, ucs_variance_risk, r.settings, key)
    return nothing
end
