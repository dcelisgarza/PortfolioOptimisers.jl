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
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key_dev = Symbol(:dev_, i)
    t_variance = model[Symbol(:t_variance_, i)] = @variable(model)
    dev = model[key_dev] = @variable(model)
    mu = pr.mu
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
                       Au[1:N, 1:N] >= 0, Symmetric
                       Al[1:N, 1:N] >= 0, Symmetric
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
function calc_risk_constraint_target(::LowOrderMoment{<:Any, <:Any, <:Any, Nothing},
                                     w::AbstractVector, mu::AbstractVector, args...)
    return dot(w, mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                       <:AbstractVector}, w::AbstractVector,
                                     args...)
    return dot(w, r.mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:Any, <:Real}, ::Any,
                                     ::Any, k)
    return r.mu * k
end
function set_risk_constraints!(model::JuMP.Model, i::Integer,
                               r::LowOrderMoment{<:Any, <:MeanAbsoluteDeviation, <:Any,
                                                 <:Any}, opt::MeanRiskEstimator,
                               pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    key = Symbol(:mad_risk_, i)
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    mad = model[Symbol(:mad_, i)] = @variable(model, [1:T], lower_bound = 0)
    mar_mad = model[Symbol(:mar_mad_, i)] = @expression(model, (net_X + mad) .- target)
    w = r.alg.w
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(w)
        @expression(model, mean(mad + mar_mad))
    else
        @expression(model, mean(mad + mar_mad, w))
    end
    model[Symbol(:cmar_mad_, i)] = @constraint(model, sc * mar_mad >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
    return nothing
end
function set_semi_variance_risk!(model::JuMP.Model, ::Quad, i::Integer, iTm1::Real,
                                 semi_variance, semi_dev, key::Symbol)
    return model[key] = @expression(model, iTm1 * dot(semi_variance, semi_variance))
end
function set_semi_variance_risk!(model::JuMP.Model, ::RSOC, i::Integer, iTm1::Real,
                                 semi_variance, semi_dev, key::Symbol)
    sc = model[:sc]
    tsemi_variance = model[Symbol(:tsemi_variance_, i)] = @variable(model)
    model[Symbol(:csemi_variance_rsoc_, i)] = @constraint(model,
                                                          [sc * tsemi_variance; 0.5;
                                                           sc * semi_variance] in
                                                          RotatedSecondOrderCone())
    return model[key] = @expression(model, iTm1 * tsemi_variance)
end
function set_semi_variance_risk!(model::JuMP.Model, ::SOC, i::Integer, iTm1::Real,
                                 semi_variance, semi_dev, key::Symbol)
    return model[key] = @expression(model, iTm1 * semi_dev^2)
end
function set_risk_constraints!(model::JuMP.Model, i::Integer,
                               r::LowOrderMoment{<:Any, <:SemiVariance, <:Any, <:Any},
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    key = Symbol(:semi_variance_risk_, i)
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    semi_variance = model[Symbol(:semi_variance_, i)] = @variable(model, [1:T],
                                                                  lower_bound = 0)
    semi_dev = model[Symbol(:semi_dev_, i)] = @variable(model)
    factor = T - r.alg.ddof
    semi_variance_risk = set_semi_variance_risk!(model, r.alg.formulation, i, inv(factor),
                                                 semi_variance, semi_dev, key)
    model[Symbol(:csemi_variance_mar_, i)], model[Symbol(:csemi_variance_soc_, i)] = @constraints(model,
                                                                                                  begin
                                                                                                      sc *
                                                                                                      ((net_X +
                                                                                                        semi_variance) .-
                                                                                                       target) >=
                                                                                                      0
                                                                                                      [sc *
                                                                                                       semi_dev
                                                                                                       sc *
                                                                                                       semi_variance] ∈
                                                                                                      SecondOrderCone()
                                                                                                  end)
    set_risk_bounds_and_expression!(model, opt, semi_variance_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer,
                               r::LowOrderMoment{<:Any, <:SemiDeviation, <:Any, <:Any},
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    key = Symbol(:semi_sd_risk_, i)
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    tsemi_sd = model[Symbol(:tsemi_sd_, i)] = @variable(model, [1:T], lower_bound = 0)
    semi_sd = model[Symbol(:semi_sd_, i)] = @variable(model)
    factor = T - r.alg.ddof
    semi_sd_risk = model[key] = @expression(model, semi_sd / sqrt(factor))
    model[Symbol(:csemi_variance_mar_, i)], model[Symbol(:csemi_variance_soc_, i)] = @constraints(model,
                                                                                                  begin
                                                                                                      sc *
                                                                                                      ((net_X +
                                                                                                        tsemi_sd) .-
                                                                                                       target) >=
                                                                                                      0
                                                                                                      [sc *
                                                                                                       semi_sd
                                                                                                       sc *
                                                                                                       tsemi_sd] ∈
                                                                                                      SecondOrderCone()
                                                                                                  end)

    set_risk_bounds_and_expression!(model, opt, semi_sd_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer,
                               r::LowOrderMoment{<:Any, <:FirstLowerMoment, <:Any, <:Any},
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    key = Symbol(:flm_risk_, i)
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    flm = model[Symbol(:flm_, i)] = @variable(model, [1:T], lower_bound = 0)
    flm_risk = model[key] = @expression(model, sum(flm) / T)
    model[Symbol(:cflm_mar_, i)] = @constraint(model, sc * ((net_X + flm) .- target) >= 0)
    set_risk_bounds_and_expression!(model, opt, flm_risk, r.settings, key)
    return nothing
end
function set_wr_risk_expression!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :wr_risk)
        return model[:wr_risk]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    @variable(model, wr_risk)
    @constraint(model, cwr, sc * (wr_risk .+ net_X) >= 0)
    return wr_risk
end
function set_risk_constraints!(model::JuMP.Model, ::Integer, r::WorstRealisation,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    if haskey(model, :wr_risk)
        return nothing
    end
    wr_risk = set_wr_risk_expression!(model, pr.X)
    set_risk_bounds_and_expression!(model, opt, wr_risk, r.settings, :wr_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, ::Integer, r::Range,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    if haskey(model, :range_risk)
        return nothing
    end
    sc = model[:sc]
    wr_risk = set_wr_risk_expression!(model, pr.X)
    net_X = model[:net_X]
    @variable(model, br_risk)
    @expression(model, range_risk, wr_risk - br_risk)
    @constraint(model, cbr, sc * (br_risk .+ net_X) <= 0)
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, :range_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::ConditionalValueatRisk,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    iat = inv(r.alpha * T)
    key = Symbol(:cvar_risk_, i)
    var = model[Symbol(:var_, i)] = @variable(model)
    z_cvar = model[Symbol(:z_cvar_, i)] = @variable(model, [1:T], lower_bound = 0)
    cvar_risk = model[key] = @expression(model, var + sum(z_cvar) * iat)
    model[Symbol(:ccvar_, i)] = @constraint(model, sc * ((z_cvar + net_X) .+ var) >= 0)
    set_risk_bounds_and_expression!(model, opt, cvar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer,
                               r::DistributionallyRobustConditionalValueatRisk,
                               opt::MeanRiskEstimator, pr::AbstractPriorResult, args...)
    sc = model[:sc]
    w = model[:w]
    net_X = set_net_portfolio_returns!(model, pr.X)
    Xap1 = set_portfolio_returns_plus_one!(model, pr.X)
    key = Symbol(:drcvar_risk_, i)
    T, N = size(pr.X)

    b1 = r.l
    alpha = r.alpha
    radius = r.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))

    lb = model[Symbol(:lb_, i)] = @variable(model)
    tau = model[Symbol(:tau_, i)] = @variable(model)
    s, tu_drcvar, tv_drcvar, u, v = model[Symbol(:s_, i)], model[Symbol(:tu_drcvar_, i)], model[Symbol(:tv_drcvar_, i)], model[Symbol(:u_, i)], model[Symbol(:v_, i)] = @variables(model,
                                                                                                                                                                                   begin
                                                                                                                                                                                       [1:T]
                                                                                                                                                                                       [1:T]
                                                                                                                                                                                       [1:T]
                                                                                                                                                                                       [1:T,
                                                                                                                                                                                        1:N],
                                                                                                                                                                                       (lower_bound = 0)
                                                                                                                                                                                       [1:T,
                                                                                                                                                                                        1:N],
                                                                                                                                                                                       (lower_bound = 0)
                                                                                                                                                                                   end)
    cu_drcvar, cv_drcvar, cu_drcvar_infnorm, cv_drcvar_infnorm, cu_drcvar_lb, cv_drcvar_lb = model[Symbol(:cu_drcvar_, i)], model[Symbol(:cv_drcvar_, i)], model[Symbol(:cu_drcvar_infnorm_, i)], model[Symbol(:cv_drcvar_infnorm_, i)], model[Symbol(:cu_drcvar_lb_, i)], model[Symbol(:cv_drcvar_lb_, i)] = @constraints(model,
                                                                                                                                                                                                                                                                                                                           begin
                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                               (b1 *
                                                                                                                                                                                                                                                                                                                                tau .+
                                                                                                                                                                                                                                                                                                                                (a1 *
                                                                                                                                                                                                                                                                                                                                 net_X +
                                                                                                                                                                                                                                                                                                                                 vec(sum(u .*
                                                                                                                                                                                                                                                                                                                                         Xap1;
                                                                                                                                                                                                                                                                                                                                         dims = 2)) -
                                                                                                                                                                                                                                                                                                                                 s)) <=
                                                                                                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                               (b2 *
                                                                                                                                                                                                                                                                                                                                tau .+
                                                                                                                                                                                                                                                                                                                                (a2 *
                                                                                                                                                                                                                                                                                                                                 net_X +
                                                                                                                                                                                                                                                                                                                                 vec(sum(v .*
                                                                                                                                                                                                                                                                                                                                         Xap1;
                                                                                                                                                                                                                                                                                                                                         dims = 2)) -
                                                                                                                                                                                                                                                                                                                                 s)) <=
                                                                                                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                                                                                                tu_drcvar[i]
                                                                                                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                                                                                                (-view(u,
                                                                                                                                                                                                                                                                                                                                       i,
                                                                                                                                                                                                                                                                                                                                       :) -
                                                                                                                                                                                                                                                                                                                                 a1 *
                                                                                                                                                                                                                                                                                                                                 w)] in
                                                                                                                                                                                                                                                                                                                               MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                    N)
                                                                                                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                                                                                                tv_drcvar[i]
                                                                                                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                                                                                                (-view(v,
                                                                                                                                                                                                                                                                                                                                       i,
                                                                                                                                                                                                                                                                                                                                       :) -
                                                                                                                                                                                                                                                                                                                                 a2 *
                                                                                                                                                                                                                                                                                                                                 w)] in
                                                                                                                                                                                                                                                                                                                               MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                    N)
                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                               (tu_drcvar .-
                                                                                                                                                                                                                                                                                                                                lb) <=
                                                                                                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                               (tv_drcvar .-
                                                                                                                                                                                                                                                                                                                                lb) <=
                                                                                                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                                                                                                           end)
    drcvar_risk = model[key] = @expression(model, radius * lb + mean(s))
    set_risk_bounds_and_expression!(model, opt, drcvar_risk, r.settings, key)
    return nothing
end