function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk, z_var = model[key], model[Symbol(:z_var_, i)] = JuMP.@variables(model,
                                                                              begin
                                                                                  ()
                                                                                  [1:T],
                                                                                  (binary = true)
                                                                              end)
    alpha = r.alpha
    wi = nothing_scalar_array_selector(r.w, pr.w)
    if isnothing(wi)
        model[Symbol(:csvar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (sum(z_var) - alpha * T + s * T) <= 0)
    else
        sw = sum(wi)
        model[Symbol(:csvar_, i)] = JuMP.@constraint(model,
                                                     sc * (LinearAlgebra.dot(wi, z_var) -
                                                           alpha * sw + s * sw) <= 0)
    end
    model[Symbol(:cvar_, i)] = JuMP.@constraint(model,
                                                sc * ((net_X + b * z_var) .+ var_risk) >= 0)
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return var_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_range_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk_l, z_var_l, var_risk_h, z_var_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:z_var_l_, i)], model[Symbol(:var_risk_h_, i)], model[Symbol(:z_var_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                          begin
                                                                                                                                                                                              ()
                                                                                                                                                                                              [1:T],
                                                                                                                                                                                              (binary = true)
                                                                                                                                                                                              ()
                                                                                                                                                                                              [1:T],
                                                                                                                                                                                              (binary = true)
                                                                                                                                                                                          end)
    alpha = r.alpha
    beta = r.beta
    wi = nothing_scalar_array_selector(r.w, pr.w)
    if isnothing(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (sum(z_var_l) -
                                                                                          alpha *
                                                                                          T +
                                                                                          s *
                                                                                          T) <=
                                                                                         0
                                                                                         sc *
                                                                                         (sum(z_var_h) -
                                                                                          beta *
                                                                                          T +
                                                                                          s *
                                                                                          T) <=
                                                                                         0
                                                                                     end)
    else
        sw = sum(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            z_var_l) -
                                                                                          alpha *
                                                                                          sw +
                                                                                          s *
                                                                                          sw) <=
                                                                                         0
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            z_var_h) -
                                                                                          beta *
                                                                                          sw +
                                                                                          s *
                                                                                          sw) <=
                                                                                         0
                                                                                     end)
    end
    model[Symbol(:cvar_, i)] = JuMP.@constraints(model,
                                                 begin
                                                     sc *
                                                     ((net_X + b * z_var_l) .+ var_risk_l) >=
                                                     0
                                                     sc *
                                                     ((net_X + b * z_var_h) .+ var_risk_h) <=
                                                     0
                                                 end)
    var_range_risk = model[key] = JuMP.@expression(model, var_risk_l - var_risk_h)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
    return var_range_risk
end
function compute_value_at_risk_z(dist::Distributions.Normal, alpha::Number)
    return Distributions.cquantile(dist, alpha)
end
function compute_value_at_risk_z(dist::Distributions.TDist, alpha::Number)
    d = dof(dist)
    @argcheck(d > 2)
    return Distributions.cquantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_z(::Distributions.Laplace, alpha::Number)
    return -log(2 * alpha) / sqrt(2)
end
function compute_value_at_risk_cz(dist::Distributions.Normal, alpha::Number)
    return quantile(dist, alpha)
end
function compute_value_at_risk_cz(dist::Distributions.TDist, alpha::Number)
    d = dof(dist)
    @argcheck(d > 2)
    return quantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_cz(::Distributions.Laplace, alpha::Number)
    return -log(2 * (one(alpha) - alpha)) / sqrt(2)
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any,
                                              <:DistributionValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = if isnothing(alg.sigma)
        get_chol_or_sigma_pm(model, pr)
    else
        LinearAlgebra.cholesky(alg.sigma).U
    end
    w = model[:w]
    sc = model[:sc]
    z = compute_value_at_risk_z(r.alg.dist, r.alpha)
    key = Symbol(:var_risk_, i)
    g_var = model[Symbol(:g_var_, i)] = JuMP.@variable(model)
    var_risk = model[key] = JuMP.@expression(model, -LinearAlgebra.dot(mu, w) + z * g_var)
    model[Symbol(:cvar_soc_, i)] = JuMP.@constraint(model,
                                                    [sc * g_var; sc * G * w] in
                                                    JuMP.SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return var_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:DistributionValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = if isnothing(alg.sigma)
        get_chol_or_sigma_pm(model, pr)
    else
        LinearAlgebra.cholesky(alg.sigma).U
    end
    w = model[:w]
    sc = model[:sc]
    dist = r.alg.dist
    z_l = compute_value_at_risk_z(dist, r.alpha)
    z_h = compute_value_at_risk_cz(dist, r.beta)
    key = Symbol(:var_range_risk_, i)
    g_var = model[Symbol(:g_var_range_, i)] = JuMP.@variable(model)
    var_range_mu = model[Symbol(:var_range_mu_, i)] = JuMP.@expression(model,
                                                                       LinearAlgebra.dot(mu,
                                                                                         w))
    var_risk_l, var_risk_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:var_risk_h_, i)] = JuMP.@expressions(model,
                                                                                                                begin
                                                                                                                    -var_range_mu +
                                                                                                                    z_l *
                                                                                                                    g_var
                                                                                                                    -var_range_mu +
                                                                                                                    z_h *
                                                                                                                    g_var
                                                                                                                end)
    var_range_risk = model[key] = JuMP.@expression(model, var_risk_l - var_risk_h)
    model[Symbol(:cvar_range_soc_, i)] = JuMP.@constraints(model,
                                                           begin
                                                               [sc * g_var; sc * G * w] in
                                                               JuMP.SecondOrderCone()
                                                           end)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
    return var_range_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:dar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    dar_risk, z_dar = model[key], model[Symbol(:z_dar_, i)] = JuMP.@variables(model,
                                                                              begin
                                                                                  ()
                                                                                  [1:T],
                                                                                  (binary = true)
                                                                              end)
    alpha = r.alpha
    wi = nothing_scalar_array_selector(r.w, pr.w)
    if isnothing(wi)
        model[Symbol(:csdar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (sum(z_dar) - alpha * T + s * T) <= 0)
    else
        sw = sum(wi)
        model[Symbol(:csdar_, i)] = JuMP.@constraint(model,
                                                     sc * (LinearAlgebra.dot(wi, z_dar) -
                                                           alpha * sw + s * sw) <= 0)
    end
    model[Symbol(:cdar_, i)] = JuMP.@constraint(model,
                                                sc *
                                                ((-view(dd, 2:(T + 1)) + b * z_dar) .+
                                                 dar_risk) >= 0)
    set_risk_bounds_and_expression!(model, opt, dar_risk, r.settings, key)
    return dar_risk
end
