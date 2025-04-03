function set_return_bounds!(args...)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Real)
    sc = model[:sc]
    k = model[:k]
    ret = model[:ret]
    @constraint(model, ret_lb, sc * ret >= sc * lb * k)
    return nothing
end
function set_max_ratio_return_constraints!(args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio,
                                           mu::AbstractVector)
    sc = model[:sc]
    k = model[:k]
    ohf = model[:ohf]
    ret = model[:ret]
    if haskey(model, :bucs_w) || haskey(model, :t_eucs_gw) || all(mu .<= zero(eltype(mu)))
        risk = model[:risk]
        rf = obj.rf
        add_to_expression!(ret, -rf, k)
        @constraint(model, sr_risk, sc * risk <= sc * ohf)
    else
        @constraint(model, sr_ret, sc * (ret - rf * k) == sc * ohf)
    end
    return nothing
end
function set_return_constraints!(model::JuMP.Model, pret::ArithmeticReturn{<:Any, Nothing},
                                 obj::ObjectiveFunction, pm::AbstractPriorModel)
    w = model[:w]
    fees = model[:fees]
    lb = pret.lb
    mu = pm.mu
    @expression(model, ret, dot(mu, w) - fees)
    set_return_bounds!(model, lb)
    set_max_ratio_return_constraints!(model, obj, mu)
    return nothing
end
function set_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySet,
                                 mu::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    fees = model[:fees]
    N = length(w)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    @variable(model, bucs_w[1:N])
    @constraint(model, bucs_ret[i = 1:N], [sc * bucs_w[i]; sc * w[i]] ∈ MOI.NormOneCone(2))
    @expression(model, ret, dot(mu, w) - fees - dot(d_mu, bucs_w))
    return nothing
end
function set_return_constraints!(model::JuMP.Model, ucs::EllipseUncertaintySet,
                                 mu::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    fees = model[:fees]
    G = cholesky(ucs.sigma).U
    k = ucs.k
    @expression(model, x_eucs_w, G * w)
    @variable(model, t_eucs_gw)
    @constraint(model, eucs_ret, [sc * t_eucs_gw; sc * x_eucs_w] ∈ SecondOrderCone())
    @expression(model, ret, dot(mu, w) - fees - k * t_eucs_gw)
    return nothing
end
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{<:Any,
                                                        Union{<:UncertaintySet,
                                                              <:UncertaintySetEstimator}},
                                 obj::ObjectiveFunction, pm::AbstractPriorModel)
    lb = pret.lb
    ucs = pret.ucs
    X = pm.X
    mu = pm.mu
    set_return_constraints!(model, mu_ucs(ucs, X), mu)
    set_return_bounds!(model, lb)
    set_max_ratio_return_constraints!(model, obj, mu)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio, k, sc, ret)
    ohf = model[:ohf]
    risk = model[:risk]
    rf = obj.rf
    add_to_expression!(ret, -rf, k)
    @constraint(model, sr_ekelly_risk, sc * risk <= sc * ohf)
end
function set_return_constraints!(model::JuMP.Model, pret::ExactKellyReturn,
                                 obj::ObjectiveFunction, pm::AbstractPriorModel)
    k = model[:k]
    sc = model[:sc]
    fees = model[:fees]
    lb = pret.lb
    X = pm.X
    set_portfolio_returns!(model, X)
    X = model[:X]
    T = length(X)
    @variable(model, t_ekelly[1:T])
    @expression(model, ret, sum(t_ekelly) / T - fees)
    set_max_ratio_return_constraints!(model, obj, k, sc, ret)
    @expression(model, kret, k .+ X)
    @constraint(model, ekelly_ret[i = 1:T],
                [sc * t_ekelly[i], sc * k, sc * kret[i]] ∈ MOI.ExponentialCone())
    set_return_bounds!(model, lb)
    return nothing
end
