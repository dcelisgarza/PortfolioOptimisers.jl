function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:evar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_evar, z_evar, u_evar = model[Symbol(:t_evar_, i)], model[Symbol(:z_evar_, i)], model[Symbol(:u_evar_, i)] = @variables(model,
                                                                                                                             begin
                                                                                                                                 ()
                                                                                                                                 (),
                                                                                                                                 (lower_bound = 0)
                                                                                                                                 [1:T]
                                                                                                                             end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    at = if isnothing(wi)
        model[Symbol(:cevar_, i)] = @constraint(model, sc * (sum(u_evar) - z_evar) <= 0)
        r.alpha * T
    else
        model[Symbol(:cevar_, i)] = @constraint(model, sc * (dot(wi, u_evar) - z_evar) <= 0)
        r.alpha * sum(wi)
    end
    model[Symbol(:cevar_exp_cone_, i)] = @constraint(model, [i = 1:T],
                                                     [sc * (-net_X[i] - t_evar),
                                                      sc * z_evar, sc * u_evar[i]] in
                                                     MOI.ExponentialCone())
    evar_risk = model[Symbol(:evar_risk_, i)] = @expression(model,
                                                            t_evar - z_evar * log(at))
    set_risk_bounds_and_expression!(model, opt, evar_risk, r.settings, key)
    return evar_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:evar_risk_range_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_evar_l, z_evar_l, u_evar_l, t_evar_h, z_evar_h, u_evar_h = model[Symbol(:t_evar_l_, i)], model[Symbol(:z_evar_l_, i)], model[Symbol(:u_evar_l_, i)], model[Symbol(:t_evar_h_, i)], model[Symbol(:z_evar_h_, i)], model[Symbol(:u_evar_h_, i)] = @variables(model,
                                                                                                                                                                                                                                                                 begin
                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                     (),
                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                     (),
                                                                                                                                                                                                                                                                     (upper_bound = 0)
                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                 end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    at, bt = if isnothing(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (sum(u_evar_l) -
                                                                                     z_evar_l) <=
                                                                                    0
                                                                                    sc *
                                                                                    (sum(u_evar_h) -
                                                                                     z_evar_h) >=
                                                                                    0
                                                                                end)
        r.alpha * T, r.beta * T
    else
        sw = sum(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         u_evar_l) -
                                                                                     z_evar_l) <=
                                                                                    0
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         u_evar_h) -
                                                                                     z_evar_h) >=
                                                                                    0
                                                                                end)
        r.alpha * sw, r.beta * sw
    end
    model[Symbol(:cevar_exp_cone_l_, i)], model[Symbol(:cevar_exp_cone_h_, i)] = @constraints(model,
                                                                                              begin
                                                                                                  [i = 1:T],
                                                                                                  [sc *
                                                                                                   (-net_X[i] -
                                                                                                    t_evar_l),
                                                                                                   sc *
                                                                                                   z_evar_l,
                                                                                                   sc *
                                                                                                   u_evar_l[i]] in
                                                                                                  MOI.ExponentialCone()
                                                                                                  [i = 1:T],
                                                                                                  [sc *
                                                                                                   (net_X[i] +
                                                                                                    t_evar_h),
                                                                                                   -sc *
                                                                                                   z_evar_h,
                                                                                                   -sc *
                                                                                                   u_evar_h[i]] in
                                                                                                  MOI.ExponentialCone()
                                                                                              end)
    evar_risk_l, evar_risk_h = model[Symbol(:evar_risk_l_, i)], model[Symbol(:evar_risk_h_, i)] = @expressions(model,
                                                                                                               begin
                                                                                                                   t_evar_l -
                                                                                                                   z_evar_l *
                                                                                                                   log(at)
                                                                                                                   t_evar_h -
                                                                                                                   z_evar_h *
                                                                                                                   log(bt)
                                                                                                               end)
    evar_risk_range = model[key] = @expression(model, evar_risk_l - evar_risk_h)
    set_risk_bounds_and_expression!(model, opt, evar_risk_range, r.settings, key)
    return evar_risk_range
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:edar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    at = r.alpha * T
    t_edar, z_edar, u_edar = model[Symbol(:t_edar_, i)], model[Symbol(:z_edar_, i)], model[Symbol(:u_edar_, i)] = @variables(model,
                                                                                                                             begin
                                                                                                                                 ()
                                                                                                                                 (),
                                                                                                                                 (lower_bound = 0)
                                                                                                                                 [1:T]
                                                                                                                             end)
    edar_risk = model[key] = @expression(model, t_edar - z_edar * log(at))
    model[Symbol(:cedar_, i)], model[Symbol(:cedar_exp_cone_, i)] = @constraints(model,
                                                                                 begin
                                                                                     sc *
                                                                                     (sum(u_edar) -
                                                                                      z_edar) <=
                                                                                     0
                                                                                     [i = 1:T],
                                                                                     [sc *
                                                                                      (dd[i + 1] -
                                                                                       t_edar),
                                                                                      sc *
                                                                                      z_edar,
                                                                                      sc *
                                                                                      u_edar[i]] in
                                                                                     MOI.ExponentialCone()
                                                                                 end)
    set_risk_bounds_and_expression!(model, opt, edar_risk, r.settings, key)
    return edar_risk
end
