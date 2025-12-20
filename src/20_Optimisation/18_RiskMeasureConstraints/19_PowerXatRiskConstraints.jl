function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pvar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    ip = inv(r.p)
    pvar_eta, pvar_t, pvar_w, pvar_v = model[Symbol(:pvar_eta_, i)], model[Symbol(:pvar_t_, i)], model[Symbol(:pvar_w_, i)], model[Symbol(:pvar_v_, i)] = @variables(model,
                                                                                                                                                                     begin
                                                                                                                                                                         ()
                                                                                                                                                                         ()
                                                                                                                                                                         [1:T],
                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                         [1:T]
                                                                                                                                                                     end)

    wi = nothing_scalar_array_selector(r.w, pr.w)
    iaT = if isnothing(wi)
        model[Symbol(:cpvar_eq_, i)] = @constraint(model, sc * (sum(pvar_v) - pvar_t) <= 0)
        inv(r.alpha * T^ip)
    else
        model[Symbol(:cpvar_eq_, i)] = @constraint(model,
                                                   sc * (dot(wi, pvar_v) - pvar_t) <= 0)
        inv(r.alpha * sum(wi)^ip)
    end
    model[Symbol(:cpvar_, i)], model[Symbol(:cpvar_pcone_, i)] = @constraints(model,
                                                                              begin
                                                                                  sc *
                                                                                  ((net_X +
                                                                                    pvar_w) .+
                                                                                   pvar_eta) >=
                                                                                  0
                                                                                  [i = 1:T],
                                                                                  [sc *
                                                                                   pvar_v[i],
                                                                                   sc *
                                                                                   pvar_t,
                                                                                   sc *
                                                                                   pvar_w[i]] in
                                                                                  MOI.PowerCone(ip)
                                                                              end)
    pvar_risk = model[key] = @expression(model, pvar_eta + iaT * pvar_t)
    set_risk_bounds_and_expression!(model, opt, pvar_risk, r.settings, key)
    return pvar_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pvar_range_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    ipa = inv(r.pa)
    ipb = inv(r.pb)
    pvar_eta_l, pvar_t_l, pvar_w_l, pvar_v_l, pvar_eta_h, pvar_t_h, pvar_w_h, pvar_v_h = model[Symbol(:pvar_eta_l_, i)], model[Symbol(:pvar_t_l_, i)], model[Symbol(:pvar_w_l_, i)], model[Symbol(:pvar_v_l_, i)], model[Symbol(:pvar_eta_h_, i)], model[Symbol(:pvar_t_h_, i)], model[Symbol(:pvar_w_h_, i)], model[Symbol(:pvar_v_h_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                                                                                         begin
                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                             [1:T],
                                                                                                                                                                                                                                                                                                                                                             (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                             [1:T],
                                                                                                                                                                                                                                                                                                                                                             (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                         end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    iaT, ibT = if isnothing(wi)
        model[Symbol(:cpvar_eq_l_, i)], model[Symbol(:cpvar_eq_h_, i)] = @constraints(model,
                                                                                      begin
                                                                                          sc *
                                                                                          (sum(pvar_v_l) -
                                                                                           pvar_t_l) <=
                                                                                          0
                                                                                          sc *
                                                                                          (sum(pvar_v_h) -
                                                                                           pvar_t_h) >=
                                                                                          0
                                                                                      end)
        inv(r.alpha * T^ipa), inv(r.beta * T^ipb)
    else
        sw = sum(wi)
        model[Symbol(:cpvar_eq_l_, i)], model[Symbol(:cpvar_eq_h_, i)] = @constraints(model,
                                                                                      begin
                                                                                          sc *
                                                                                          (dot(wi,
                                                                                               pvar_v_l) -
                                                                                           pvar_t_l) <=
                                                                                          0
                                                                                          sc *
                                                                                          (dot(wi,
                                                                                               pvar_v_h) -
                                                                                           pvar_t_h) >=
                                                                                          0
                                                                                      end)
        inv(r.alpha * sw^ipa), inv(r.beta * sw^ipb)
    end
    model[Symbol(:cpvar_l_, i)], model[Symbol(:cpvar_pcone_l_, i)], model[Symbol(:cpvar_h_, i)] = @constraints(model,
                                                                                                               begin
                                                                                                                   sc *
                                                                                                                   ((net_X +
                                                                                                                     pvar_w_l) .+
                                                                                                                    pvar_eta_l) >=
                                                                                                                   0
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    pvar_v_l[i],
                                                                                                                    sc *
                                                                                                                    pvar_t_l,
                                                                                                                    sc *
                                                                                                                    pvar_w_l[i]] in
                                                                                                                   MOI.PowerCone(ipa)
                                                                                                                   sc *
                                                                                                                   ((net_X +
                                                                                                                     pvar_w_h) .+
                                                                                                                    pvar_eta_h) <=
                                                                                                                   0
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    -pvar_v_h[i],
                                                                                                                    sc *
                                                                                                                    -pvar_t_h,
                                                                                                                    sc *
                                                                                                                    -pvar_w_h[i]] in
                                                                                                                   MOI.PowerCone(ipb)
                                                                                                               end)
    pvar_risk_l, pvar_risk_h = model[Symbol(:pvar_risk_l_, i)], model[Symbol(:pvar_risk_h_, i)] = @expressions(model,
                                                                                                               begin
                                                                                                                   pvar_eta_l +
                                                                                                                   iaT *
                                                                                                                   pvar_t_l
                                                                                                                   pvar_eta_h +
                                                                                                                   ibT *
                                                                                                                   pvar_t_h
                                                                                                               end)
    pvar_range_risk = model[key] = @expression(model, pvar_risk_l - pvar_risk_h)
    set_risk_bounds_and_expression!(model, opt, pvar_range_risk, r.settings, key)
    return pvar_range_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pdar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    ip = inv(r.p)
    pdar_eta, pdar_t, pdar_w, pdar_v = model[Symbol(:pdar_eta_, i)], model[Symbol(:pdar_t_, i)], model[Symbol(:pdar_w_, i)], model[Symbol(:pdar_v_, i)] = @variables(model,
                                                                                                                                                                     begin
                                                                                                                                                                         ()
                                                                                                                                                                         ()
                                                                                                                                                                         [1:T],
                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                         [1:T]
                                                                                                                                                                     end)
    iaT = inv(r.alpha * T^ip)
    model[Symbol(:cpdar_eq_, i)] = @constraint(model, sc * (sum(pdar_v) - pdar_t) <= 0)
    model[Symbol(:cpdar_, i)], model[Symbol(:cpdar_pcone_, i)] = @constraints(model,
                                                                              begin
                                                                                  sc *
                                                                                  ((pdar_w -
                                                                                    view(dd,
                                                                                         2:(T + 1))) .+
                                                                                   pdar_eta) >=
                                                                                  0
                                                                                  [i = 1:T],
                                                                                  [sc *
                                                                                   pdar_v[i],
                                                                                   sc *
                                                                                   pdar_t,
                                                                                   sc *
                                                                                   pdar_w[i]] in
                                                                                  MOI.PowerCone(ip)
                                                                              end)
    pdar_risk = model[key] = @expression(model, pdar_eta + iaT * pdar_t)
    set_risk_bounds_and_expression!(model, opt, pdar_risk, r.settings, key)
    return pdar_risk
end
