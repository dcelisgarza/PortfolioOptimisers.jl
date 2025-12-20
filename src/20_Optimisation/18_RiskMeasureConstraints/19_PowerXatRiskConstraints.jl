function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:pvar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    ip = inv(p)
    pvar_eta, pvar_t, pvar_w, pvar_v = model[Symbol(:pvar_eta_, i)], model[Symbol(:pvar_t_, i)], model[Symbol(:pvar_w_, i)], model[Symbol(:pvar_v_, i)] = @variables(model,
                                                                                                                                                                     begin
                                                                                                                                                                         ()
                                                                                                                                                                         ()
                                                                                                                                                                         [1:T],
                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                         [1:T]
                                                                                                                                                                     end)

    wi = nothing_scalar_array_selector(r.w, pr.w)
    pvar_risk = model[key] = if isnothing(wi)
        iaT = inv(alpha * T^ip)
        @constraint(model, sc * (sum(pvar_v) - pvar_t) == 0)
    else
        iaT = inv(alpha * sum(wi)^ip)
        @constraint(model, sc * (dot(wi, pvar_v) - pvar_t) == 0)
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
    set_risk_bounds_and_expression!(model, opt, pvar_risk, r.settings, key)
    return pvar_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...) end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...) end
