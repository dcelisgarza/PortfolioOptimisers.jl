function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::NormOneConeBrownianDistanceVariance,
                                                     Dt::MatNum, Dx::MatNum)
    T = size(Dt, 1)
    sc = model[:sc]
    JuMP.@constraint(model, cbdvariance_noc[j = 1:T, i = j:T],
                     [sc * Dt[i, j]; sc * Dx[i, j]] in JuMP.MOI.NormOneCone(2))
    return nothing
end
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::IneqBrownianDistanceVariance,
                                                     Dt::MatNum, Dx::MatNum)
    sc = model[:sc]
    JuMP.@constraints(model, begin
                          cp_bdvariance, sc * (Dt - Dx) in JuMP.Nonnegatives()
                          cn_bdvariance, sc * (Dt + Dx) in JuMP.Nonnegatives()
                      end)
    return nothing
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::QuadRiskExpr,
                                                Dt::MatNum, iT2::Number)
    JuMP.@expression(model, bdvariance_risk,
                     iT2 * (LinearAlgebra.dot(Dt, Dt) + iT2 * sum(Dt)^2))
    return bdvariance_risk
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::RSOCRiskExpr,
                                                Dt::MatNum, iT2::Number)
    sc = model[:sc]
    JuMP.@variable(model, tDt)
    JuMP.@constraint(model, rsoc_Dt,
                     [sc * tDt;
                      0.5;
                      sc * vec(Dt)] in JuMP.RotatedSecondOrderCone())
    JuMP.@expression(model, bdvariance_risk, iT2 * (tDt + iT2 * sum(Dt)^2))
    return bdvariance_risk
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :bdvariance_risk)
        return model[:bdvariance_risk]
    end
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    iT2 = inv(T^2)
    ovec = range(one(eltype(pr.X)), one(eltype(pr.X)); length = T)
    JuMP.@variable(model, Dt[1:T, 1:T], Symmetric)
    JuMP.@expression(model, Dx, net_X * transpose(ovec) - ovec * transpose(net_X))
    bdvariance_risk = set_brownian_distance_risk_constraint!(model, r.alg, Dt, iT2)
    set_brownian_distance_variance_constraints!(model, r.algc, Dt, Dx)
    set_risk_bounds_and_expression!(model, opt, bdvariance_risk, r.settings,
                                    :bdvariance_risk)
    return bdvariance_risk
end
