function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::NormOneConeBrownianDistanceVariance,
                                                     Dt::NumMat, Dx::NumMat)
    T = size(Dt, 1)
    sc = model[:sc]
    @constraint(model, cbdvariance_noc[j = 1:T, i = j:T],
                [sc * Dt[i, j]; sc * Dx[i, j]] in MOI.NormOneCone(2))
    return nothing
end
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::IneqBrownianDistanceVariance,
                                                     Dt::NumMat, Dx::NumMat)
    sc = model[:sc]
    @constraints(model, begin
                     cp_bdvariance, sc * (Dt - Dx) in Nonnegatives()
                     cn_bdvariance, sc * (Dt + Dx) in Nonnegatives()
                 end)
    return nothing
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::QuadRiskExpr,
                                                Dt::NumMat, iT2::Number)
    @expression(model, bdvariance_risk, iT2 * (dot(Dt, Dt) + iT2 * sum(Dt)^2))
    return bdvariance_risk
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::RSOCRiskExpr,
                                                Dt::NumMat, iT2::Number)
    sc = model[:sc]
    @variable(model, tDt)
    @constraint(model, rsoc_Dt, [sc * tDt;
                                 0.5;
                                 sc * vec(Dt)] in RotatedSecondOrderCone())
    @expression(model, bdvariance_risk, iT2 * (tDt + iT2 * sum(Dt)^2))
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
    @variable(model, Dt[1:T, 1:T], Symmetric)
    @expression(model, Dx, net_X * transpose(ovec) - ovec * transpose(net_X))
    bdvariance_risk = set_brownian_distance_risk_constraint!(model, r.alg, Dt, iT2)
    set_brownian_distance_variance_constraints!(model, r.algc, Dt, Dx)
    set_risk_bounds_and_expression!(model, opt, bdvariance_risk, r.settings,
                                    :bdvariance_risk)
    return bdvariance_risk
end
