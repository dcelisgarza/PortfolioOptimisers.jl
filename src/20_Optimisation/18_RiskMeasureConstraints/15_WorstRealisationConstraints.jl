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
function set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :wr_risk)
        return model[:wr_risk]
    end
    wr_risk = set_wr_risk_expression!(model, pr.X)
    set_risk_bounds_and_expression!(model, opt, wr_risk, r.settings, :wr_risk)
    return wr_risk
end
