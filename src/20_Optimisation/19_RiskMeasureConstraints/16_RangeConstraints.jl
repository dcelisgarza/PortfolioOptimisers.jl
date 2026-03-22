function set_risk_constraints!(model::JuMP.Model, ::Any, r::Range,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :range_risk)
        return model[:range_risk]
    end
    sc = model[:sc]
    wr_risk = set_wr_risk_expression!(model, pr.X)
    net_X = model[:net_X]
    JuMP.@variable(model, br_risk)
    JuMP.@expression(model, range_risk, wr_risk - br_risk)
    JuMP.@constraint(model, cbr, sc * (br_risk .+ net_X) <= 0)
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, :range_risk)
    return range_risk
end
