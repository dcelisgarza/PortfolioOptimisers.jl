function set_risk_constraints!(model::JuMP.Model, ::Any, r::UlcerIndex,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :uci)
        return model[:uci_risk]
    end
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, cuci_soc, [sc * uci; sc * view(dd, 2:(T + 1))] in SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, uci_risk, r.settings, :uci_risk)
    return uci_risk
end
