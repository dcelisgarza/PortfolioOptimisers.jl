function set_risk_constraints!(model::JuMP.Model, ::Any, r::MaximumDrawdown,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :mdd_risk)
        return model[:mdd_risk]
    end
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    @variable(model, mdd_risk)
    @constraint(model, cmdd_risk, sc * (mdd_risk .- view(dd, 2:(T + 1))) >= 0)
    set_risk_bounds_and_expression!(model, opt, mdd_risk, r.settings, :mdd_risk)
    return mdd_risk
end
