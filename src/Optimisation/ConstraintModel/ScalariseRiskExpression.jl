function set_scalar_risk_expression!(model::JuMP.Model, ::SumScalariser)
    risk_vec = model[:risk_vec]
    if any(isa.(risk_vec, QuadExpr))
        @expression(model, risk, zero(QuadExpr))
    else
        @expression(model, risk, zero(AffExpr))
    end
    for r_risk ∈ risk_vec
        add_to_expression!(risk, r_risk)
    end
    return nothing
end
function set_scalar_risk_expression!(model::JuMP.Model, scalariser::LogSumExpScalariser)
    sc = model[:sc]
    risk_vec = model[:risk_vec]
    N = length(risk_vec)
    gamma = scalariser.gamma
    @variables(model, begin
                   risk
                   u_risk[1:N]
               end)
    @constraints(model,
                 begin
                     cs_risk_lse_u, sc * sum(u_risk) <= sc * 1
                     cs_risk_lse[i = 1:N],
                     [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
                     MOI.ExponentialCone()
                 end)
    return nothing
end
function set_scalar_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    risk_vec = model[:risk_vec]
    @variable(model, risk)
    @constraint(model, csrm, risk >= risk_vec)
    return nothing
end
