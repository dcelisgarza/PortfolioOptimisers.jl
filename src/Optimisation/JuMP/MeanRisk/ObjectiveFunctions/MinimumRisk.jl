struct MinimumRisk <: ObjectiveFunction end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MinimumRisk,
                                           pret::PortfolioReturnType,
                                           co::Union{Nothing, <:CustomObjective},
                                           mr::JuMPOptimisationType, pm::AbstractPriorModel)
    so = model[:so]
    op = model[:op]
    risk = model[:risk]
    @expression(model, obj_expr, risk)
    add_to_expression!(obj_expr, 1, op)
    add_custom_objective_term!(obj, pret, co, obj_expr, mr, pm)
    @objective(model, Min, so * obj_expr)
    return nothing
end

export MinimumRisk
