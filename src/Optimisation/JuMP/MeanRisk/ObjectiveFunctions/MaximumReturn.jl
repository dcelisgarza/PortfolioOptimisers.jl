struct MaximumReturn <: ObjectiveFunction end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumReturn,
                                           pret::PortfolioReturnType,
                                           cobj::Union{Nothing, <:CustomObjective},
                                           mr::JuMPOptimisationType,
                                           pm::AbstractPriorResult)
    so = model[:so]
    ret = model[:ret]
    op = model[:op]
    @expression(model, obj_expr, ret)
    add_to_expression!(obj_expr, -1, op)
    add_custom_objective_term!(obj, pret, cobj, obj_expr, mr, pm)
    @objective(model, Max, so * obj_expr)
    return nothing
end

export MaximumReturn
