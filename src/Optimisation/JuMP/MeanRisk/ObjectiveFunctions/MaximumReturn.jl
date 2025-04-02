struct MaximumReturn <: ObjectiveFunction end
function set_portfolio_objective_function!(model::JuMP.Model, obj_func::MaximumReturn,
                                           ret_type::PortfolioReturnType,
                                           custom_obj::Union{Nothing, <:CustomObjective})
    so = model[:so]
    ret = model[:ret]
    op = model[:op]
    @expression(model, obj_expr, ret)
    add_to_expression!(obj_expr, -1, op)
    add_custom_objective_term!(obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Max, so * obj_expr)
    return nothing
end

export MaximumReturn
