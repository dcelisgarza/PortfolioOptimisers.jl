struct MinimumRisk <: ObjectiveFunction end
function set_portfolio_objective_function!(port, obj_func::MinimumRisk,
                                           ret_type::PortfolioReturnType,
                                           custom_obj::CustomObjective)
    model = port.model
    so = model[:so]
    obj_pen = model[:obj_pen]
    risk = model[:risk]
    @expression(model, obj_expr, risk)
    add_to_expression!(obj_expr, 1, obj_pen)
    add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Min, so * obj_expr)
    return nothing
end

export MinimumRisk
