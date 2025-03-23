struct MinimumRisk <: ObjectiveFunction end
function set_portfolio_objective_function!(port, obj_func::MinimumRisk,
                                           ret_type::PortfolioReturnType,
                                           custom_obj::CustomObjective)
    model = port.model
    scale_obj = model[:scale_obj]
    risk = model[:risk]
    @expression(model, obj_expr, risk)
    add_objective_function_penalty!(model, obj_expr, 1)
    add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Min, scale_obj * obj_expr)
    return nothing
end

export MinimumRisk
