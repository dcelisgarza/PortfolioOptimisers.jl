struct MinimumRisk <: ObjectiveFunction end
function portfolio_objective_function(port, obj_func::MinimumRisk,
                                      ret_type::OptimisationReturnsType,
                                      custom_obj::CustomObjective)
    model = port.model
    scale_obj = model[:scale_obj]
    risk = model[:risk]
    @expression(model, obj_expr, risk)
    add_objective_function_penalty(model, obj_expr, 1)
    custom_objective(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Min, scale_obj * obj_expr)
    return nothing
end
