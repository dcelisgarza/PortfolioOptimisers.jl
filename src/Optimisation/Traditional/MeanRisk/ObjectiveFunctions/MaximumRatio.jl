struct MaximumRatio{T1 <: Real, T2 <: Real} <: ObjectiveFunction
    rf::T1
    ohf::T2
end
function MaximumRatio(; rf::Real = 0.0, ohf::Real = 0.0)
    @smart_assert(rf >= zero(rf))
    @smart_assert(ohf >= zero(ohf))
    return MaximumRatio{typeof(rf), typeof(ohf)}(rf, ohf)
end
function set_portfolio_objective_function!(port, obj_func::MaximumRatio,
                                           ret_type::ExactKellyReturn,
                                           custom_obj::CustomObjective)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    @expression(model, obj_expr, ret)
    add_objective_function_penalty!(model, obj_expr, -1)
    add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Max, scale_obj * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(port, obj_func::MaximumRatio,
                                           ret_type::ArithmeticReturn,
                                           custom_obj::CustomObjective)
    model = port.model
    scale_obj = model[:scale_obj]
    if haskey(model, :constr_risk_ratio)
        ret = model[:ret]
        @expression(model, obj_expr, ret)
        add_objective_function_penalty!(model, obj_expr, -1)
        add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
        @objective(model, Max, scale_obj * obj_expr)
    else
        risk = model[:risk]
        @expression(model, obj_expr, risk)
        add_objective_function_penalty!(model, obj_expr, 1)
        add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
        @objective(model, Min, scale_obj * obj_expr)
    end
    return nothing
end

export MaximumRatio
