struct MaximumUtility{T1 <: Real} <: ObjectiveFunction
    l::T1
end
function Utility(; l::Real = 2.0)
    @smart_assert(l >= zero(l))
    return Utility{typeof(l)}(l)
end
function portfolio_objective_function(port, obj_func::MaximumUtility,
                                      ret_type::OptimisationReturnsType,
                                      custom_obj::CustomObjective)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @expression(model, obj_expr, ret - l * risk)
    add_objective_penalty(model, obj_expr, -1)
    custom_objective(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Max, scale_obj * obj_expr)
    return nothing
end

export Utility
