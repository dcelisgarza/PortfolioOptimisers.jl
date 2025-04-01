struct MaximumUtility{T1 <: Real} <: ObjectiveFunction
    l::T1
end
function Utility(; l::Real = 2.0)
    @smart_assert(l >= zero(l))
    return Utility{typeof(l)}(l)
end
function set_portfolio_objective_function!(port, obj_func::MaximumUtility,
                                           ret_type::PortfolioReturnType,
                                           custom_obj::CustomObjective)
    model = port.model
    so = model[:so]
    ret = model[:ret]
    risk = model[:risk]
    l = obj_func.l
    op = model[:op]
    @expression(model, obj_expr, ret - l * risk)
    add_to_expression!(obj_expr, -1, op)
    add_custom_objective_term!(port, obj_func, ret_type, custom_obj, obj_expr)
    @objective(model, Max, so * obj_expr)
    return nothing
end

export Utility
