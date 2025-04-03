struct MaximumUtility{T1 <: Real} <: ObjectiveFunction
    l::T1
end
function MaximumUtility(; l::Real = 2.0)
    @smart_assert(l >= zero(l))
    return MaximumUtility{typeof(l)}(l)
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumUtility,
                                           pret::PortfolioReturnType,
                                           cobj::Union{Nothing, <:CustomObjective},
                                           mr::JuMPOptimisationType, pm::AbstractPriorModel)
    so = model[:so]
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    op = model[:op]
    @expression(model, obj_expr, ret - l * risk)
    add_to_expression!(obj_expr, -1, op)
    add_custom_objective_term!(obj, pret, cobj, obj_expr, mr, pm)
    @objective(model, Max, so * obj_expr)
    return nothing
end

export MaximumUtility
