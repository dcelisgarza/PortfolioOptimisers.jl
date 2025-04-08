struct MaximumRatio{T1 <: Real, T2 <: Real} <: ObjectiveFunction
    rf::T1
    ohf::T2
end
function MaximumRatio(; rf::Real = 0.0, ohf::Real = 0.0)
    @smart_assert(rf >= zero(rf))
    @smart_assert(ohf >= zero(ohf))
    return MaximumRatio{typeof(rf), typeof(ohf)}(rf, ohf)
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::ExactKellyReturn,
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
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::PortfolioReturnType,
                                           cobj::Union{Nothing, <:CustomObjective},
                                           mr::JuMPOptimisationType,
                                           pm::AbstractPriorResult)
    so = model[:so]
    op = model[:op]
    if haskey(model, :sr_risk)
        ret = model[:ret]
        @expression(model, obj_expr, ret)
        add_to_expression!(obj_expr, -1, op)
        add_custom_objective_term!(obj, pret, cobj, obj_expr, mr, pm)
        @objective(model, Max, so * obj_expr)
    else
        risk = model[:risk]
        @expression(model, obj_expr, risk)
        add_to_expression!(obj_expr, 1, op)
        add_custom_objective_term!(obj, pret, cobj, obj_expr, mr, pm)
        @objective(model, Min, so * obj_expr)
    end
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, mu::AbstractVector,
                                             obj::MaximumRatio)
    ohf = if iszero(obj.ohf)
        min(1e3, max(1e-3, mean(abs.(mu))))
    else
        obj.ohf
    end
    @expression(model, ohf, ohf)
    @variable(model, k >= 0)
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, args...)
    @expression(model, k, 1)
    return nothing
end

export MaximumRatio
