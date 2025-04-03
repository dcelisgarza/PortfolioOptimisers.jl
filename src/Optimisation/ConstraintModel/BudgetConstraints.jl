abstract type BudgetConstraint <: AbstractConstraintModel end
function set_budget_constraints!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, val::Real)
    w, k, sc = get_w_k_sc(model)
    @constraint(model, wb, sc * sum(w) == sc * k * val)
    return nothing
end
struct BudgetRange{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       BudgetConstraint
    lb::T1
    ub::T2
end
function BudgetRange(; lb::Union{Nothing, <:Real} = 1.0, ub::Union{Nothing, <:Real} = 1.0)
    lb_flag = isnothing(lb)
    ub_flag = isnothing(ub)
    @smart_assert(lb_flag ⊼ ub_flag)
    if !lb_flag
        @smart_assert(isfinite(lb))
    end
    if !ub_flag
        @smart_assert(isfinite(ub))
    end
    if !lb_flag ⊼ !ub_flag
        @smart_assert(lb <= ub)
    end
    return BudgetRange{typeof(lb), typeof(ub)}(lb, ub)
end
function set_budget_constraints!(model::JuMP.Model, b::BudgetRange)
    w, k, sc = get_w_k_sc(model)
    if !isnothing(b.lb)
        lb = b.lb
        @constraint(model, wb_lb, sc * sum(w) >= sc * k * lb)
    end
    if !isnothing(b.ub)
        ub = b.ub
        @constraint(model, wb_ub, sc * sum(w) <= sc * k * ub)
    end
    return nothing
end
