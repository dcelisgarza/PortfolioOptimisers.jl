abstract type BudgetConstraints <: ConstraintModel end
struct Budget{T1 <: Real} <: BudgetConstraints
    val::T1
end
function Budget(; val::Real = 1.0)
    @smart_assert(isfinite(val))
    return Budget{typeof(val)}(val)
end
function set_budget_constraints!(model::JuMP.Model, b::Budget)
    w, k, sc = get_w_k_sc(model)
    @constraint(model, wb, sc * sum(w) == sc * k * b.val)
    return nothing
end
struct BudgetRange{T1 <: Real, T2 <: Real} <: BudgetConstraints
    lb::T1
    ub::T2
end
function BudgetRange(; lb::Real = 1.0, ub::Real = 1.0)
    @smart_assert(isinf(lb) ⊼ isinf(ub))
    return BudgetRange{typeof(lb), typeof(ub)}(lb, ub)
end
function set_budget_constraints!(model::JuMP.Model, b::BudgetRange)
    w, k, sc = get_w_k_sc(model)
    if isfinite(b.lb)
        @constraint(model, wb_lb, sc * sum(w) >= sc * k * b.lb)
    end
    if isfinite(b.ub)
        @constraint(model, wb_ub, sc * sum(w) <= sc * k * b.ub)
    end
    return nothing
end
