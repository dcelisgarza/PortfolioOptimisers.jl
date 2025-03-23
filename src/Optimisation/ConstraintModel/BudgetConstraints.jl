abstract type BudgetConstraints <: ConstraintModel end
struct Budget{T1 <: Real} <: BudgetConstraints
    val::T1
end
function Budget(; val::Real = 1.0)
    return Budget{typeof(val)}(val)
end
function set_budget_constraints!(model::JuMP.Model, b::Budget)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    model[:w_budget] = @constraint(model, sc * sum(w) == sc * k * b.val)
    return nothing
end
struct BudgetRange{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       BudgetConstraints
    lb::T1
    ub::T2
end
function BudgetRange(; lb::Union{Nothing, <:Real} = 1.0, ub::Union{Nothing, <:Real} = 1.0)
    return BudgetRange{typeof(lb), typeof(ub)}(lb, ub)
end
function set_budget_constraints!(model::JuMP.Model, b::BudgetRange{<:Real, <:Real})
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    model[:w_budget_lb] = @constraint(model, sc * sum(w) >= sc * k * b.lb)
    model[:w_budget_ub] = @constraint(model, sc * sum(w) <= sc * k * b.ub)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, b::BudgetRange{Nothing, <:Real})
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    model[:w_budget_ub] = @constraint(model, sc * sum(w) <= sc * k * b.ub)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, b::BudgetRange{<:Real, Nothing})
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    model[:w_budget_lb] = @constraint(model, sc * sum(w) >= sc * k * b.lb)
    return nothing
end
