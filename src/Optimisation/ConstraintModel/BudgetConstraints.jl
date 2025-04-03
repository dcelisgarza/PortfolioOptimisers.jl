abstract type BudgetConstraint <: AbstractConstraintModel end
function set_budget_constraints!(::JuMP.Model, ::Nothing, args...)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, val::Real, w::AbstractVector;
                                 key::Symbol = :bgt, kwargs...)
    k = model[:k]
    sc = model[:sc]
    model[key] = @constraint(model, sc * sum(w) == sc * k * val)
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
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::AbstractVector;
                                 key_lb::Symbol = :bgt_lb, key_ub::Symbol = :bgt_ub,
                                 kwargs...)
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    ub = bgt.ub
    if !isnothing(lb)
        model[key_lb] = @constraint(model, sc * sum(w) >= sc * k * lb)
    end
    if !isnothing(ub)
        model[key_ub] = @constraint(model, sc * sum(w) <= sc * k * ub)
    end
    return nothing
end
function set_budget_constraints!(::JuMP.Model, ::Nothing, ::Any)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::Real, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lbgt, sc * sum(lw) == sc * k * bgt)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::Real)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lbgt, sc * sum(lw) == sc * k * (bgt + sbgt))
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraint(model, lbdgt_lb, sc * sum(lw) >= sc * k * (lb + bgt))
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraint(model, lbdgt_ub, sc * sum(lw) <= sc * k * (ub + bgt))
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbdgt_lb, sc * sum(lw) >= sc * k * lb)
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbdgt_ub, sc * sum(lw) <= sc * k * ub)
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, sbgt::Real)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbdgt_lb, sc * sum(lw) >= sc * k * (lb + sbgt))
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbdgt_ub, sc * sum(lw) <= sc * k * (ub + sbgt))
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    slb = sbgt.lb
    lb_flag = isnothing(lb)
    slb_flag = isnothing(slb)
    if !lb_flag && !slb_flag
        @constraint(model, lbdgt_lb, sc * sum(lw) >= sc * k * (lb + sbgt))
    elseif !lb_flag && slb_flag
        @constraint(model, lbdgt_lb, sc * sum(lw) >= sc * k * lb)
    end
    ub = bgt.ub
    sub = sbgt.ub
    ub_flag = isnothing(ub)
    sub_flag = isnothing(sub)
    if !ub_flag && !sub_flag
        @constraint(model, lbdgt_ub, sc * sum(lw) <= sc * k * (ub + sbgt))
    elseif !ub_flag && sub_flag
        @constraint(model, lbdgt_ub, sc * sum(lw) <= sc * k * ub)
    end
    return nothing
end
