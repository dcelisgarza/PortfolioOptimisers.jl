abstract type JuMPBudgetConstraint <: ConstraintEstimator end
struct BudgetRange{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       JuMPBudgetConstraint
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
    if !lb_flag && !ub_flag
        @smart_assert(lb <= ub)
    end
    return BudgetRange{typeof(lb), typeof(ub)}(lb, ub)
end
function set_budget_constraints!(args...; kwargs...)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, val::Real, w::AbstractVector;
                                 key::Symbol = :bgt, kwargs...)
    k = model[:k]
    sc = model[:sc]
    model[key] = @constraint(model, sc * sum(w) == sc * k * val)
    return nothing
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
function set_long_short_budget_constraints!(::JuMP.Model, ::Nothing, ::Nothing)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lbgt, sc * sum(lw) == sc * k * bgt)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, bgt::Real)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, sbgt, sc * sum(sw) == sc * k * sbgt)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::Real)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lsbgt, sc * sum(lw) == sc * k * (bgt + sbgt))
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * sum(lw) >= sc * k * lb)
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * sum(lw) <= sc * k * ub)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::BudgetRange)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraint(model, sbgt_lb, sc * sum(sw) >= sc * k * lb)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraint(model, sbgt_ub, sc * sum(sw) <= sc * k * ub)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, sbgt::Real)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * sum(lw) >= sc * k * (lb + sbgt))
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * sum(lw) <= sc * k * (ub + sbgt))
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * sum(lw) >= sc * k * (lb + bgt))
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * sum(lw) <= sc * k * (ub + bgt))
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    slb = sbgt.lb
    lb_flag = isnothing(lb)
    slb_flag = isnothing(slb)
    if !lb_flag && !slb_flag
        @constraint(model, lbgt_lb, sc * sum(lw) >= sc * k * (lb + slb))
    elseif !lb_flag && slb_flag
        @constraint(model, lbgt_lb, sc * sum(lw) >= sc * k * lb)
    end
    ub = bgt.ub
    sub = sbgt.ub
    ub_flag = isnothing(ub)
    sub_flag = isnothing(sub)
    if !ub_flag && !sub_flag
        @constraint(model, lbgt_ub, sc * sum(lw) <= sc * k * (ub + sub))
    elseif !ub_flag && sub_flag
        @constraint(model, lbgt_ub, sc * sum(lw) <= sc * k * ub)
    end
    return nothing
end
function w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function w_neg_flag(wb::AbstractVector)
    return any(wb .< zero(eltype(wb)))
end
function w_finite_flag(wb::Real)
    return isfinite(wb)
end
function w_finite_flag(wb::AbstractVector)
    return any(isfinite, wb)
end
function set_weight_constraints!(args...)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBoundsResult,
                                 bgt::Union{Nothing, <:Real, <:JuMPBudgetConstraint},
                                 sbgt::Union{Nothing, <:Real, <:JuMPBudgetConstraint},
                                 long_only::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @smart_assert(long_only ⊼ flag, "Long-only strategy cannot have negative weight limits")
    w, k, sc = get_w_k_sc(model)
    if !isnothing(lb) && w_finite_flag(lb)
        @constraint(model, w_lb, sc * w .>= sc * k * lb)
    end
    if !isnothing(ub) && w_finite_flag(ub)
        @constraint(model, w_ub, sc * w .<= sc * k * ub)
    end
    set_budget_constraints!(model, bgt, w; key = :bgt, key_lb = :bgt_lb, key_ub = :bgt_ub)
    if flag
        @variables(model, begin
                       lw[1:N] >= 0
                       sw[1:N] >= 0
                   end)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         w_sw, sc * w >= -sc * sw
                     end)
        set_budget_constraints!(model, sbgt, sw; key = :sbgt, key_lb = :sbgt_lb,
                                key_ub = :sbgt_ub)
        set_long_short_budget_constraints!(model, bgt, sbgt)
    else
        @expression(model, lw, w)
    end
    return nothing
end