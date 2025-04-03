function _w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function _w_neg_flag(wb::AbstractVector)
    return any(wb .< zero(eltype(wb)))
end
function _w_finite_flag(wb::Real)
    return isfinite(wb)
end
function _w_finite_flag(wb::AbstractVector)
    return any(isfinite.(wb))
end
function set_weight_constraints!(::JuMP.Model, ::Nothing, ::Nothing, ::Bool)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Union{<:Real, <:BudgetConstraint},
                                 sbgt::Union{Nothing, <:Real, <:BudgetConstraint},
                                 long_only::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = _w_neg_flag(lb) || _w_neg_flag(ub)
    @smart_assert(long_only ⊼ flag, "Long-only strategy cannot have negative weight limits")
    w, k, sc = get_w_k_sc(model)
    if !isnothing(lb) && _w_finite_flag(lb)
        @constraint(model, w_lb, sc * w .>= sc * k * lb)
    end
    if !isnothing(ub) && _w_finite_flag(ub)
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
        set_budget_constraints!(model, bgt, sbgt)
    else
        @expression(model, lw, w)
    end
    return nothing
end

export WeightBounds
