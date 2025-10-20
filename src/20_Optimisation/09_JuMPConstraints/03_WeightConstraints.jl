function w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function w_neg_flag(wb::AbstractVector)
    return any(x -> x < zero(x), wb)
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
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Union{Nothing, <:Real, <:BudgetRange},
                                 sbgt::Union{Nothing, <:Real, <:BudgetRange},
                                 long::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @argcheck(!(long && flag), "Long-only strategy cannot have negative weight limits")
    w = model[:w]
    N = length(w)
    k = model[:k]
    sc = model[:sc]
    if !isnothing(lb) && w_finite_flag(lb)
        @constraint(model, w_lb, sc * (w - k * lb) >= 0)
    end
    if !isnothing(ub) && w_finite_flag(ub)
        @constraint(model, w_ub, sc * (w - k * ub) <= 0)
    end
    set_budget_constraints!(model, bgt, w)
    if flag
        @variables(model, begin
                       lw[1:N] >= 0
                       sw[1:N] >= 0
                   end)
        @constraints(model, begin
                         w_lw, sc * (w - lw) <= 0
                         w_sw, sc * (w + sw) >= 0
                     end)
        set_long_short_budget_constraints!(model, bgt, sbgt)
    else
        @expression(model, lw, w)
    end
    return nothing
end
function non_zero_real_or_vec(::Nothing)
    return false
end
function non_zero_real_or_vec(x::Real)
    return !iszero(x)
end
function non_zero_real_or_vec(x::AbstractVector{<:Real})
    return any(!iszero, x)
end
function set_linear_weight_constraints!(args...)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model,
                                        lcms::Union{<:LinearConstraint,
                                                    <:AbstractVector{<:LinearConstraint}},
                                        key_ineq::Symbol, key_eq::Symbol)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    for (i, lcm) in enumerate(lcms)
        if !isnothing(lcm.ineq)
            A = lcm.ineq.A
            B = lcm.ineq.B
            model[Symbol(key_ineq, i)] = @constraint(model, sc * (A * w - k * B) <= 0)
        end
        if !isnothing(lcm.eq)
            A = lcm.eq.A
            B = lcm.eq.B
            model[Symbol(key_eq, i)] = @constraint(model, sc * (A * w - k * B) == 0)
        end
    end
    return nothing
end
