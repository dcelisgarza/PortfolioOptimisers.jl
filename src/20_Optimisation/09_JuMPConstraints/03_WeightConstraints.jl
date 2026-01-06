function set_weight_constraints!(args...)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Option{<:Num_BgtRg}, sbgt::Option{<:Num_BgtRg},
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
        JuMP.@constraint(model, w_lb, sc * (w - k * lb) >= 0)
    end
    if !isnothing(ub) && w_finite_flag(ub)
        JuMP.@constraint(model, w_ub, sc * (w - k * ub) <= 0)
    end
    set_budget_constraints!(model, bgt, w)
    if flag
        JuMP.@variables(model, begin
                            lw[1:N] >= 0
                            sw[1:N] >= 0
                        end)
        JuMP.@constraints(model, begin
                              w_lw, sc * (w - lw) <= 0
                              w_sw, sc * (w + sw) >= 0
                          end)
        set_long_short_budget_constraints!(model, bgt, sbgt)
    else
        JuMP.@expression(model, lw, w)
    end
    return nothing
end
function non_zero_real_or_vec(::Nothing)
    return false
end
function non_zero_real_or_vec(x::Number)
    return !iszero(x)
end
function non_zero_real_or_vec(x::VecNum)
    return any(!iszero, x)
end
function set_linear_weight_constraints!(args...)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcms::Lc_VecLc, key_ineq::Symbol,
                                        key_eq::Symbol)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    for (i, lcm) in enumerate(lcms)
        if !isnothing(lcm.ineq)
            A = lcm.ineq.A
            B = lcm.ineq.B
            model[Symbol(key_ineq, i)] = JuMP.@constraint(model, sc * (A * w - k * B) <= 0)
        end
        if !isnothing(lcm.eq)
            A = lcm.eq.A
            B = lcm.eq.B
            model[Symbol(key_eq, i)] = JuMP.@constraint(model, sc * (A * w - k * B) == 0)
        end
    end
    return nothing
end
