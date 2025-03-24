struct LongShortBounds{T1 <: Real, T2 <: Real} <: ConstraintModel
    s_lb::T1
    l_ub::T2
end
function LongShortBounds(; s_lb::Real = 1.0, l_ub::Real = 1.0)
    @smart_assert(s_lb >= zero(eltype(s_lb)))
    @smart_assert(l_ub >= zero(eltype(l_ub)))
    return LongShortBounds{typeof(s_lb), typeof(l_ub)}(s_lb, l_ub)
end
function set_long_short_bounds_constraints!(model::JuMP.Model, lsb::LongShortBounds,
                                            long_only::Bool = false)
    ub_flag = isinf(lsb.l_ub)
    lb_flag = isinf(lsb.s_lb)
    if long_only || (ub_flag && lb_flag)
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    if !ub_flag
        @variable(model, lw[1:N] >= 0)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         lw_ub, sc * sum(lw) <= sc * k * lsb.l_ub
                     end)
    end
    if !lb_flag
        @variable(model, sw[1:N] >= 0)
        @constraints(model, begin
                         w_sw, sc * w >= -sc * sw
                         sw_lb, sc * sum(sw) <= sc * k * lsb.s_lb
                     end)
    end
    return nothing
end
