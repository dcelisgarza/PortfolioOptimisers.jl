struct LongShortBounds{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       AbstractConstraintModel
    s_lb::T1
    l_ub::T2
end
function LongShortBounds(; s_lb::Union{Nothing, <:Real} = 1.0,
                         l_ub::Union{Nothing, <:Real} = 1.0)
    lb_flag = isnothing(s_lb)
    ub_flag = isnothing(l_ub)
    @smart_assert(lb_flag ⊼ ub_flag)
    if !lb_flag
        @smart_assert(isfinite(s_lb) && s_lb >= zero(s_lb))
    end
    if !ub_flag
        @smart_assert(isfinite(l_ub) && l_ub >= zero(l_ub))
    end
    return LongShortBounds{typeof(s_lb), typeof(l_ub)}(s_lb, l_ub)
end
function set_long_short_bounds_constraints!(::JuMP.Model, ::Nothing, ::Bool)
    return nothing
end
function set_long_short_bounds_constraints!(model::JuMP.Model, lsb::LongShortBounds,
                                            long_only::Bool = false)
    if long_only && isnothing(lsb.l_ub)
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    if !isnothing(lsb.l_ub)
        @variable(model, lw[1:N] >= 0)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         lw_ub, sc * sum(lw) <= sc * k * lsb.l_ub
                     end)
    end
    if !long_only && !isnothing(lsb.s_lb)
        @variable(model, sw[1:N] >= 0)
        @constraints(model, begin
                         w_sw, sc * w >= -sc * sw
                         sw_lb, sc * sum(sw) <= sc * k * lsb.s_lb
                     end)
    end
    return nothing
end
