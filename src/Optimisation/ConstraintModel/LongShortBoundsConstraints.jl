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
    if long_only
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variables(model, begin
                   lw[1:N] >= 0
                   sw[1:N] >= 0
               end)
    @constraints(model, begin
                     sc * w <= sc * lw
                     sc * w >= -sc * sw
                     lw_ub, sc * sum(lw) <= sc * k * lsb.l_ub
                     sw_lb, sc * sum(sw) <= sc * k * lsb.s_lb
                 end)
    return nothing
end
