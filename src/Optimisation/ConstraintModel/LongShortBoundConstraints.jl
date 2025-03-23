struct LongShortBounds{T1 <: Real, T2 <: Real} <: ConstraintModel
    s_lb::T1
    l_ub::T2
end
function LongShortBounds(; s_lb::Real = 0.0, l_ub::Real = 1.0)
    @smart_assert(s_lb >= zero(eltype(s_lb)))
    @smart_assert(l_ub >= zero(eltype(l_ub)))
    return LongShortBounds{typeof(s_lb), typeof(l_ub)}(s_lb, l_ub)
end
function set_long_short_bounds_constraints!(model::JuMP.Model,
                                            slb::LongShortBounds{<:Real, <:Real},
                                            long_only::Bool = false)
    if long_only
        return nothing
    end
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    @variables(model, begin
                   lw[1:N] >= w
                   sw[1:N] >= -w
               end)
    model[:sw_lb] = @constraint(model, sc * sum(sw) <= sc * k * slb.s_lb)
    model[:lw_ub] = @constraint(model, sc * sum(lw) <= sc * k * slb.l_ub)
    return nothing
end
