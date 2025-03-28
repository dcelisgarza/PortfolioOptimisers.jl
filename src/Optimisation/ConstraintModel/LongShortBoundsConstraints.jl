struct LongShortSum{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       AbstractConstraintModel
    ss::T1
    ls::T2
end
function LongShortSum(; ss::Union{Nothing, <:Real} = 1.0, ls::Union{Nothing, <:Real} = 1.0)
    lb_flag = isnothing(ss)
    ub_flag = isnothing(ls)
    @smart_assert(lb_flag ⊼ ub_flag)
    if !lb_flag
        @smart_assert(isfinite(ss) && ss >= zero(ss))
    end
    if !ub_flag
        @smart_assert(isfinite(ls) && ls >= zero(ls))
    end
    return LongShortSum{typeof(ss), typeof(ls)}(ss, ls)
end
function set_long_short_bounds_constraints!(::JuMP.Model, ::Nothing, ::Bool)
    return nothing
end
function set_long_short_bounds_constraints!(model::JuMP.Model, lsb::LongShortSum,
                                            long_only::Bool = false)
    if long_only && isnothing(lsb.ls)
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    if !isnothing(lsb.ls)
        @variable(model, lw[1:N] >= 0)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         lw_ub, sc * sum(lw) <= sc * k * lsb.ls
                     end)
    end
    if !long_only && !isnothing(lsb.ss)
        @variable(model, sw[1:N] >= 0)
        @constraints(model, begin
                         w_sw, sc * w >= -sc * sw
                         sw_lb, sc * sum(sw) <= sc * k * lsb.ss
                     end)
    end
    return nothing
end
