struct WeightLimits{T1 <: Union{<:Real, <:AbstractVector},
                    T2 <: Union{<:Real, <:AbstractVector}}
    lb::T1
    ub::T2
end
function WeightLimits(; lb::Union{<:Real, <:AbstractVector} = 0.0,
                      ub::Union{<:Real, <:AbstractVector} = 1.0)
    lb_flag = isa(lb, AbstractVector)
    ub_flag = isa(ub, AbstractVector)
    if lb_flag
        @smart_assert(!isempty(lb))
    end
    if ub_flag
        @smart_assert(!isempty(ub))
    end
    if lb_flag && ub_flag
        @smart_assert(length(lb) == length(ub))
        @smart_assert(all(iszero.(lb)) && all(iszero.(ub)))
    end
    @smart_assert(all(lb .<= ub))
    return WeightLimits{typeof(lb), typeof(ub)}(lb, ub)
end
function _w_limit_flag(wl::Real)
    return isinf(wl)
end
function _w_limit_flag(wl::AbstractVector)
    return all(isinf.(wl))
end
function _w_neg_flag(wl::Real)
    return wl < zero(wl)
end
function _w_neg_flag(wl::AbstractVector)
    return any(wl .< zero(eltype(wl)))
end
function set_weight_constraints!(model::JuMP.Model, wl::WeightLimits,
                                 long_only::Bool = false)
    lb_flag = _w_limit_flag(wl.lb)
    ub_flag = _w_limit_flag(wl.ub)
    if lb_flag && ub_flag
        return nothing
    end
    @smart_assert(long_only ⊼ _w_neg_flag(wl.lb),
                  "Long-only strategy cannot have negative weight limits")
    w, k, sc = get_w_k_sc(model)
    if !lb_flag
        @constraint(model, w_lb, sc * w >= sc * k * wl.lb)
    end
    if !ub_flag
        @constraint(model, w_ub, sc * w <= sc * k * wl.ub)
    end
    return nothing
end

export WeightLimits
