struct WeightLimits{T1 <: Union{<:Real, <:AbstractVector},
                    T2 <: Union{<:Real, <:AbstractVector}}
    lb::T1
    ub::T2
end
function WeightLimits(; lb::Union{<:Real, <:AbstractVector} = 0.0,
                      ub::Union{<:Real, <:AbstractVector} = 1.0)
    if isa(lb, AbstractVector)
        @smart_assert(!isempty(lb))
    end
    if isa(ub, AbstractVector)
        @smart_assert(!isempty(ub))
    end
    @smart_assert(all(lb .<= ub))
    return WeightLimits{typeof(lb), typeof(ub)}(lb, ub)
end
function _w_limit_flag(wl::Real)
    return ifelse(isfinite(wl), true, false)
end
function _w_limit_flag(wl::AbstractVector)
    return ifelse(all(isfinite.(wl)), true, false)
end
function _w_neg_flag(wl::Real)
    return ifelse(wl < zero(wl), true, false)
end
function _w_neg_flag(wl::AbstractVector)
    return ifelse(any(wl .< zero(eltype(wl))), true, false)
end
function set_weight_constraints!(model::JuMP.Model, wl::WeightLimits,
                                 long_only::Bool = false)
    @smart_assert(long_only && _w_neg_flag(wl.lb),
                  "Negative lower weight limits are not allowed when shorting is unavailable.")
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    if _w_limit_flag(wl.lb)
        model[:w_lb] = @constraint(model, sc * w >= sc * k * wl.lb)
    end
    if _w_limit_flag(wl.ub)
        model[:w_ub] = @constraint(model, sc * w <= sc * k * wl.ub)
    end
    return nothing
end

export WeightLimits
