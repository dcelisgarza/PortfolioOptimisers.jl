function _w_bounds_flag(wb::Real)
    return isinf(wb)
end
function _w_bounds_flag(wb::AbstractVector)
    return all(isinf.(wb))
end
function _w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function _w_neg_flag(wb::AbstractVector)
    return any(wb .< zero(eltype(wb)))
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 long_only::Bool = false)
    lb_flag = _w_bounds_flag(wb.lb)
    ub_flag = _w_bounds_flag(wb.ub)
    if lb_flag && ub_flag
        return nothing
    end
    @smart_assert(long_only ⊼ _w_neg_flag(wb.lb),
                  "Long-only strategy cannot have negative weight limits")
    w, k, sc = get_w_k_sc(model)
    if !lb_flag
        @constraint(model, w_lb, sc * w >= sc * k * wb.lb)
    end
    if !ub_flag
        @constraint(model, w_ub, sc * w <= sc * k * wb.ub)
    end
    return nothing
end

export WeightBounds
